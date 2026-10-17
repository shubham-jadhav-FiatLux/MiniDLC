#include "dlc/ir/Builder.h"
#include "dlc/passes/Autodiff.h"
#include "dlc/passes/CSE.h"
#include "dlc/passes/Canonicalize.h"
#include "dlc/passes/DCE.h"
#include "dlc/passes/Fusion.h"
#include "dlc/passes/MemoryPlanning.h"
#include "dlc/passes/ShapeInference.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <unordered_map>

namespace {

using namespace dlc;

TypePtr f32(Shape shape) { return std::make_shared<TensorType>(DataType::Float32, std::move(shape)); }

std::vector<std::string> opNames(Function& func) {
    std::vector<std::string> names;
    for (const auto& op : func.getEntryBlock()->getOperations()) names.push_back(op->getOpName());
    return names;
}

TEST(ShapeInference, BroadcastsTrailingDimensions) {
    Shape out;
    ASSERT_TRUE(ShapeInferencePass::broadcastShapes({128, 1}, {256}, out));
    EXPECT_EQ(out, (Shape{128, 256}));
}

TEST(ShapeInference, RejectsIncompatibleBroadcast) {
    Shape out;
    EXPECT_FALSE(ShapeInferencePass::broadcastShapes({128, 3}, {256}, out));
}

TEST(ShapeInference, CorrectsWrongResultTypes) {
    IRBuilder builder;
    auto func = builder.createFunction("infer", {f32({128, 784}), f32({784, 256}), f32({256})}, {f32({128, 256})});
    const auto& args = func->getArguments();
    auto mm = builder.createMatmul(args[0], args[1], f32({1}));
    auto add = builder.createAdd(mm, args[2], f32({1}));
    builder.createReturn({add});

    EXPECT_TRUE(ShapeInferencePass().runOnFunction(*func));
    EXPECT_EQ(mm->getType()->toString(), "f32[128,256]");
    EXPECT_EQ(add->getType()->toString(), "f32[128,256]");
}

TEST(CSEAndDCE, MergesDuplicatesAndRemovesDeadOps) {
    IRBuilder builder;
    auto func = builder.createFunction("test_cse", {f32({128}), f32({128})}, {f32({128})});
    const auto& args = func->getArguments();

    auto sum1 = builder.createAdd(args[0], args[1]);
    auto sum2 = builder.createAdd(args[0], args[1]); // duplicate of sum1
    builder.createMul(args[0], args[1]);             // dead
    auto out = builder.createRelu(sum2);
    builder.createReturn({out});

    EXPECT_TRUE(CSEPass().runOnFunction(*func));
    EXPECT_EQ(out->getDefiningOp()->getOperand(0), sum1);

    EXPECT_TRUE(DCEPass().runOnFunction(*func));
    EXPECT_EQ(opNames(*func), (std::vector<std::string>{"add", "relu", "return"}));
}

TEST(Canonicalize, ReplacesAddOfZeroWithItsOperand) {
    IRBuilder builder;
    auto func = builder.createFunction("test_canon", {f32({128})}, {f32({128})});
    auto x = func->getArguments()[0];

    auto zero = builder.createConstant(0.0, f32({128}));
    auto added = builder.createAdd(x, zero);
    builder.createReturn({added});

    EXPECT_TRUE(CanonicalizePass().runOnFunction(*func));
    EXPECT_EQ(func->getEntryBlock()->getOperations().back()->getOperand(0), x);
}

// Structure only: the v1 VJPs for relu/gelu/mean/layernorm are numerically wrong; see test_known_bugs.cpp (P2).
TEST(Autodiff, ReturnsLossAndArgumentGradients) {
    IRBuilder builder;
    auto func = builder.createFunction("test_ad", {f32({128, 784}), f32({784, 256})}, {f32({})});
    const auto& args = func->getArguments();

    auto mm = builder.createMatmul(args[0], args[1]);
    auto relu = builder.createRelu(mm);
    auto loss = builder.createSum(relu);
    builder.createReturn({loss});

    EXPECT_TRUE(AutodiffPass().runOnFunction(*func));
    const auto& ret = func->getEntryBlock()->getOperations().back();
    ASSERT_EQ(ret->getOpName(), "return");
    ASSERT_EQ(ret->getNumOperands(), 3u);
    EXPECT_EQ(ret->getOperand(0), loss);
    EXPECT_EQ(ret->getOperand(1)->getType()->toString(), "f32[128,784]");
    EXPECT_EQ(ret->getOperand(2)->getType()->toString(), "f32[784,256]");
}

TEST(Fusion, GroupsMatmulBiasReluIntoOneEpilogueKernel) {
    IRBuilder builder;
    auto func = builder.createFunction("test_fuse", {f32({128, 784}), f32({784, 256}), f32({256})},
                                       {f32({128, 256})});
    const auto& args = func->getArguments();

    auto mm = builder.createMatmul(args[0], args[1]);
    auto add = builder.createAdd(mm, args[2]);
    auto relu = builder.createRelu(add);
    builder.createReturn({relu});

    FusionPass fusion;
    EXPECT_TRUE(fusion.runOnFunction(*func));
    ASSERT_FALSE(fusion.getFusedGroups().empty());
    const auto& group = fusion.getFusedGroups()[0];
    EXPECT_EQ(group.kind, FusedGroupKind::MatmulEpilogue);
    EXPECT_EQ(group.name, "fused_matmul_bias_relu");
    EXPECT_EQ(group.operations.size(), 3u);
}

// Op-by-op schedule: values whose live ranges overlap must get disjoint arena ranges.
TEST(MemoryPlanning, LiveValuesNeverShareMemory) {
    IRBuilder builder;
    auto func = builder.createFunction("test_mem", {f32({128, 256}), f32({128, 256})}, {f32({128, 256})});
    const auto& args = func->getArguments();

    auto t1 = builder.createAdd(args[0], args[1]);
    auto r1 = builder.createRelu(t1);
    auto t2 = builder.createMul(r1, args[0]);
    auto r2 = builder.createRelu(t2);
    builder.createReturn({r2});

    MemoryPlanningPass memPlan;
    ASSERT_TRUE(memPlan.runOnFunction(*func));
    EXPECT_GT(memPlan.getTotalArenaSize(), 0);
    EXPECT_LT(memPlan.getTotalArenaSize(), memPlan.getUnoptimizedSize());

    // Live range = [defining step, last use]; arguments are defined at step 0, return keeps values live to the end.
    const auto& ops = func->getEntryBlock()->getOperations();
    std::unordered_map<Value*, std::pair<int64_t, int64_t>> live;
    for (const auto& arg : args) live[arg.get()] = {0, 0};
    for (int64_t step = 0; step < static_cast<int64_t>(ops.size()); ++step) {
        for (const auto& operand : ops[step]->getOperands()) {
            auto end = ops[step]->getOpName() == "return" ? static_cast<int64_t>(ops.size()) : step;
            live[operand.get()].second = std::max(live[operand.get()].second, end);
        }
        for (const auto& result : ops[step]->getResults()) live[result.get()] = {step, step};
    }

    const auto& offsets = memPlan.getOffsets();
    for (const auto& [a, ra] : live) {
        for (const auto& [b, rb] : live) {
            if (a == b || std::max(ra.first, rb.first) > std::min(ra.second, rb.second)) continue;
            const auto& oa = offsets.at(a);
            const auto& ob = offsets.at(b);
            EXPECT_GE(std::max(oa.offsetBytes, ob.offsetBytes),
                      std::min(oa.offsetBytes + oa.sizeBytes, ob.offsetBytes + ob.sizeBytes))
                << "live values share arena bytes";
        }
    }
}

} // namespace
