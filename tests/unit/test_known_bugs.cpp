// Known defects found by the 2026-10-17 audit. Each test states the correct behaviour and stays DISABLED_ until
// the phase that fixes it (see docs/architecture.md); when a phase fixes one, it moves into the regular test files.
// To see the current failures, run: dlc_unit_tests --gtest_also_run_disabled_tests --gtest_filter='KnownBugs.*'
#include "dlc/backends/llvm/LLVMCodeGen.h"
#include "dlc/ir/Builder.h"
#include "dlc/ir/Parser.h"
#include "dlc/loopir/Lowering.h"
#include "dlc/passes/Autodiff.h"
#include "dlc/passes/CSE.h"
#include "dlc/passes/Canonicalize.h"
#include "dlc/passes/Fusion.h"
#include "dlc/passes/MemoryPlanning.h"
#include <gtest/gtest.h>
#include <algorithm>

namespace {

using namespace dlc;

TypePtr f32(Shape shape) { return std::make_shared<TensorType>(DataType::Float32, std::move(shape)); }

bool overlaps(int64_t aOff, int64_t aSize, int64_t bOff, int64_t bSize) {
    return std::max(aOff, bOff) < std::min(aOff + aSize, bOff + bSize);
}

bool hasOp(Function& func, const std::string& name) {
    const auto& ops = func.getEntryBlock()->getOperations();
    return std::any_of(ops.begin(), ops.end(), [&](const OperationPtr& op) { return op->getOpName() == name; });
}

constexpr const char* kMlpLayer = R"(
func @layer(%x: f32[128,784], %w: f32[784,256], %b: f32[256]) -> f32[128,256] {
  %0 = matmul(%x, %w) : f32[128,256]
  %1 = add(%0, %b) : f32[128,256]
  %2 = relu(%1) : f32[128,256]
  return %2
}
)";

// P2: the ReLU VJP must mask the upstream gradient; v1 computes dOut * x because there is no select op.
TEST(KnownBugs, DISABLED_AutodiffReluBackwardMasksTheGradient) {
    IRBuilder builder;
    auto func = builder.createFunction("ad", {f32({4, 8}), f32({8, 3})}, {f32({})});
    const auto& args = func->getArguments();
    auto relu = builder.createRelu(builder.createMatmul(args[0], args[1]));
    builder.createReturn({builder.createSum(relu)});

    AutodiffPass().runOnFunction(*func);
    EXPECT_TRUE(hasOp(*func, "select"));
}

// P2: with activity analysis only arguments with the param role get gradients; v1 differentiates every argument.
TEST(KnownBugs, DISABLED_AutodiffBuildsGradientsOnlyForParameters) {
    IRBuilder builder;
    auto func = builder.createFunction("ad", {f32({4, 8}), f32({8, 3})}, {f32({})});
    func->setArgRole(1, ArgRole::Param); // arg0 is the input batch, arg1 the weight
    const auto& args = func->getArguments();
    builder.createReturn({builder.createSum(builder.createMatmul(args[0], args[1]))});

    AutodiffPass().runOnFunction(*func);
    const auto& ret = func->getEntryBlock()->getOperations().back();
    ASSERT_EQ(ret->getNumOperands(), 2u); // loss, dW
    EXPECT_EQ(ret->getOperand(1)->getType()->toString(), "f32[8,3]");
}

// P2: identity rewrites must keep the result type; v1 rewrites x[4,4] * 0 into the scalar constant.
TEST(KnownBugs, DISABLED_CanonicalizePreservesResultTypes) {
    IRBuilder builder;
    auto func = builder.createFunction("mul0", {f32({4, 4})}, {f32({4, 4})});
    auto product = builder.createMul(func->getArguments()[0], builder.createConstant(0.0, f32({})));
    builder.createReturn({product});

    CanonicalizePass().runOnFunction(*func);
    EXPECT_EQ(func->getEntryBlock()->getOperations().back()->getOperand(0)->getType()->toString(), "f32[4,4]");
}

// P2: CSE must compare result types; v1 merges 1.0 : f32[] with 1.0 : f32[4,4].
TEST(KnownBugs, DISABLED_CSEKeepsConstantsWithDifferentTypes) {
    IRBuilder builder;
    auto func = builder.createFunction("cse_types", {f32({4, 4})}, {f32({4, 4})});
    auto scalarOne = builder.createConstant(1.0, f32({}));
    auto tensorOne = builder.createConstant(1.0, f32({4, 4}));
    auto product = builder.createMul(builder.createAdd(func->getArguments()[0], scalarOne), tensorOne);
    builder.createReturn({product});

    CSEPass().runOnFunction(*func);
    EXPECT_EQ(product->getDefiningOp()->getOperand(1), tensorOne);
}

// P2: CSE must key doubles exactly; v1 prints them with 6 significant digits, so nearby constants merge.
TEST(KnownBugs, DISABLED_CSEKeepsNearbyConstantsApart) {
    IRBuilder builder;
    auto func = builder.createFunction("cse_values", {f32({4})}, {f32({4})});
    auto c1 = builder.createConstant(0.1000001, f32({4}));
    auto c2 = builder.createConstant(0.1000002, f32({4}));
    auto diff = builder.createSub(builder.createAdd(func->getArguments()[0], c1), c2);
    builder.createReturn({diff});

    CSEPass().runOnFunction(*func);
    EXPECT_EQ(diff->getDefiningOp()->getOperand(1), c2);
}

// P3: liveness is computed on the unfused schedule, so a fused kernel's output lands inside its own inputs.
TEST(KnownBugs, DISABLED_FusedKernelOutputDoesNotAliasItsInputs) {
    auto mod = IRParser::parseModule(kMlpLayer);
    auto& func = *mod->getFunctions()[0];
    FusionPass fusion;
    fusion.runOnFunction(func);
    MemoryPlanningPass memPlan;
    memPlan.runOnFunction(func);

    for (const auto& kernel : loopir::Lowering::lowerFunction(func, &fusion, &memPlan)) {
        for (const auto& out : kernel.getBuffers()) {
            if (!out.isOutput) continue;
            for (const auto& in : kernel.getBuffers()) {
                if (in.isOutput) continue;
                EXPECT_FALSE(overlaps(out.arenaOffsetBytes, out.sizeBytes, in.arenaOffsetBytes, in.sizeBytes))
                    << kernel.getName() << ": output %" << out.name << " overlaps input %" << in.name;
            }
        }
    }
}

// P3: parameters persist across training steps, so no other value may ever be placed on top of them.
TEST(KnownBugs, DISABLED_ParameterBuffersAreNeverReused) {
    auto mod = IRParser::parseModule(kMlpLayer);
    auto& func = *mod->getFunctions()[0];
    MemoryPlanningPass memPlan;
    memPlan.runOnFunction(func);

    const auto& offsets = memPlan.getOffsets();
    for (const auto& arg : func.getArguments()) {
        const auto& a = offsets.at(arg.get());
        for (const auto& op : func.getEntryBlock()->getOperations()) {
            for (const auto& result : op->getResults()) {
                const auto& r = offsets.at(result.get());
                EXPECT_FALSE(overlaps(a.offsetBytes, a.sizeBytes, r.offsetBytes, r.sizeBytes))
                    << "parameter " << arg->toString() << " shares memory with " << op->getOpName();
            }
        }
    }
}

// P4: the LLVM emitter must translate the loop IR; v1 picks a template by kernel name and turns div into fadd.
TEST(KnownBugs, DISABLED_LLVMCodegenLowersDivision) {
    auto mod = IRParser::parseModule(R"(
func @d(%a: f32[64], %b: f32[64]) -> f32[64] {
  %0 = div(%a, %b) : f32[64]
  return %0
}
)");
    auto& func = *mod->getFunctions()[0];
    FusionPass fusion;
    fusion.runOnFunction(func);
    auto kernels = loopir::Lowering::lowerFunction(func, &fusion, nullptr);
    ASSERT_EQ(kernels.size(), 1u);
    EXPECT_NE(backends::llvm::LLVMCodeGen::generateKernelIR(kernels[0]).find("fdiv"), std::string::npos);
}

} // namespace
