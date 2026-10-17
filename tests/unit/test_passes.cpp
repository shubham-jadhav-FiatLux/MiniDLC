#include "dlc/ir/Builder.h"
#include "dlc/ir/Error.h"
#include "dlc/ir/Parser.h"
#include "dlc/ir/Verifier.h"
#include "dlc/loopir/Lowering.h"
#include "dlc/passes/Autodiff.h"
#include "dlc/passes/CSE.h"
#include "dlc/passes/Canonicalize.h"
#include "dlc/passes/DCE.h"
#include "dlc/passes/Fusion.h"
#include "dlc/passes/MemoryPlanning.h"
#include "dlc/passes/ShapeInference.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <set>
#include <unordered_map>

namespace {

using namespace dlc;

TypePtr f32(Shape shape) { return std::make_shared<TensorType>(DataType::Float32, std::move(shape)); }

std::vector<std::string> opNames(Function& func) {
    std::vector<std::string> names;
    for (const auto& op : func.getEntryBlock()->getOperations()) names.push_back(op->getOpName());
    return names;
}

std::string allErrors(const VerificationResult& res) {
    std::string text;
    for (const auto& err : res.errors) text += err + "\n";
    return text;
}

constexpr const char* kTwoLayerMlp = R"(
func @mlp(%x: f32[32,64], %w1: f32[64,128] {param}, %b1: f32[128] {param}, %w2: f32[128,10] {param},
          %b2: f32[10] {param}, %y: i32[32]) -> f32[] {
  %h = matmul(%x, %w1) : f32[32,128]
  %hb = add(%h, %b1) : f32[32,128]
  %a = relu(%hb) : f32[32,128]
  %o = matmul(%a, %w2) : f32[32,10]
  %ob = add(%o, %b2) : f32[32,10]
  %loss = cross_entropy(%ob, %y) : f32[]
  return %loss
}
)";

// Gives the first op a wrong result type, as a buggy pass might.
class BreakTypesPass : public Pass {
public:
    std::string getName() const override { return "break-types"; }
    bool runOnFunction(Function& func) override {
        func.getEntryBlock()->getOperations().front()->getResult(0)->setType(f32({1}));
        return true;
    }
};

class ThrowingPass : public Pass {
public:
    std::string getName() const override { return "throwing"; }
    bool runOnFunction(Function&) override { throw IRError("cannot handle this op"); }
};

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
    EXPECT_TRUE(Verifier::verify(*func).success);
}

TEST(DCE, UnlinksErasedOpsFromTheirOperands) {
    IRBuilder builder;
    auto func = builder.createFunction("dce", {f32({4})}, {f32({4})});
    auto x = func->getArguments()[0];
    auto live = builder.createExp(x);
    builder.createRelu(builder.createLog(x)); // a dead chain: relu uses log
    builder.createReturn({live});

    EXPECT_TRUE(DCEPass().runOnFunction(*func));
    EXPECT_EQ(opNames(*func), (std::vector<std::string>{"exp", "return"}));
    EXPECT_EQ(x->getUses().size(), 1u);
    EXPECT_TRUE(Verifier::verify(*func).success);
    EXPECT_FALSE(DCEPass().runOnFunction(*func));
}

TEST(CSE, ErasedDuplicatesLeaveConsistentUseLists) {
    auto mod = IRParser::parseModule(R"(
func @f(%x: f32[8], %y: f32[8]) -> f32[8] {
  %a = mul(%x, %y)
  %b = mul(%x, %y)
  %c = mul(%x, %y)
  %s = add(%a, %b)
  %t = add(%s, %c)
  return %t
}
)");
    auto& func = *mod->getFunctions()[0];
    EXPECT_TRUE(CSEPass().runOnFunction(func));
    EXPECT_EQ(opNames(func), (std::vector<std::string>{"mul", "add", "add", "return"}));
    auto res = Verifier::verify(func);
    EXPECT_TRUE(res.success) << allErrors(res);
    EXPECT_EQ(func.getArguments()[0]->getUses().size(), 1u);
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

// Folding inserts ops into the block it is walking; it must not invalidate that walk (it was a use-after-free).
TEST(Canonicalize, FoldsConstantsSafely) {
    IRBuilder builder;
    auto func = builder.createFunction("fold", {}, {f32({})});
    auto sum = builder.createAdd(builder.createConstant(2.0, f32({})), builder.createConstant(3.0, f32({})));
    builder.createReturn({sum});

    EXPECT_TRUE(CanonicalizePass().runOnFunction(*func));
    const Operation* folded = func->getEntryBlock()->getOperations().back()->getOperand(0)->getDefiningOp();
    ASSERT_EQ(folded->getOpName(), "constant");
    EXPECT_DOUBLE_EQ(folded->getDoubleAttr("value"), 5.0);
    EXPECT_TRUE(Verifier::verify(*func).success);
}

// Parsed IR has generic ops rather than the builder's typed subclasses; patterns must match it too.
TEST(Canonicalize, FoldsParsedConstants) {
    auto mod = IRParser::parseModule(R"(
func @f() -> f32[2] {
  %a = constant() : f32[2] {value = 6.0}
  %b = constant() : f32[] {value = 1.5}
  %c = div(%a, %b)
  %d = sub(%c, %b)
  return %d
}
)");
    auto& func = *mod->getFunctions()[0];
    EXPECT_TRUE(CanonicalizePass().runOnFunction(func));
    EXPECT_TRUE(DCEPass().runOnFunction(func));
    auto res = Verifier::verify(func);
    EXPECT_TRUE(res.success) << allErrors(res);
    EXPECT_EQ(func.toString(), "func @f() -> f32[2] {\n  %0 = constant() : f32[2] {value = 2.5}\n  return %0\n}\n");
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
    EXPECT_EQ(func->getResultTypes().size(), 3u);
    auto res = Verifier::verify(*func);
    EXPECT_TRUE(res.success) << allErrors(res);
}

TEST(Autodiff, OutputOfATrainingGraphVerifies) {
    auto mod = IRParser::parseModule(kTwoLayerMlp);
    PassManager pm;
    pm.setVerifyEach(true);
    pm.addPass(std::make_unique<AutodiffPass>());
    pm.addPass(std::make_unique<CSEPass>());
    pm.addPass(std::make_unique<DCEPass>());
    ASSERT_NO_THROW(pm.run(*mod));
    const auto& func = *mod->getFunctions()[0];
    EXPECT_GT(func.getResultTypes().size(), 1u);
    EXPECT_EQ(func.getResultTypes()[0]->toString(), "f32[]");
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

TEST(PassManager, VerifiesAfterEachPassAndNamesTheCulprit) {
    auto mod = IRParser::parseModule(kTwoLayerMlp);
    PassManager pm;
    pm.setVerifyEach(true);
    pm.addPass(std::make_unique<DCEPass>());
    pm.addPass(std::make_unique<BreakTypesPass>());
    pm.addPass(std::make_unique<CSEPass>());
    try {
        pm.run(*mod);
        ADD_FAILURE() << "the broken IR went unnoticed";
    } catch (const IRError& e) {
        std::string message = e.what();
        EXPECT_EQ(message.rfind("verification failed after pass 'break-types':", 0), 0u) << message;
        EXPECT_NE(message.find("[@mlp] '%0 = matmul(%x, %w1) : f32[1]': result type f32[1] does not match the "
                               "inferred type f32[32,128]"),
                  std::string::npos)
            << message;
    }
}

TEST(PassManager, PrefixesPassErrorsWithThePassName) {
    auto mod = IRParser::parseModule(kTwoLayerMlp);
    PassManager pm;
    pm.addPass(std::make_unique<ThrowingPass>());
    try {
        pm.run(*mod);
        ADD_FAILURE() << "the pass error was swallowed";
    } catch (const IRError& e) {
        EXPECT_STREQ(e.what(), "pass 'throwing': cannot handle this op");
    }
}

TEST(PassManager, FullPipelineKeepsTheModuleValid) {
    auto mod = IRParser::parseModule(kTwoLayerMlp);
    PassManager pm;
    pm.setVerifyEach(true);
    pm.addPass(std::make_unique<ShapeInferencePass>());
    pm.addPass(std::make_unique<CanonicalizePass>());
    pm.addPass(std::make_unique<CSEPass>());
    pm.addPass(std::make_unique<DCEPass>());
    pm.addPass(std::make_unique<FusionPass>());
    pm.addPass(std::make_unique<MemoryPlanningPass>());
    ASSERT_NO_THROW(pm.run(*mod));

    // The annotations the passes add print and parse back.
    std::string text = mod->toString();
    EXPECT_NE(text.find("fusion.group"), std::string::npos) << text;
    EXPECT_NE(text.find("mem.offset"), std::string::npos) << text;
    EXPECT_EQ(IRParser::parseModule(text)->toString(), text);
}

// Kernel buffers are named after the values they hold; unnamed builder values used to all print as '%'.
TEST(Lowering, BufferNamesAreUniqueAndNonEmpty) {
    IRBuilder builder;
    auto func = builder.createFunction("names", {f32({16, 32}), f32({32, 8}), f32({8})}, {f32({16, 8})});
    const auto& args = func->getArguments();
    for (const auto& arg : args) arg->setName("");
    auto mm = builder.createMatmul(args[0], args[1]);
    auto biased = builder.createAdd(mm, args[2]);
    auto act = builder.createExp(builder.createRelu(biased));
    builder.createReturn({builder.createMul(act, act)});

    FusionPass fusion(/*annotate=*/false);
    fusion.runOnFunction(*func);
    MemoryPlanningPass memPlan(/*annotate=*/false);
    memPlan.runOnFunction(*func);
    auto kernels = loopir::Lowering::lowerFunction(*func, &fusion, &memPlan);
    ASSERT_FALSE(kernels.empty());
    for (const auto& kernel : kernels) {
        std::set<std::string> names;
        for (const auto& buffer : kernel.getBuffers()) {
            EXPECT_FALSE(buffer.name.empty()) << kernel.getName();
            EXPECT_TRUE(names.insert(buffer.name).second) << kernel.getName() << " repeats buffer " << buffer.name;
        }
    }
}

} // namespace

