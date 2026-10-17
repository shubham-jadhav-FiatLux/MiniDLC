#include "dlc/ir/Builder.h"
#include "dlc/ir/Error.h"
#include "dlc/ir/Parser.h"
#include "dlc/ir/Verifier.h"
#include "dlc/loopir/Lowering.h"
#include "dlc/passes/Activity.h"
#include "dlc/passes/Autodiff.h"
#include "dlc/passes/Optimizer.h"
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

bool hasOp(const Function& func, const std::string& name) {
    const auto& ops = func.getEntryBlock()->getOperations();
    return std::any_of(ops.begin(), ops.end(), [&](const OperationPtr& op) { return op->getOpName() == name; });
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

// The CSE key includes the result type (v1 merged 1.0 : f32[] with 1.0 : f32[4,4]).
TEST(CSE, KeepsConstantsWithDifferentTypes) {
    IRBuilder builder;
    auto func = builder.createFunction("cse_types", {f32({4, 4})}, {f32({4, 4})});
    auto scalarOne = builder.createConstant(1.0, f32({}));
    auto tensorOne = builder.createConstant(1.0, f32({4, 4}));
    auto product = builder.createMul(builder.createAdd(func->getArguments()[0], scalarOne), tensorOne);
    builder.createReturn({product});

    CSEPass().runOnFunction(*func);
    EXPECT_EQ(product->getDefiningOp()->getOperand(1), tensorOne);
}

// Doubles are keyed by their bits (v1 printed 6 significant digits, so nearby constants merged).
TEST(CSE, KeepsNearbyConstantsApart) {
    IRBuilder builder;
    auto func = builder.createFunction("cse_values", {f32({4})}, {f32({4})});
    auto c1 = builder.createConstant(0.1000001, f32({4}));
    auto c2 = builder.createConstant(0.1000002, f32({4}));
    auto diff = builder.createSub(builder.createAdd(func->getArguments()[0], c1), c2);
    builder.createReturn({diff});

    CSEPass().runOnFunction(*func);
    EXPECT_EQ(diff->getDefiningOp()->getOperand(1), c2);
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

// Identity rewrites keep the result type (v1 rewrote x[4,4] * 0 into the scalar constant).
TEST(Canonicalize, PreservesResultTypes) {
    IRBuilder builder;
    auto func = builder.createFunction("mul1", {f32({4})}, {f32({4, 4})});
    auto product = builder.createMul(func->getArguments()[0], builder.createConstant(1.0, f32({4, 4})));
    builder.createReturn({product});

    EXPECT_TRUE(CanonicalizePass().runOnFunction(*func));
    const Operation* out = func->getEntryBlock()->getOperations().back()->getOperand(0)->getDefiningOp();
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->getOpName(), "broadcast_to");
    EXPECT_EQ(out->getResult(0)->getType()->toString(), "f32[4,4]");
    EXPECT_TRUE(Verifier::verify(*func).success);
}

// x * 0 is NaN for x = NaN or Inf, so it is not rewritten to 0.
TEST(Canonicalize, LeavesMultiplicationByZeroAlone) {
    IRBuilder builder;
    auto func = builder.createFunction("mul0", {f32({4})}, {f32({4})});
    builder.createReturn({builder.createMul(func->getArguments()[0], builder.createScalar(0.0))});
    CanonicalizePass().runOnFunction(*func);
    EXPECT_TRUE(hasOp(*func, "mul"));
}

TEST(Canonicalize, ReachesAFixpoint) {
    auto mod = IRParser::parseModule(R"(
func @f(%x: f32[4]) -> f32[4] {
  %one = constant() : f32[] {value = 1.0}
  %zero = constant() : f32[] {value = 0.0}
  %two = constant() : f32[] {value = 2.0}
  %half = constant() : f32[] {value = 0.5}
  %c = mul(%two, %half) : f32[]
  %0 = mul(%x, %c) : f32[4]
  %1 = add(%0, %zero) : f32[4]
  %2 = div(%1, %one) : f32[4]
  return %2
}
)");
    auto& func = *mod->getFunctions()[0];
    EXPECT_TRUE(CanonicalizePass().runOnFunction(func));
    EXPECT_EQ(func.getEntryBlock()->getOperations().back()->getOperand(0), func.getArguments()[0]);
    EXPECT_FALSE(CanonicalizePass().runOnFunction(func));
}

TEST(Canonicalize, FoldsTransposesIntoMatmulFlags) {
    auto mod = IRParser::parseModule(R"(
func @f(%a: f32[8,4], %b: f32[3,8], %q: f32[2,3,5,4]) -> (f32[4,3], f32[2,3,5,5]) {
  %0 = transpose(%a) : f32[4,8] {perm = [1, 0]}
  %1 = transpose(%b) : f32[8,3] {perm = [1, 0]}
  %2 = matmul(%0, %1) : f32[4,3]
  %3 = transpose(%q) : f32[2,3,4,5] {perm = [0, 1, 3, 2]}
  %4 = matmul(%q, %3) : f32[2,3,5,5]
  return %2, %4
}
)");
    PassManager pm;
    pm.setVerifyEach(true);
    pm.addPass(std::make_unique<CanonicalizePass>());
    pm.addPass(std::make_unique<DCEPass>());
    pm.run(*mod);
    std::string text = mod->toString();
    EXPECT_NE(text.find("matmul(%a, %b) : f32[4,3] {transpose_a = 1, transpose_b = 1}"), std::string::npos) << text;
    EXPECT_NE(text.find("matmul(%q, %q) : f32[2,3,5,5] {transpose_b = 1}"), std::string::npos) << text;
    EXPECT_EQ(text.find("transpose("), std::string::npos) << text;
}

TEST(Canonicalize, CollapsesShapeOpChains) {
    auto mod = IRParser::parseModule(R"(
func @f(%x: f32[2,3,4]) -> (f32[2,3,4], f32[24]) {
  %0 = transpose(%x) : f32[4,3,2] {perm = [2, 1, 0]}
  %1 = transpose(%0) : f32[2,3,4] {perm = [2, 1, 0]}
  %2 = reshape(%x) : f32[6,4] {shape = [6, 4]}
  %3 = reshape(%2) : f32[24] {shape = [24]}
  return %1, %3
}
)");
    auto& func = *mod->getFunctions()[0];
    CanonicalizePass().runOnFunction(func);
    DCEPass().runOnFunction(func);
    EXPECT_EQ(func.toString(), "func @f(%x: f32[2,3,4]) -> (f32[2,3,4], f32[24]) {\n"
                               "  %0 = reshape(%x) : f32[24] {shape = [24]}\n"
                               "  return %x, %0\n}\n");
}

TEST(Canonicalize, OrdersCommutativeOperandsForCSE) {
    auto mod = IRParser::parseModule(R"(
func @f(%x: f32[4], %y: f32[4]) -> f32[4] {
  %0 = add(%x, %y) : f32[4]
  %1 = add(%y, %x) : f32[4]
  %2 = mul(%0, %1) : f32[4]
  return %2
}
)");
    auto& func = *mod->getFunctions()[0];
    CanonicalizePass().runOnFunction(func);
    EXPECT_TRUE(CSEPass().runOnFunction(func));
    EXPECT_EQ(func.getEntryBlock()->getOperations().size(), 3u); // add, mul, return
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

// The numerical checks of every VJP are in tests/python/test_cpp_gradients.py; these check the structure.
TEST(Autodiff, ReturnsLossAndParameterGradients) {
    IRBuilder builder;
    auto func = builder.createFunction("test_ad", {f32({128, 784}), f32({784, 256})}, {f32({})});
    func->setArgRole(0, ArgRole::Param);
    func->setArgRole(1, ArgRole::Param);
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
    EXPECT_EQ(func->getResultInfo()[1].kind, ResultKind::Grad);
    EXPECT_EQ(func->getResultInfo()[2].arg, 1u);
    auto res = Verifier::verify(*func);
    EXPECT_TRUE(res.success) << allErrors(res);
}

// The ReLU VJP masks the upstream gradient (v1 multiplied it by the input).
TEST(Autodiff, ReluBackwardMasksTheGradient) {
    IRBuilder builder;
    auto func = builder.createFunction("ad", {f32({4, 8}), f32({8, 3})}, {f32({})});
    func->setArgRole(1, ArgRole::Param);
    const auto& args = func->getArguments();
    auto relu = builder.createRelu(builder.createMatmul(args[0], args[1]));
    builder.createReturn({builder.createSum(relu)});

    AutodiffPass().runOnFunction(*func);
    EXPECT_TRUE(hasOp(*func, "select"));
    EXPECT_TRUE(hasOp(*func, "cmp_gt"));
}

// Activity analysis: only params get gradients, and nothing is built for the input batch.
TEST(Autodiff, BuildsGradientsOnlyForParameters) {
    IRBuilder builder;
    auto func = builder.createFunction("ad", {f32({4, 8}), f32({8, 3})}, {f32({})});
    func->setArgRole(1, ArgRole::Param); // arg0 is the input batch, arg1 the weight
    const auto& args = func->getArguments();
    builder.createReturn({builder.createSum(builder.createMatmul(args[0], args[1]))});

    AutodiffPass().runOnFunction(*func);
    const auto& ret = func->getEntryBlock()->getOperations().back();
    ASSERT_EQ(ret->getNumOperands(), 2u); // loss, dW
    EXPECT_EQ(ret->getOperand(1)->getType()->toString(), "f32[8,3]");
    for (const auto& op : func->getEntryBlock()->getOperations()) {
        for (const auto& result : op->getResults()) {
            EXPECT_NE(result->getType()->toString(), "f32[4,8]") << "a gradient for the input was built";
        }
    }
}

TEST(Autodiff, ParameterThatDoesNotReachTheLossGetsAZeroGradient) {
    auto mod = IRParser::parseModule(R"(
func @f(%x: f32[3] {param}, %unused: f32[2] {param}) -> f32[] {
  %0 = sum(%x) : f32[] {axes = [0], keepdims = 0}
  return %0
}
)");
    auto& func = *mod->getFunctions()[0];
    AutodiffPass().runOnFunction(func);
    const Operation* zero = func.getEntryBlock()->getOperations().back()->getOperand(2)->getDefiningOp();
    ASSERT_NE(zero, nullptr);
    EXPECT_EQ(zero->getOpName(), "constant");
    EXPECT_EQ(zero->getDoubleAttr("value"), 0.0);
    EXPECT_TRUE(Verifier::verify(func).success);
}

TEST(Autodiff, RejectsANonScalarLoss) {
    auto mod = IRParser::parseModule("func @f(%x: f32[3] {param}) -> f32[3] {\n  %0 = exp(%x)\n  return %0\n}\n");
    EXPECT_THROW(AutodiffPass().runOnFunction(*mod->getFunctions()[0]), IRError);
}

TEST(Activity, ActiveValuesDependOnAParameterAndReachTheLoss) {
    auto mod = IRParser::parseModule(R"(
func @f(%x: f32[4], %w: f32[4] {param}, %ids: i32[2], %table: f32[5,4] {param}) -> f32[] {
  %0 = exp(%x) : f32[4]
  %1 = mul(%0, %w) : f32[4]
  %2 = embedding(%table, %ids) : f32[2,4]
  %3 = cmp_gt(%1, %0) : f32[4]
  %4 = mul(%1, %3) : f32[4]
  %5 = sum(%4) : f32[] {axes = [0], keepdims = 0}
  %6 = relu(%w) : f32[4]
  return %5
}
)");
    const auto& func = *mod->getFunctions()[0];
    const auto& ops = func.getEntryBlock()->getOperations();
    ActivityAnalysis activity(func, ops[5]->getResult(0).get());
    EXPECT_FALSE(activity.isActive(ops[0]->getResult(0).get())); // exp(x): no parameter
    EXPECT_TRUE(activity.isActive(ops[1]->getResult(0).get()));
    EXPECT_FALSE(activity.isActive(ops[2]->getResult(0).get())); // embedding: never reaches the loss
    EXPECT_FALSE(activity.isActive(ops[3]->getResult(0).get())); // a comparison carries no derivative
    EXPECT_TRUE(activity.isActive(ops[4]->getResult(0).get()));
    EXPECT_FALSE(activity.isActive(ops[6]->getResult(0).get())); // relu(w): varied but not useful
    EXPECT_FALSE(activity.isActive(func.getArguments()[3].get()));
}

TEST(Optimizer, AdamAddsStateAndDonatesTheUpdates) {
    auto mod = IRParser::parseModule(R"(
func @f(%x: f32[4], %w: f32[4] {param}) -> f32[] {
  %0 = mul(%x, %w) : f32[4]
  %1 = sum(%0) : f32[] {axes = [0], keepdims = 0}
  return %1
}
)");
    auto& func = *mod->getFunctions()[0];
    PassManager pm;
    pm.setVerifyEach(true);
    pm.addPass(std::make_unique<AutodiffPass>());
    OptimizerConfig adam;
    adam.lr = 0.01;
    pm.addPass(std::make_unique<OptimizerPass>(adam));
    ASSERT_NO_THROW(pm.run(*mod));
    std::string text = func.toString();
    for (const char* fragment : {"%adam.beta1_pow: f32[] {state, init = 1.0}", "%w.m: f32[4] {state, init = 0.0}",
                                 "%w.v: f32[4] {state, init = 0.0}", "f32[4] {donates = %w}",
                                 "f32[] {donates = %adam.beta2_pow}", "sqrt("}) {
        EXPECT_NE(text.find(fragment), std::string::npos) << fragment << "\n" << text;
    }
    EXPECT_EQ(text.find("grad ="), std::string::npos) << "gradients are consumed by the update";
    EXPECT_EQ(IRParser::parseModule(mod->toString())->toString(), mod->toString());
}

TEST(Optimizer, SGDWithMomentumKeepsAVelocity) {
    auto mod = IRParser::parseModule(R"(
func @f(%w: f32[3] {param}) -> f32[] {
  %0 = sum(%w) : f32[] {axes = [0], keepdims = 0}
  return %0
}
)");
    auto& func = *mod->getFunctions()[0];
    AutodiffPass().runOnFunction(func);
    OptimizerConfig sgd;
    sgd.kind = OptimizerConfig::Kind::SGD;
    sgd.momentum = 0.9;
    OptimizerPass(sgd).runOnFunction(func);
    auto res = Verifier::verify(func);
    EXPECT_TRUE(res.success) << allErrors(res);
    EXPECT_NE(func.toString().find("%w.velocity: f32[3] {state, init = 0.0}"), std::string::npos) << func.toString();
    EXPECT_EQ(func.getResultInfo().size(), 3u); // loss, w, velocity
}

TEST(Optimizer, NeedsGradients) {
    auto mod = IRParser::parseModule("func @f(%w: f32[3] {param}) -> f32[3] {\n  return %w\n}\n");
    EXPECT_THROW(OptimizerPass(OptimizerConfig{}).runOnFunction(*mod->getFunctions()[0]), IRError);
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

