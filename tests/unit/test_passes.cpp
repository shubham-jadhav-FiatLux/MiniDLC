#include "dlc/ir/Builder.h"
#include "dlc/ir/Error.h"
#include "dlc/ir/Parser.h"
#include "dlc/ir/Verifier.h"
#include "dlc/passes/Activity.h"
#include "dlc/passes/Autodiff.h"
#include "dlc/passes/Optimizer.h"
#include "dlc/passes/CSE.h"
#include "dlc/passes/Canonicalize.h"
#include "dlc/passes/DCE.h"
#include "dlc/passes/Fusion.h"
#include "dlc/passes/BufferAssignment.h"
#include "dlc/passes/CostModel.h"
#include "dlc/passes/ShapeInference.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <set>
#include <unordered_map>
#include <unordered_set>

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

// --- Fusion -----------------------------------------------------------------------------------------------------

constexpr const char* kLayer = R"(
func @layer(%x: f32[128,784], %w: f32[784,256], %b: f32[256]) -> f32[128,256] {
  %0 = matmul(%x, %w) : f32[128,256]
  %1 = add(%0, %b) : f32[128,256]
  %2 = relu(%1) : f32[128,256]
  return %2
}
)";

std::vector<Kernel> kernelsOf(Function& func) {
    FusionPass fusion(/*annotate=*/false);
    fusion.runOnFunction(func);
    return fusion.getKernels();
}

// Every kernel input is an argument or the output of an earlier kernel (looking through reshapes).
void expectTopologicalOrder(const Function& func, const std::vector<Kernel>& kernels) {
    std::unordered_set<const Value*> available;
    for (const auto& arg : func.getArguments()) available.insert(arg.get());
    for (const auto& k : kernels) {
        for (const Value* in : k.inputs) {
            const Value* v = in;
            while (v->getDefiningOp() && v->getDefiningOp()->getOpName() == "reshape") {
                v = v->getDefiningOp()->getOperand(0).get();
            }
            EXPECT_TRUE(available.count(v)) << k.name << " reads a value that no earlier kernel wrote";
        }
        for (const Value* out : k.outputs) available.insert(out);
    }
}

TEST(Fusion, GroupsMatmulBiasReluIntoOneEpilogueKernel) {
    auto mod = IRParser::parseModule(kLayer);
    auto kernels = kernelsOf(*mod->getFunctions()[0]);
    ASSERT_EQ(kernels.size(), 1u);
    EXPECT_EQ(kernels[0].kind, KernelKind::Matmul);
    EXPECT_EQ(kernels[0].name, "k0_matmul_add_relu");
    EXPECT_EQ(kernels[0].ops.size(), 3u);
    EXPECT_EQ(kernels[0].inputs.size(), 3u);  // x, w, b
    EXPECT_EQ(kernels[0].outputs.size(), 1u); // only the relu leaves the kernel
}

TEST(Fusion, ElementwiseKernelsWriteEveryValueNeededOutside) {
    auto mod = IRParser::parseModule(R"(
func @f(%x: f32[4,8], %y: f32[4,8]) -> (f32[4,8], f32[4]) {
  %0 = mul(%x, %y) : f32[4,8]
  %1 = exp(%0) : f32[4,8]
  %2 = add(%1, %x) : f32[4,8]
  %3 = sum(%1) : f32[4] {axes = [1]}
  return %2, %3
}
)");
    auto& func = *mod->getFunctions()[0];
    auto kernels = kernelsOf(func);
    ASSERT_EQ(kernels.size(), 2u);
    EXPECT_EQ(kernels[0].kind, KernelKind::Elementwise);
    EXPECT_EQ(kernels[0].ops.size(), 3u);
    EXPECT_EQ(kernels[0].outputs.size(), 2u); // %1 for the reduction, %2 for the return
    EXPECT_EQ(kernels[1].kind, KernelKind::Reduction);
    expectTopologicalOrder(func, kernels);
}

// exp(x) → sum → mul(exp(x), sum): merging the mul into the exp's kernel would need the sum's result before the sum
// can run, a cycle in the kernel graph.
TEST(Fusion, RejectsMergesThatMakeTheKernelGraphCyclic) {
    auto mod = IRParser::parseModule(R"(
func @f(%x: f32[4]) -> f32[4] {
  %0 = exp(%x) : f32[4]
  %1 = sum(%0) : f32[] {axes = [0]}
  %2 = mul(%0, %1) : f32[4]
  return %2
}
)");
    auto& func = *mod->getFunctions()[0];
    auto kernels = kernelsOf(func);
    ASSERT_EQ(kernels.size(), 3u);
    EXPECT_EQ(kernels[0].ops[0]->getOpName(), "exp");
    EXPECT_EQ(kernels[1].ops[0]->getOpName(), "sum");
    EXPECT_EQ(kernels[2].ops[0]->getOpName(), "mul");
    expectTopologicalOrder(func, kernels);
}

// A matmul kernel only absorbs ops after the matmul: exp(%x) feeds the matmul, so it stays a kernel of its own even
// though the epilogue add reads it too.
TEST(Fusion, MatmulEpiloguesNeverFeedTheMatmul) {
    auto mod = IRParser::parseModule(R"(
func @f(%x: f32[4,4], %w: f32[4,4]) -> f32[4,4] {
  %0 = exp(%x) : f32[4,4]
  %1 = matmul(%0, %w) : f32[4,4]
  %2 = add(%1, %0) : f32[4,4]
  return %2
}
)");
    auto& func = *mod->getFunctions()[0];
    auto kernels = kernelsOf(func);
    ASSERT_EQ(kernels.size(), 2u);
    EXPECT_EQ(kernels[0].kind, KernelKind::Elementwise);
    EXPECT_EQ(kernels[1].kind, KernelKind::Matmul);
    EXPECT_EQ(kernels[1].ops.size(), 2u);
    expectTopologicalOrder(func, kernels);
}

TEST(Fusion, InlinesSplatsIntoElementwiseCodeAndFillsTheRest) {
    auto mod = IRParser::parseModule(R"(
func @f(%x: f32[4,4]) -> f32[4,4] {
  %c = constant() : f32[4,4] {value = 0.5}
  %0 = mul(%x, %c) : f32[4,4]
  %1 = matmul(%0, %c) : f32[4,4]
  return %1
}
)");
    auto& func = *mod->getFunctions()[0];
    auto kernels = kernelsOf(func);
    ASSERT_EQ(kernels.size(), 3u);
    EXPECT_EQ(kernels[0].kind, KernelKind::Fill); // the matmul reads %c from memory
    for (const auto& k : kernels) {
        if (k.kind == KernelKind::Elementwise) {
            EXPECT_EQ(k.inputs.size(), 1u) << "the splat is inlined";
        }
    }
    expectTopologicalOrder(func, kernels);
}

TEST(CostModel, CountsFlopsAndTheTrafficFusionSaves) {
    auto mod = IRParser::parseModule(kLayer);
    auto& func = *mod->getFunctions()[0];
    auto kernels = kernelsOf(func);
    Cost c = kernelCost(kernels[0]);
    EXPECT_DOUBLE_EQ(c.flops, 2.0 * 128 * 256 * 784 + 2.0 * 128 * 256);
    EXPECT_EQ(c.bytesRead, (128 * 784 + 784 * 256 + 256) * 4);
    EXPECT_EQ(c.bytesWritten, 128 * 256 * 4); // the matmul and add results never reach memory
    std::string report = costReport(func, kernels);
    EXPECT_NE(report.find("k0_matmul_add_relu"), std::string::npos) << report;
    EXPECT_NE(report.find("op by op"), std::string::npos) << report;
}

// --- Buffer assignment ------------------------------------------------------------------------------------------

BufferAssignment assign(Function& func, std::vector<Kernel>& kernels) {
    kernels = kernelsOf(func);
    BufferAssignment a(func, kernels);
    a.verify();
    return a;
}

TEST(BufferAssignment, LiveBuffersNeverOverlap) {
    auto mod = IRParser::parseModule(R"(
func @f(%x: f32[64,64], %y: f32[64,64]) -> (f32[64,64], f32[64]) {
  %0 = add(%x, %y) : f32[64,64]
  %1 = sum(%0) : f32[64] {axes = [1]}
  %2 = transpose(%0) : f32[64,64]
  %3 = matmul(%2, %x) : f32[64,64]
  %4 = sum(%3) : f32[64] {axes = [0]}
  %5 = softmax(%3) : f32[64,64]
  %6 = mul(%5, %y) : f32[64,64]
  %7 = add(%4, %1) : f32[64]
  return %6, %7
}
)");
    auto& func = *mod->getFunctions()[0];
    std::vector<Kernel> kernels;
    BufferAssignment a = assign(func, kernels);
    const auto& buffers = a.buffers();
    for (size_t i = 0; i < buffers.size(); ++i) {
        for (size_t j = i + 1; j < buffers.size(); ++j) {
            const Buffer& p = buffers[i];
            const Buffer& q = buffers[j];
            bool live = p.begin <= q.end && q.begin <= p.end;
            bool overlap = p.offset < q.offset + q.size && q.offset < p.offset + p.size;
            EXPECT_FALSE(p.region == q.region && live && overlap);
        }
        EXPECT_EQ(buffers[i].offset % BufferAssignment::kAlignment, 0);
    }
    EXPECT_LT(a.transientSize(), a.transientUnshared()); // liveness lets some buffers share memory
}

// v1 planned memory on the unfused op schedule, so a fused kernel's output landed inside its own inputs.
TEST(BufferAssignment, FusedKernelOutputDoesNotAliasItsInputs) {
    auto mod = IRParser::parseModule(kLayer);
    auto& func = *mod->getFunctions()[0];
    std::vector<Kernel> kernels;
    BufferAssignment a = assign(func, kernels);
    const Buffer& out = a.bufferOf(kernels[0].outputs[0]);
    for (const Value* in : kernels[0].inputs) {
        const Buffer& b = a.bufferOf(in);
        if (b.region != out.region) continue;
        EXPECT_FALSE(out.offset < b.offset + b.size && b.offset < out.offset + out.size);
    }
}

// Parameters persist across training steps, so nothing else may ever be placed on top of them (v1 reused them).
TEST(BufferAssignment, ParametersLiveInThePersistentRegion) {
    auto mod = IRParser::parseModule(R"(
func @f(%x: f32[8,4], %w: f32[4,3] {param}, %b: f32[3] {param}, %mask: f32[8,3] {const}) -> f32[8,3] {
  %0 = matmul(%x, %w) : f32[8,3]
  %1 = add(%0, %b) : f32[8,3]
  %2 = mul(%1, %mask) : f32[8,3]
  return %2
}
)");
    auto& func = *mod->getFunctions()[0];
    std::vector<Kernel> kernels;
    BufferAssignment a = assign(func, kernels);
    for (size_t i = 1; i < 4; ++i) EXPECT_EQ(a.bufferOf(func.getArguments()[i].get()).region, Region::Persistent);
    EXPECT_EQ(a.bufferOf(func.getArguments()[0].get()).region, Region::Transient);
    EXPECT_GE(a.persistentSize(), (12 + 3 + 24) * 4);
}

TEST(BufferAssignment, UpdatesParametersAndStateInPlace) {
    auto mod = IRParser::parseModule(R"(
func @f(%x: f32[8,4], %w: f32[4,3] {param}) -> f32[] {
  %0 = matmul(%x, %w) : f32[8,3]
  %1 = mean(%0) : f32[]
  return %1
}
)");
    PassManager pm;
    pm.setVerifyEach(true);
    pm.addPass(std::make_unique<AutodiffPass>());
    pm.addPass(std::make_unique<OptimizerPass>(OptimizerConfig{}));
    pm.addPass(std::make_unique<CanonicalizePass>());
    pm.addPass(std::make_unique<CSEPass>());
    pm.addPass(std::make_unique<DCEPass>());
    pm.run(*mod);
    auto& func = *mod->getFunctions()[0];
    std::vector<Kernel> kernels;
    BufferAssignment a = assign(func, kernels);
    const Operation* ret = func.getEntryBlock()->getOperations().back().get();
    size_t donated = 0;
    for (size_t i = 0; i < ret->getNumOperands(); ++i) {
        if (func.getResultInfo()[i].kind != ResultKind::Donates) continue;
        ++donated;
        EXPECT_TRUE(a.updatesInPlace(ret->getOperand(i).get())) << "result " << i;
        const Value* arg = func.getArguments()[func.getResultInfo()[i].arg].get();
        EXPECT_EQ(&a.bufferOf(ret->getOperand(i).get()), &a.bufferOf(arg));
    }
    EXPECT_EQ(donated, 5u); // w, w.m, w.v and the two bias-correction powers
    EXPECT_TRUE(a.finalCopies().empty());
}

// %w is read by the sum after the kernel computing its update has run, so the update cannot overwrite %w in place.
TEST(BufferAssignment, CopiesAnUpdateThatCannotBeInPlace) {
    auto mod = IRParser::parseModule(R"(
func @f(%w: f32[4] {param}, %x: f32[4]) -> (f32[4] {donates = %w}, f32[]) {
  %0 = add(%w, %x) : f32[4]
  %1 = sum(%w) : f32[] {axes = [0]}
  return %0, %1
}
)");
    auto& func = *mod->getFunctions()[0];
    std::vector<Kernel> kernels;
    BufferAssignment a = assign(func, kernels);
    const Value* update = func.getEntryBlock()->getOperations()[0]->getResult(0).get();
    EXPECT_FALSE(a.updatesInPlace(update));
    ASSERT_EQ(a.finalCopies().size(), 1u);
    EXPECT_EQ(a.finalCopies()[0].from, update);
}

// The sum reads the update after it is written into %w's buffer: that is the new value, so the update stays in place
// (the checker once mistook it for a read of the overwritten %w; unfused Adam steps hit it).
TEST(BufferAssignment, LaterReadersOfTheUpdatedValueKeepItInPlace) {
    auto mod = IRParser::parseModule(R"(
func @f(%w: f32[4] {param}, %x: f32[4]) -> (f32[4] {donates = %w}, f32[]) {
  %0 = add(%w, %x) : f32[4]
  %1 = sum(%0) : f32[] {axes = [0]}
  return %0, %1
}
)");
    auto& func = *mod->getFunctions()[0];
    std::vector<Kernel> kernels;
    BufferAssignment a = assign(func, kernels);
    EXPECT_TRUE(a.updatesInPlace(func.getEntryBlock()->getOperations()[0]->getResult(0).get()));
    EXPECT_NO_THROW(a.verify());
}

TEST(BufferAssignment, ReshapesAliasTheirOperand) {
    auto mod = IRParser::parseModule(R"(
func @f(%x: f32[4,6]) -> f32[6] {
  %0 = exp(%x) : f32[4,6]
  %1 = reshape(%0) : f32[24] {shape = [24]}
  %2 = reshape(%1) : f32[6,4] {shape = [6, 4]}
  %3 = sum(%2) : f32[6] {axes = [1]}
  return %3
}
)");
    auto& func = *mod->getFunctions()[0];
    std::vector<Kernel> kernels;
    BufferAssignment a = assign(func, kernels);
    const auto& ops = func.getEntryBlock()->getOperations();
    EXPECT_EQ(&a.bufferOf(ops[2]->getResult(0).get()), &a.bufferOf(ops[0]->getResult(0).get()));
    // The alias keeps exp's buffer alive until the sum reads it.
    EXPECT_EQ(a.bufferOf(ops[0]->getResult(0).get()).end, 1);
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
    pm.addPass(std::make_unique<BufferAssignmentPass>());
    ASSERT_NO_THROW(pm.run(*mod));

    // The annotations the passes add print and parse back.
    std::string text = mod->toString();
    EXPECT_NE(text.find("fusion.group"), std::string::npos) << text;
    EXPECT_NE(text.find("mem.offset"), std::string::npos) << text;
    EXPECT_EQ(IRParser::parseModule(text)->toString(), text);
}

} // namespace

