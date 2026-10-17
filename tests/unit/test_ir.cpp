#include "dlc/ir/Builder.h"
#include "dlc/ir/Error.h"
#include "dlc/ir/OpRegistry.h"
#include "dlc/ir/Parser.h"
#include "dlc/ir/SlotTracker.h"
#include "dlc/ir/Verifier.h"
#include "dlc/passes/Autodiff.h"
#include "dlc/passes/CSE.h"
#include "dlc/passes/Canonicalize.h"
#include "dlc/passes/DCE.h"
#include "dlc/passes/Fusion.h"
#include "dlc/passes/MemoryPlanning.h"
#include "dlc/passes/ShapeInference.h"
#include <gtest/gtest.h>
#include <cctype>
#include <iostream>
#include <random>
#include <regex>
#include <set>
#include <sstream>

namespace {

using namespace dlc;

TypePtr f32(Shape shape) { return std::make_shared<TensorType>(DataType::Float32, std::move(shape)); }
TypePtr i32(Shape shape) { return std::make_shared<TensorType>(DataType::Int32, std::move(shape)); }

bool hasError(const VerificationResult& res, const std::string& fragment) {
    for (const auto& err : res.errors) {
        if (err.find(fragment) != std::string::npos) return true;
    }
    return false;
}

std::string allErrors(const VerificationResult& res) {
    std::string text;
    for (const auto& err : res.errors) text += err + "\n";
    return text;
}

// Uses every registered op, argument roles, multiple results, annotations and special float values.
constexpr const char* kEveryOp = R"(
func @everything(%x: f32[2,3,4], %w: f32[4,5] {param}, %gamma: f32[5] {param}, %beta: f32[5] {param},
                 %table: f32[10,5] {param}, %ids: i32[2,3], %targets: i32[2,3], %mask: f32[3,5] {const},
                 %q: f32[2,2,3,4] {state}, %k: f32[2,2,4,3], %bw: f32[2,5,6]) -> (f32[], f32[2,3,5], f32[2,2,3,3]) {
  %mm = matmul(%x, %w) : f32[2,3,5]
  %emb = embedding(%table, %ids) : f32[2,3,5]
  %sum = add(%mm, %emb) : f32[2,3,5] {fusion.group = 0}
  %masked = mul(%sum, %mask) : f32[2,3,5]
  %half = constant() : f32[] {value = 0.5}
  %tiny = constant() : f32[] {value = 1e-30}
  %big = constant() : f32[3,5] {value = -inf}
  %nan = constant() : f32[] {value = nan}
  %scaled = div(%masked, %half)
  %ln = layernorm(%scaled, %gamma, %beta) : f32[2,3,5] {eps = 0.001}
  %g = gelu(%ln) : f32[2,3,5]
  %s = softmax(%g) : f32[2,3,5] {axis = 1}
  %ls = log_softmax(%s) : f32[2,3,5]
  %e = exp(%ls) : f32[2,3,5]
  %r = relu(%e) : f32[2,3,5]
  %d = sub(%r, %g) : f32[2,3,5]
  %t = transpose(%d) : f32[5,3,2]
  %rs = reshape(%t) : f32[15,2] {shape = [15, 2]}
  %red = sum(%rs) : f32[2] {axes = [0]}
  %avg = mean(%d) : f32[1,1,5] {axes = [0, 1], keepdims = 1}
  %lg = log(%avg) : f32[1,1,5]
  %oh = onehot(%targets) : f32[2,3,5] {num_classes = 5}
  %bmm = matmul(%d, %bw) : f32[2,3,6]
  %att = matmul(%q, %k) : f32[2,2,3,3] {mem.offset = 128}
  %loss = cross_entropy(%d, %targets) : f32[]
  return %loss, %mm, %att
}
)";

TEST(IRTypes, TensorTypeBasics) {
    TensorType t(DataType::Float32, {128, 784});
    EXPECT_EQ(t.getRank(), 2);
    EXPECT_EQ(t.getNumElements(), 128 * 784);
    EXPECT_EQ(t.getSizeBytes(), 128 * 784 * 4);
    EXPECT_EQ(t.toString(), "f32[128,784]");
}

TEST(IRBuilder, BuildsVerifiableMlpLayer) {
    IRBuilder builder;
    auto func = builder.createFunction("test_mlp_layer", {f32({128, 784}), f32({784, 256}), f32({256})},
                                       {f32({128, 256})});
    const auto& args = func->getArguments();

    auto mm = builder.createMatmul(args[0], args[1]);
    auto bias = builder.createAdd(mm, args[2]);
    auto relu = builder.createRelu(bias);
    builder.createReturn({relu});

    EXPECT_TRUE(Verifier::verify(*func).success);
    EXPECT_EQ(mm->getType()->toString(), "f32[128,256]");
    EXPECT_EQ(bias->getType()->toString(), "f32[128,256]");
    EXPECT_EQ(relu->getType()->toString(), "f32[128,256]");
}

TEST(IRParser, ParsesAndVerifiesMlpLayer) {
    std::string irCode = R"(
func @test_layer(%arg0: f32[128,784], %arg1: f32[784,256], %arg2: f32[256]) -> f32[128,256] {
  %0 = matmul(%arg0, %arg1) : f32[128,256]
  %1 = add(%0, %arg2) : f32[128,256]
  %2 = relu(%1) : f32[128,256]
  return %2
}
)";

    auto mod = IRParser::parseModule(irCode);
    ASSERT_NE(mod, nullptr);
    ASSERT_EQ(mod->getFunctions().size(), 1u);
    EXPECT_TRUE(Verifier::verify(*mod).success);

    const auto& ops = mod->getFunctions()[0]->getEntryBlock()->getOperations();
    ASSERT_EQ(ops.size(), 4u);
    EXPECT_EQ(ops[0]->getOpName(), "matmul");
    EXPECT_EQ(ops[1]->getOpName(), "add");
    EXPECT_EQ(ops[2]->getOpName(), "relu");
    EXPECT_EQ(ops[3]->getOpName(), "return");
    EXPECT_EQ(ops[3]->getOperand(0), ops[2]->getResult(0));
}

TEST(IRParser, PrintParsePrintIsAFixpoint) {
    auto mod = IRParser::parseModule(R"(
func @f(%x: f32[4,8]) -> f32[8] {
  %0 = softmax(%x) : f32[4,8] {axis = 0}
  %1 = sum(%0) : f32[8] {axes = [0], keepdims = 0}
  return %1
}
)");
    std::string once = mod->toString();
    EXPECT_NE(once.find("{axis = 0}"), std::string::npos) << once;
    EXPECT_NE(once.find("{axes = [0], keepdims = 0}"), std::string::npos) << once;
    EXPECT_NE(once.find("return %1"), std::string::npos) << once;
    EXPECT_EQ(IRParser::parseModule(once)->toString(), once);
}

TEST(IRParser, EveryOpRoundTrips) {
    auto mod = IRParser::parseModule(kEveryOp);
    auto res = Verifier::verify(*mod);
    ASSERT_TRUE(res.success) << allErrors(res);

    std::string once = mod->toString();
    EXPECT_EQ(IRParser::parseModule(once)->toString(), once);

    // Every registered op is covered, so a new op cannot skip the round trip.
    std::set<std::string> used;
    for (const auto& op : mod->getFunctions()[0]->getEntryBlock()->getOperations()) used.insert(op->getOpName());
    for (const auto& def : OpRegistry::instance().ops()) EXPECT_TRUE(used.count(def.name)) << def.name;

    // Special values, roles and annotations survive printing.
    for (const char* fragment : {"{value = 1e-30}", "{value = -inf}", "{value = nan}", "{eps = 0.001}",
                                 "%w: f32[4,5] {param}", "%mask: f32[3,5] {const}", "%q: f32[2,2,3,4] {state}",
                                 "{fusion.group = 0}", "{mem.offset = 128}", "return %24, %0, %23",
                                 "-> (f32[], f32[2,3,5], f32[2,2,3,3])"}) {
        EXPECT_NE(once.find(fragment), std::string::npos) << fragment << "\n" << once;
    }
}

TEST(IRParser, AcceptsBothReturnForms) {
    for (const char* ret : {"return %a, %b", "return(%a, %b)", "return (%a, %b)"}) {
        std::string src = std::string("func @f(%a: f32[2], %b: f32[3]) -> (f32[2], f32[3]) {\n  ") + ret + "\n}\n";
        auto func = IRParser::parseFunction(src);
        const auto& ops = func->getEntryBlock()->getOperations();
        ASSERT_EQ(ops.size(), 1u) << ret;
        EXPECT_EQ(ops[0]->getNumOperands(), 2u) << ret;
        EXPECT_TRUE(Verifier::verify(*func).success) << ret;
    }
    auto empty = IRParser::parseFunction("func @g() -> () {\n  return\n}\n");
    EXPECT_EQ(empty->getEntryBlock()->getOperations()[0]->getNumOperands(), 0u);
    EXPECT_EQ(empty->toString(), "func @g() -> () {\n  return\n}\n");
}

TEST(IRParser, InfersOmittedTypesAndFillsDefaults) {
    auto func = IRParser::parseFunction(R"(
func @f(%x: f32[4,8], %w: f32[8,3]) -> f32[] {
  %0 = matmul(%x, %w)
  %1 = softmax(%0)
  %2 = transpose(%1)
  %3 = sum(%2)
  return %3
}
)");
    EXPECT_EQ(func->toString(), R"(func @f(%x: f32[4,8], %w: f32[8,3]) -> f32[] {
  %0 = matmul(%x, %w) : f32[4,3]
  %1 = softmax(%0) : f32[4,3] {axis = -1}
  %2 = transpose(%1) : f32[3,4] {perm = [1, 0]}
  %3 = sum(%2) : f32[] {axes = [0, 1], keepdims = 0}
  return %3
}
)");
}

TEST(IRParser, IntegerLiteralsAreAcceptedForFloatAttributes) {
    auto func = IRParser::parseFunction(
        "func @f(%x: f32[4]) -> f32[4] {\n  %0 = constant() : f32[4] {value = 2}\n  %1 = add(%x, %0)\n  return %1\n}\n");
    EXPECT_DOUBLE_EQ(func->getEntryBlock()->getOperations()[0]->getDoubleAttr("value"), 2.0);
    EXPECT_NE(func->toString().find("{value = 2.0}"), std::string::npos);
}

TEST(IRParser, ParsesSeveralFunctions) {
    auto mod = IRParser::parseModule(R"(
// leading comment
func @a(%x: f32[2]) -> f32[2] {
  %0 = relu(%x)
  return %0
}
func @b(%x: f32[2]) -> f32[2] {
  %0 = exp(%x)   // names are scoped per function
  return %0
}
)");
    ASSERT_EQ(mod->getFunctions().size(), 2u);
    EXPECT_EQ(mod->getFunctions()[1]->getName(), "b");
    EXPECT_TRUE(Verifier::verify(*mod).success);
}

TEST(IRParser, RejectsUndefinedOperands) {
    EXPECT_THROW(IRParser::parseModule(R"(
func @g(%a: f32[4]) -> f32[4] {
  %0 = add(%a, %typo) : f32[4]
  return %0
}
)"),
                 IRError);
}

struct MalformedCase {
    const char* source;
    const char* error;
};

TEST(IRParser, RejectsMalformedInputWithALocation) {
    const MalformedCase cases[] = {
        {"func @f(%a: f32[4]) -> f32[4] {\n  %0 = add(%a, %typo)\n  return %0\n}", "2:16: use of undefined value '%typo'"},
        {"func @f(%a: f32[4]) -> f32[4] {\n  %0 = frobnicate(%a)\n  return %0\n}", "2:8: unknown op 'frobnicate'"},
        {"func @f(%a: f32[4]) -> f32[4] {\n  %0 = return(%a)\n  return %0\n}", "unknown op 'return'"},
        {"func @f(%a: f32[4]) -> f32[4] {\n  %0 = add(%a)\n  return %0\n}", "'add' expects 2 operand(s), got 1"},
        {"func @f(%a: f32[4]) -> f32[4] {\n  %0 = relu(%a)\n  %0 = exp(%a)\n  return %0\n}", "3:3: redefinition of '%0'"},
        {"func @f(%a: f32[4], %a: f32[4]) -> f32[4] {\n  return %a\n}", "redefinition of '%a'"},
        {"func @f(%a: f32[4]) -> f32[4] {\n  %a = relu(%a)\n  return %a\n}", "redefinition of '%a'"},
        {"func @f(%a: f32[4,4]) -> f32[4,4] {\n  %0 = softmax(%a) {axis = 1.5}\n  return %0\n}",
         "attribute 'axis' of 'softmax' must be int"},
        {"func @f(%a: f32[4,4]) -> f32[4,4] {\n  %0 = softmax(%a) {axiz = 1}\n  return %0\n}",
         "unknown attribute 'axiz' for 'softmax'"},
        {"func @f(%a: f32[4,4]) -> f32[4,4] {\n  %0 = softmax(%a) {axis = 1, axis = 0}\n  return %0\n}",
         "duplicate attribute 'axis'"},
        {"func @f(%a: f32[4]) -> f32[4] {\n  %0 = relu(%a\n  return %0\n}", "expected ')'"},
        {"func @f(%a: f32[4]) -> f32[4] {\n  %0 = relu(%a) ; \n  return %0\n}", "unexpected character ';'"},
        {"func @f(%a: f32[4]) -> f32[4] {\n  %0 = relu(%a)\n  return %0\n", "expected '}' to close function '@f'"},
        {"func @f(%a: f64[4]) -> f32[4] {\n  return %a\n}", "unknown element type 'f64'"},
        {"func @f(%a: f32[-3]) -> f32[4] {\n  return %a\n}", "dimensions must be non-negative"},
        {"func @f(%a: f32[99999999999999999999]) -> f32[4] {\n  return %a\n}", "is out of range"},
        {"func @f(%a: f32[1000000,1000000,1000000]) -> f32[4] {\n  return %a\n}", "tensor type has too many elements"},
        {"func @f(%a: f32[4] {weights}) -> f32[4] {\n  return %a\n}", "unknown argument role 'weights'"},
        {"func @f(%a: f32[4]) f32[4] {\n  return %a\n}", "expected '->'"},
        {"func @f(%a: f32[4,8], %b: f32[4,8]) -> f32[4,8] {\n  %0 = matmul(%a, %b)\n  return %0\n}",
         "2:8: matmul operand shapes don't match: f32[4,8] x f32[4,8]"},
        {"func @f(%a: f32[4]) -> f32[4] {\n  %0 = constant() {value = 1.0}\n  return %0\n}",
         "'constant' needs an explicit result type"},
        {"func @f(%a: f32[4]) -> f32[4] {\n  %0 = constant() : f32[4] {value = \"one}\n  return %0\n}",
         "unterminated string"},
        {"func @f(%a: f32[4]) -> f32[4] {\n  %0 = relu(%a) : f32[4] {fusion.tag = [1, x]}\n  return %0\n}",
         "expected a number in the list"},
        {"module {}", "1:1: expected 'func'"},
        {"func f(%a: f32[4]) -> f32[4] {\n  return %a\n}", "expected a function name such as '@main'"},
        {"func @f(%a: f32[4]) -> f32[4] {\n  %0 = relu(%a) : f32[4] {value = 1e}\n  return %0\n}",
         "malformed exponent"},
    };
    const std::regex located(R"(^\d+:\d+: )");
    for (const auto& c : cases) {
        try {
            IRParser::parseModule(c.source);
            ADD_FAILURE() << "accepted:\n" << c.source;
        } catch (const IRError& e) {
            std::string message = e.what();
            EXPECT_NE(message.find(c.error), std::string::npos) << "got: " << message << "\nfor:\n" << c.source;
            EXPECT_TRUE(std::regex_search(message, located)) << "no line:col in: " << message;
        }
    }
    EXPECT_THROW(IRParser::parseFunction("// nothing here\n"), IRError);
}

// One random edit of IR text. Most edits stay close to valid IR (a changed number, operand, op or line), so many
// mutants get past the parser and exercise the verifier and the passes.
std::string mutate(std::string text, std::mt19937& rng) {
    auto pick = [&rng](size_t n) { return static_cast<size_t>(rng() % n); };
    const std::string alphabet = "%@(){}[],:=->0123456789.eE+-_ \n\"/abcfimnprstuvx";
    switch (rng() % 9) {
        case 0: text.erase(pick(text.size()), 1 + pick(8)); break;
        case 1: text.insert(pick(text.size()), 1, alphabet[pick(alphabet.size())]); break;
        case 2: text[pick(text.size())] = alphabet[pick(alphabet.size())]; break;
        case 3: std::swap(text[pick(text.size())], text[pick(text.size())]); break;
        case 4: { // change one digit of a shape or attribute
            std::vector<size_t> digits;
            for (size_t i = 0; i < text.size(); ++i) {
                if (std::isdigit(static_cast<unsigned char>(text[i]))) digits.push_back(i);
            }
            text[digits[pick(digits.size())]] = static_cast<char>('0' + pick(10));
            break;
        }
        case 5: { // point a value reference at another value
            std::vector<std::pair<size_t, size_t>> refs;
            for (size_t i = 0; i < text.size(); ++i) {
                if (text[i] != '%') continue;
                size_t end = i + 1;
                while (end < text.size() && (std::isalnum(static_cast<unsigned char>(text[end])) || text[end] == '_')) {
                    ++end;
                }
                refs.emplace_back(i, end);
            }
            auto [begin, end] = refs[pick(refs.size())];
            auto [otherBegin, otherEnd] = refs[pick(refs.size())];
            text.replace(begin, end - begin, text.substr(otherBegin, otherEnd - otherBegin));
            break;
        }
        case 6: { // swap in another op
            std::vector<std::pair<size_t, size_t>> names;
            for (size_t i = text.find("= "); i != std::string::npos; i = text.find("= ", i + 1)) {
                size_t end = i + 2;
                while (end < text.size() && (std::isalpha(static_cast<unsigned char>(text[end])) || text[end] == '_')) {
                    ++end;
                }
                if (end < text.size() && text[end] == '(') names.emplace_back(i + 2, end);
            }
            const auto& ops = OpRegistry::instance().ops();
            auto [begin, end] = names[pick(names.size())];
            text.replace(begin, end - begin, ops[pick(ops.size())].name);
            break;
        }
        default: { // drop a line, or copy one somewhere else
            std::vector<std::string> lines;
            std::stringstream stream(text);
            for (std::string line; std::getline(stream, line);) lines.push_back(line);
            size_t i = pick(lines.size());
            if (rng() % 2) lines.erase(lines.begin() + static_cast<std::ptrdiff_t>(i));
            else lines.insert(lines.begin() + static_cast<std::ptrdiff_t>(pick(lines.size())), lines[i]);
            text.clear();
            for (const auto& line : lines) text += line + "\n";
            break;
        }
    }
    return text;
}

// Random edits of a valid module never crash, hang or throw anything but IRError, from the parser through the
// verifier, the printer and every graph pass; whatever verifies prints back to a fixpoint.
TEST(Fuzz, MutatedModulesOnlyRaiseIRError) {
    std::mt19937 rng(20261018);
    int parsed = 0;
    int verified = 0;
    int optimized = 0;
    for (int iter = 0; iter < 2000; ++iter) {
        std::string text = kEveryOp;
        for (int edits = 1 + static_cast<int>(rng() % 3); edits > 0; --edits) text = mutate(text, rng);

        ModulePtr mod;
        try {
            mod = IRParser::parseModule(text);
        } catch (const IRError&) {
            continue;
        }
        ++parsed;
        std::string once = mod->toString();
        if (!Verifier::verify(*mod).success) continue;
        ++verified;
        ModulePtr again;
        ASSERT_NO_THROW(again = IRParser::parseModule(once)) << once;
        ASSERT_EQ(again->toString(), once);

        PassManager pm;
        pm.setVerifyEach(true);
        pm.addPass(std::make_unique<ShapeInferencePass>());
        pm.addPass(std::make_unique<CanonicalizePass>());
        pm.addPass(std::make_unique<CSEPass>());
        pm.addPass(std::make_unique<DCEPass>());
        if (iter % 2) pm.addPass(std::make_unique<AutodiffPass>());
        pm.addPass(std::make_unique<FusionPass>());
        pm.addPass(std::make_unique<MemoryPlanningPass>());
        try {
            pm.run(*mod);
            ++optimized;
        } catch (const IRError&) {
        }
    }
    // Guards against a fuzzer that only ever reaches the lexer.
    EXPECT_GT(parsed, 120);
    EXPECT_GT(verified, 50);
    EXPECT_GT(optimized, 40);
    std::cout << "[ fuzz ] parsed " << parsed << ", verified " << verified << ", optimized " << optimized << " of 2000\n";
}

TEST(OpRegistry, InfersResultTypes) {
    struct Case {
        const char* op;
        std::vector<TypePtr> operands;
        std::map<std::string, AttributeValue> attrs;
        const char* expected;
    };
    const Case cases[] = {
        {"add", {f32({4, 1}), f32({3})}, {}, "f32[4,3]"},
        {"div", {f32({}), f32({2, 3})}, {}, "f32[2,3]"},
        {"gelu", {f32({2, 3})}, {}, "f32[2,3]"},
        {"matmul", {f32({4, 8}), f32({8, 3})}, {}, "f32[4,3]"},
        {"matmul", {f32({2, 4, 8}), f32({8, 3})}, {}, "f32[2,4,3]"},
        {"matmul", {f32({2, 4, 8}), f32({2, 8, 3})}, {}, "f32[2,4,3]"},
        {"matmul", {f32({2, 2, 4, 8}), f32({2, 2, 8, 3})}, {}, "f32[2,2,4,3]"},
        {"softmax", {f32({2, 5})}, {{"axis", int64_t{0}}}, "f32[2,5]"},
        {"cross_entropy", {f32({2, 4, 7}), i32({2, 4})}, {}, "f32[]"},
        {"layernorm", {f32({2, 4, 7}), f32({7}), f32({7})}, {}, "f32[2,4,7]"},
        {"sum", {f32({2, 3, 4})}, {{"axes", std::vector<int64_t>{-1, 0}}}, "f32[3]"},
        {"mean", {f32({2, 3, 4})}, {{"axes", std::vector<int64_t>{1}}, {"keepdims", int64_t{1}}}, "f32[2,1,4]"},
        {"mean", {f32({2, 3})}, {}, "f32[]"},
        {"transpose", {f32({2, 3, 4})}, {{"perm", std::vector<int64_t>{1, 2, 0}}}, "f32[3,4,2]"},
        {"transpose", {i32({2, 3})}, {}, "i32[3,2]"},
        {"reshape", {f32({2, 3, 4})}, {{"shape", std::vector<int64_t>{6, 4}}}, "f32[6,4]"},
        {"embedding", {f32({10, 4}), i32({2, 3})}, {}, "f32[2,3,4]"},
        {"onehot", {i32({5})}, {{"num_classes", int64_t{3}}}, "f32[5,3]"},
    };
    for (const auto& c : cases) {
        IRBuilder builder;
        auto func = builder.createFunction("f", c.operands, {});
        auto result = builder.createOp(c.op, func->getArguments(), c.attrs);
        EXPECT_EQ(result->getType()->toString(), c.expected) << c.op;
    }
}

TEST(OpRegistry, RejectsInvalidOperandsAndAttributes) {
    struct Case {
        const char* op;
        std::vector<TypePtr> operands;
        std::map<std::string, AttributeValue> attrs;
        const char* error;
    };
    const Case cases[] = {
        {"add", {f32({4, 3}), f32({4})}, {}, "not broadcast-compatible"},
        {"add", {f32({4}), i32({4})}, {}, "operand 1 of 'add' must be f32"},
        {"matmul", {f32({4, 8}), f32({4, 8})}, {}, "matmul operand shapes don't match"},
        {"matmul", {f32({8}), f32({8, 3})}, {}, "matmul supports"},
        {"softmax", {f32({2, 5})}, {{"axis", int64_t{2}}}, "axis 2 of 'softmax' is out of range for rank 2"},
        {"softmax", {f32({})}, {}, "rank >= 1"},
        {"cross_entropy", {f32({2, 4}), f32({2})}, {}, "operand 1 of 'cross_entropy' must be i32"},
        {"cross_entropy", {f32({2, 4}), i32({3})}, {}, "don't match logits"},
        {"layernorm", {f32({2, 7}), f32({6}), f32({7})}, {}, "gamma must have shape [7]"},
        {"layernorm", {f32({2, 7}), f32({7}), f32({7})}, {{"eps", 0.0}}, "eps must be positive"},
        {"sum", {f32({2, 3})}, {{"axes", std::vector<int64_t>{1, -1}}}, "lists axis 1 twice"},
        {"sum", {f32({2, 3})}, {{"keepdims", int64_t{2}}}, "keepdims of 'sum' must be 0 or 1"},
        {"transpose", {f32({2, 3})}, {{"perm", std::vector<int64_t>{0, 0}}}, "is not a permutation"},
        {"reshape", {f32({2, 3})}, {{"shape", std::vector<int64_t>{5}}}, "changes the element count"},
        {"reshape", {f32({2, 3})}, {}, "missing attribute 'shape'"},
        {"embedding", {f32({10}), i32({2})}, {}, "embedding table must be 2-D"},
        {"onehot", {i32({5})}, {{"num_classes", int64_t{0}}}, "num_classes must be positive"},
        {"exp", {i32({5})}, {}, "must be f32"},
        {"softmax", {f32({2, 5})}, {{"axis", 1.0}}, "attribute 'axis' of 'softmax' must be an integer"},
        {"bogus", {f32({2})}, {}, "unknown op 'bogus'"},
        {"onehot", {i32({int64_t{1} << 30})}, {{"num_classes", int64_t{1} << 30}}, "has too many elements"},
        {"matmul", {f32({int64_t{1} << 24, 1}), f32({1, int64_t{1} << 25})}, {}, "has too many elements"},
    };
    for (const auto& c : cases) {
        IRBuilder builder;
        auto func = builder.createFunction("f", c.operands, {});
        try {
            builder.createOp(c.op, func->getArguments(), c.attrs);
            ADD_FAILURE() << c.op << " accepted, expected: " << c.error;
        } catch (const IRError& e) {
            EXPECT_NE(std::string(e.what()).find(c.error), std::string::npos) << "got: " << e.what();
        }
    }
}

TEST(OpRegistry, MaterializesDefaultAttributes) {
    IRBuilder builder;
    auto func = builder.createFunction("f", {f32({2, 3, 4}), f32({4}), f32({4})}, {});
    const auto& args = func->getArguments();
    auto softmax = builder.createSoftmax(args[0])->getDefiningOp();
    auto sum = builder.createSum(args[0])->getDefiningOp();
    auto perm = builder.createTranspose(args[0], {})->getDefiningOp();
    auto ln = builder.createOp("layernorm", {args[0], args[1], args[2]})->getDefiningOp();

    EXPECT_EQ(softmax->getIntAttr("axis"), -1);
    EXPECT_EQ(sum->getIntListAttr("axes"), (std::vector<int64_t>{0, 1, 2}));
    EXPECT_EQ(sum->getIntAttr("keepdims"), 0);
    EXPECT_EQ(perm->getIntListAttr("perm"), (std::vector<int64_t>{2, 1, 0}));
    EXPECT_DOUBLE_EQ(ln->getDoubleAttr("eps"), 1e-5);
}

TEST(OpRegistry, DescribesEveryOp) {
    const auto& registry = OpRegistry::instance();
    std::set<std::string> names;
    for (const auto& def : registry.ops()) {
        EXPECT_TRUE(names.insert(def.name).second) << "registered twice: " << def.name;
        EXPECT_EQ(registry.lookup(def.name), &def);
        EXPECT_EQ(def.hasResult, def.name != "return") << def.name;
        EXPECT_EQ(static_cast<bool>(def.inferType), def.hasResult) << def.name;
    }
    EXPECT_EQ(names.size(), 21u);
    EXPECT_EQ(registry.lookup("select"), nullptr);
    EXPECT_EQ(registry.lookup("matmul")->opClass, OpClass::Matmul);
    EXPECT_TRUE(registry.lookup("reshape")->findAttr("shape")->required);
    EXPECT_EQ(registry.lookup("return")->numOperands, -1);
}

TEST(Verifier, RejectsInvalidMatmulShapes) {
    IRBuilder builder;
    auto func = builder.createFunction("bad", {f32({4, 8}), f32({4, 8})}, {f32({4, 8})});
    const auto& args = func->getArguments();
    auto mm = builder.createMatmul(args[0], args[1], f32({4, 8}));
    builder.createReturn({mm});

    EXPECT_THROW(ShapeInferencePass().runOnFunction(*func), IRError);
    auto res = Verifier::verify(*func);
    EXPECT_FALSE(res.success);
    EXPECT_TRUE(hasError(res, "matmul operand shapes don't match: f32[4,8] x f32[4,8]")) << allErrors(res);
}

TEST(Verifier, RejectsResultTypesThatDisagreeWithInference) {
    IRBuilder builder;
    auto func = builder.createFunction("f", {f32({4, 8}), f32({8, 3})}, {f32({4, 4})});
    const auto& args = func->getArguments();
    builder.createReturn({builder.createMatmul(args[0], args[1], f32({4, 4}))});

    auto res = Verifier::verify(*func);
    EXPECT_TRUE(hasError(res, "'%0 = matmul(%arg0, %arg1) : f32[4,4]': result type f32[4,4] does not match the "
                              "inferred type f32[4,3]"))
        << allErrors(res);
}

TEST(Verifier, ChecksTheReturnAgainstTheSignature) {
    {
        IRBuilder builder;
        auto func = builder.createFunction("f", {f32({4, 4})}, {f32({4})});
        builder.createReturn({builder.createRelu(func->getArguments()[0])});
        auto res = Verifier::verify(*func);
        EXPECT_TRUE(hasError(res, "returned value 0 has type f32[4,4] but the function declares f32[4]"))
            << allErrors(res);
    }
    {
        IRBuilder builder;
        auto func = builder.createFunction("f", {f32({4})}, {f32({4}), f32({4})});
        builder.createReturn({func->getArguments()[0]});
        auto res = Verifier::verify(*func);
        EXPECT_TRUE(hasError(res, "returns 1 value(s) but the function declares 2")) << allErrors(res);
    }
    {
        IRBuilder builder;
        auto func = builder.createFunction("f", {f32({4})}, {f32({4})});
        builder.createRelu(func->getArguments()[0]);
        EXPECT_TRUE(hasError(Verifier::verify(*func), "function has no 'return'"));
    }
    {
        IRBuilder builder;
        auto func = builder.createFunction("f", {f32({4})}, {f32({4})});
        auto x = func->getArguments()[0];
        builder.createReturn({x});
        builder.createRelu(x);
        EXPECT_TRUE(hasError(Verifier::verify(*func), "'return' must be the last op"));
    }
}

TEST(Verifier, ChecksAttributes) {
    IRBuilder builder;
    auto func = builder.createFunction("f", {f32({2, 3})}, {f32({6})});
    auto x = func->getArguments()[0];
    auto reshaped = builder.createOp("reshape", {x}, {}, f32({6})); // explicit type: nothing is inferred
    auto softmax = builder.createSoftmax(x);
    softmax->getDefiningOp()->setAttribute("axis", 1.5);
    auto relu = builder.createRelu(x);
    relu->getDefiningOp()->setAttribute("bogus", int64_t{1});
    relu->getDefiningOp()->setAttribute("fusion.group", int64_t{3}); // annotations are allowed
    builder.createReturn({reshaped});

    auto res = Verifier::verify(*func);
    EXPECT_TRUE(hasError(res, "missing required attribute 'shape'")) << allErrors(res);
    EXPECT_TRUE(hasError(res, "attribute 'axis' must be int")) << allErrors(res);
    EXPECT_TRUE(hasError(res, "unknown attribute 'bogus'")) << allErrors(res);
    EXPECT_FALSE(hasError(res, "attribute 'fusion.group'")) << allErrors(res);
    EXPECT_EQ(res.errors.size(), 3u) << allErrors(res);
}

TEST(Verifier, RejectsUsesBeforeDefinitions) {
    IRBuilder builder;
    auto func = builder.createFunction("f", {f32({4})}, {f32({4})});
    auto e = builder.createExp(func->getArguments()[0]);
    builder.createReturn({builder.createRelu(e)});
    func->getEntryBlock()->insertOperation(0, std::make_shared<Operation>("relu", std::vector<ValuePtr>{e},
                                                                          std::vector<TypePtr>{f32({4})}));
    auto res = Verifier::verify(*func);
    EXPECT_TRUE(hasError(res, "'%0 = relu(%1) : f32[4]': operand 0 is used before it is defined")) << allErrors(res);
}

TEST(Verifier, DetectsCorruptedUseLists) {
    IRBuilder builder;
    auto func = builder.createFunction("f", {f32({4})}, {f32({4})});
    auto x = func->getArguments()[0];
    auto e = builder.createExp(x);
    builder.createReturn({e});
    ASSERT_TRUE(Verifier::verify(*func).success);

    x->addUse(e->getDefiningOp()); // a stale entry, as erasing without unlinking used to leave
    auto res = Verifier::verify(*func);
    EXPECT_TRUE(hasError(res, "use list of '%arg0' does not match the ops that use it")) << allErrors(res);
    x->removeUse(e->getDefiningOp());
    EXPECT_TRUE(Verifier::verify(*func).success);
}

TEST(Verifier, PrefixesModuleErrorsWithTheFunction) {
    auto mod = IRParser::parseModule(R"(
func @good(%x: f32[2]) -> f32[2] {
  return %x
}
func @bad(%x: f32[2]) -> f32[3] {
  return %x
}
)");
    auto res = Verifier::verify(*mod);
    ASSERT_EQ(res.errors.size(), 1u) << allErrors(res);
    EXPECT_EQ(res.errors[0].rfind("[@bad] 'return %x': returned value 0 has type f32[2]", 0), 0u) << res.errors[0];
}

TEST(SlotTracker, KeepsValidArgumentNamesAndNumbersResults) {
    IRBuilder builder;
    auto func = builder.createFunction("f", {f32({2}), f32({2}), f32({2}), f32({2}), f32({2})}, {f32({2})});
    const auto& args = func->getArguments();
    args[0]->setName("x");
    args[1]->setName("x");     // duplicate: falls back to its position
    args[2]->setName("1st");   // not a valid name
    args[3]->setName("arg1");  // takes the fallback name of args[1]
    args[4]->setName("w.t$0");
    auto a = builder.createAdd(args[0], args[1]);
    a->setName("ignored");     // results are always numbered
    auto b = builder.createMul(a, args[4]);
    builder.createReturn({b});

    SlotTracker names(*func);
    EXPECT_EQ(names.name(args[0].get()), "x");
    EXPECT_EQ(names.name(args[1].get()), "arg1_1");
    EXPECT_EQ(names.name(args[2].get()), "arg2");
    EXPECT_EQ(names.name(args[3].get()), "arg1");
    EXPECT_EQ(names.name(args[4].get()), "w.t$0");
    EXPECT_EQ(names.name(a.get()), "0");
    EXPECT_EQ(names.name(b.get()), "1");
    EXPECT_EQ(names.name(nullptr), "<null>");

    // The printed form names everything consistently and parses back.
    std::string text = func->toString();
    EXPECT_EQ(IRParser::parseFunction(text)->toString(), text);
}

TEST(SlotTracker, ValidatesArgumentNames) {
    for (const char* name : {"x", "_tmp", "w1.weight", "a$b", "Z9"}) EXPECT_TRUE(isValidArgumentName(name)) << name;
    for (const char* name : {"", "1x", "a-b", "a b", ".x", "%x", "é"}) EXPECT_FALSE(isValidArgumentName(name)) << name;
}

TEST(OperationHygiene, ErasingUnlinksAndRefusesLiveOps) {
    IRBuilder builder;
    auto func = builder.createFunction("f", {f32({4})}, {f32({4})});
    auto x = func->getArguments()[0];
    auto live = builder.createExp(x);
    auto dead = builder.createRelu(x);
    builder.createReturn({live});
    Block* block = func->getEntryBlock();

    EXPECT_THROW(block->eraseOperation(live->getDefiningOp()), IRError);
    EXPECT_EQ(x->getUses().size(), 2u);
    block->eraseOperation(dead->getDefiningOp());
    EXPECT_EQ(x->getUses().size(), 1u);
    EXPECT_EQ(dead->getDefiningOp(), nullptr); // the erased op is gone and no longer claims its result
    EXPECT_TRUE(Verifier::verify(*func).success);
}

TEST(OperationHygiene, ReplaceAllUsesWithMovesEveryUse) {
    IRBuilder builder;
    auto func = builder.createFunction("f", {f32({4}), f32({4})}, {f32({4})});
    auto x = func->getArguments()[0];
    auto y = func->getArguments()[1];
    auto sq = builder.createMul(x, x);
    auto out = builder.createAdd(sq, x);
    builder.createReturn({out});

    replaceAllUsesWith(x, y);
    EXPECT_TRUE(x->getUses().empty());
    EXPECT_EQ(y->getUses().size(), 3u);
    EXPECT_EQ(sq->getDefiningOp()->getOperand(0), y);
    EXPECT_EQ(sq->getDefiningOp()->getOperand(1), y);
    EXPECT_EQ(out->getDefiningOp()->getOperand(1), y);
    EXPECT_TRUE(Verifier::verify(*func).success);
}

TEST(OperationHygiene, TypedAttributeAccessThrowsIRError) {
    IRBuilder builder;
    auto func = builder.createFunction("f", {f32({2, 3})}, {});
    auto op = builder.createSoftmax(func->getArguments()[0])->getDefiningOp();
    EXPECT_EQ(op->getIntAttr("axis"), -1);
    EXPECT_THROW(op->getDoubleAttr("axis"), IRError);
    EXPECT_THROW(op->getIntAttr("missing"), IRError);
    EXPECT_THROW(op->getIntListAttr("axis"), IRError);
}

} // namespace
