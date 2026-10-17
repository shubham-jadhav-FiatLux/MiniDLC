#include "dlc/ir/Builder.h"
#include "dlc/ir/Parser.h"
#include "dlc/ir/Verifier.h"
#include <gtest/gtest.h>

namespace {

using namespace dlc;

TypePtr f32(Shape shape) { return std::make_shared<TensorType>(DataType::Float32, std::move(shape)); }

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

// Parsing only: the print -> parse -> print round trip is DISABLED_PrintParsePrintIsAFixpoint in
// test_known_bugs.cpp until the parser handles attributes and the printed return syntax (P1).
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

} // namespace
