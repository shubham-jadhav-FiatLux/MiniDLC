#include "dlc/ir/Builder.h"
#include "dlc/ir/Verifier.h"
#include "dlc/ir/Printer.h"
#include "dlc/ir/Parser.h"
#include <cassert>
#include <iostream>

void testTypes() {
    dlc::TensorType t1(dlc::DataType::Float32, {128, 784});
    assert(t1.getRank() == 2);
    assert(t1.getNumElements() == 128 * 784);
    assert(t1.getSizeBytes() == 128 * 784 * 4);
    assert(t1.toString() == "f32[128,784]");
    std::cout << "[PASS] testTypes\n";
}

void testBuilderAndVerifier() {
    dlc::IRBuilder builder;
    auto f32_2d = std::make_shared<dlc::TensorType>(dlc::DataType::Float32, dlc::Shape{128, 784});
    auto f32_w = std::make_shared<dlc::TensorType>(dlc::DataType::Float32, dlc::Shape{784, 256});
    auto f32_b = std::make_shared<dlc::TensorType>(dlc::DataType::Float32, dlc::Shape{256});
    auto f32_out = std::make_shared<dlc::TensorType>(dlc::DataType::Float32, dlc::Shape{128, 256});

    auto func = builder.createFunction("test_mlp_layer", {f32_2d, f32_w, f32_b}, {f32_out});
    auto x = func->getArguments()[0];
    auto w = func->getArguments()[1];
    auto b = func->getArguments()[2];

    auto mm = builder.createMatmul(x, w);
    auto bias = builder.createAdd(mm, b);
    auto relu = builder.createRelu(bias);
    builder.createReturn({relu});

    auto vRes = dlc::Verifier::verify(*func);
    assert(vRes.success);
    std::cout << "[PASS] testBuilderAndVerifier\n";
}

void testParserRoundTrip() {
    std::string irCode = R"(
func @test_roundtrip(%arg0: f32[128,784], %arg1: f32[784,256], %arg2: f32[256]) -> f32[128,256] {
  %0 = matmul(%arg0, %arg1) : f32[128,256]
  %1 = add(%0, %arg2) : f32[128,256]
  %2 = relu(%1) : f32[128,256]
  return %2
}
)";

    auto mod = dlc::IRParser::parseModule(irCode);
    assert(mod != nullptr);
    assert(mod->getFunctions().size() == 1);
    auto vRes = dlc::Verifier::verify(*mod);
    assert(vRes.success);
    std::cout << "[PASS] testParserRoundTrip\n";
}

int main_ir_tests() {
    testTypes();
    testBuilderAndVerifier();
    testParserRoundTrip();
    return 0;
}
