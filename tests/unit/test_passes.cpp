#include "dlc/ir/Builder.h"
#include "dlc/passes/ShapeInference.h"
#include "dlc/passes/Canonicalize.h"
#include "dlc/passes/CSE.h"
#include "dlc/passes/DCE.h"
#include "dlc/passes/Autodiff.h"
#include <cassert>
#include <iostream>

void testShapeBroadcasting() {
    dlc::Shape a = {128, 1};
    dlc::Shape b = {256};
    dlc::Shape out;
    bool ok = dlc::ShapeInferencePass::broadcastShapes(a, b, out);
    assert(ok);
    (void)ok;
    assert(out.size() == 2);
    assert(out[0] == 128);
    assert(out[1] == 256);
    std::cout << "[PASS] testShapeBroadcasting\n";
}

void testCSEAndDCE() {
    dlc::IRBuilder builder;
    auto f32_vec = std::make_shared<dlc::TensorType>(dlc::DataType::Float32, dlc::Shape{128});
    auto func = builder.createFunction("test_cse", {f32_vec, f32_vec}, {f32_vec});
    auto a = func->getArguments()[0];
    auto b = func->getArguments()[1];

    // Create identical operations
    auto sum1 = builder.createAdd(a, b);
    auto sum2 = builder.createAdd(a, b); // Redundant

    // Dead op
    auto dead = builder.createMul(a, b);

    // Live return uses sum2
    auto out = builder.createRelu(sum2);
    builder.createReturn({out});

    // Run CSE
    dlc::CSEPass cse;
    bool cseChanged = cse.runOnFunction(*func);
    assert(cseChanged);
    (void)cseChanged;

    // Run DCE
    dlc::DCEPass dce;
    bool dceChanged = dce.runOnFunction(*func);
    assert(dceChanged);
    (void)dceChanged;

    std::cout << "[PASS] testCSEAndDCE\n";
}

void testCanonicalize() {
    dlc::IRBuilder builder;
    auto f32_vec = std::make_shared<dlc::TensorType>(dlc::DataType::Float32, dlc::Shape{128});
    auto func = builder.createFunction("test_canon", {f32_vec}, {f32_vec});
    auto x = func->getArguments()[0];

    auto cZero = builder.createConstant(0.0, f32_vec);
    auto added = builder.createAdd(x, cZero); // x + 0 -> x
    builder.createReturn({added});

    dlc::CanonicalizePass canon;
    bool changed = canon.runOnFunction(*func);
    assert(changed);
    (void)changed;
    std::cout << "[PASS] testCanonicalize\n";
}

void testAutodiff() {
    dlc::IRBuilder builder;
    auto f32_2d = std::make_shared<dlc::TensorType>(dlc::DataType::Float32, dlc::Shape{128, 784});
    auto f32_w = std::make_shared<dlc::TensorType>(dlc::DataType::Float32, dlc::Shape{784, 256});
    auto func = builder.createFunction("test_ad", {f32_2d, f32_w}, {f32_2d});
    auto x = func->getArguments()[0];
    auto w = func->getArguments()[1];

    auto mm = builder.createMatmul(x, w);
    auto relu = builder.createRelu(mm);
    auto loss = builder.createSum(relu);
    builder.createReturn({loss});

    dlc::AutodiffPass ad;
    bool changed = ad.runOnFunction(*func);
    assert(changed);
    (void)changed;
    std::cout << "[PASS] testAutodiff\n";
}

int main_passes_tests() {
    testShapeBroadcasting();
    testCSEAndDCE();
    testCanonicalize();
    testAutodiff();
    return 0;
}
