#include "dlc/ir/Builder.h"
#include "dlc/passes/ShapeInference.h"
#include "dlc/passes/CSE.h"
#include "dlc/passes/DCE.h"
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

int main_passes_tests() {
    testShapeBroadcasting();
    testCSEAndDCE();
    return 0;
}
