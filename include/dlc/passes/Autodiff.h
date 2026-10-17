#pragma once

#include "dlc/passes/Pass.h"

namespace dlc {

// Reverse-mode differentiation of the first returned value (an f32[] loss) with respect to every param argument.
// Activity analysis limits the backward graph to values that depend on a parameter and reach the loss. The
// gradients are appended to the results, each annotated `{grad = %param}`.
class AutodiffPass : public Pass {
public:
    std::string getName() const override { return "autodiff"; }
    bool runOnFunction(Function& func) override;
};

} // namespace dlc
