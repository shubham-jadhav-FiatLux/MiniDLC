#pragma once

#include "dlc/passes/Pass.h"

namespace dlc {

// Recomputes every result type from the op registry's type rules, in program order; invalid shapes raise IRError.
class ShapeInferencePass : public Pass {
public:
    std::string getName() const override { return "shape-infer"; }
    bool runOnFunction(Function& func) override;

    // Helper broadcasting routine
    static bool broadcastShapes(const Shape& a, const Shape& b, Shape& outShape);
};

} // namespace dlc
