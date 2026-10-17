#pragma once

#include "dlc/passes/Pass.h"

namespace dlc {

// Pattern rewrites applied until none fires. Every rewrite keeps the result type and is exact in IEEE arithmetic up
// to the sign of zero (so no `x * 0 → 0`, which is wrong for NaN and Inf): constant folding of splat constants,
// identities (x + 0, x - 0, x * 1, x / 1, select on a constant), transposes folded into matmul flags,
// transpose/reshape/broadcast chains collapsed, and commutative operands put in a canonical order for CSE.
class CanonicalizePass : public Pass {
public:
    std::string getName() const override { return "canonicalize"; }
    bool runOnFunction(Function& func) override;
};

} // namespace dlc
