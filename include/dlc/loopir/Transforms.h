#pragma once

#include "dlc/loopir/LoopIR.h"
#include <string>

// Loop IR rewriting: substitution and the simplification pass.
namespace dlc::loopir {

// `e` with new operands, rebuilt through the constructors (so constants fold again).
ExprPtr rebuild(const Expr& e, std::vector<ExprPtr> args);

// Replaces every use of the variable `name` in `stmts` (and nested bodies) by `value`.
void substitute(Body& stmts, const std::string& name, const ExprPtr& value);

// Removes the Let statements that bind `name`; returns how many.
int removeLets(Body& stmts, const std::string& name);

struct SimplifyReport {
    int propagated{0}; // integer constants bound by Let, substituted into their uses
    int folded{0};     // divisions and remainders that range analysis decided
    int hoisted{0};    // loop-invariant Lets moved out of their loops (LICM)
};

// The loop simplification pass, run after the bounds and dependence checks:
//   - constant propagation: `let x = 7` (an integer constant, bound once in the kernel and never assigned) is
//     substituted into its uses, and the constructors fold the arithmetic that becomes constant;
//   - index arithmetic: with the range facts of the loops, branches and integer locals, `x / c` and `x % c` (c > 0,
//     x >= 0) become `q` and `x - q*c` when the quotient q is the same over x's whole range (e.g. 0 and x when
//     x < c);
//   - LICM: a Let whose value is pure (no loads or shuffles, no integer division except by a nonzero constant: it will
//     run even if the loop runs zero times) and uses nothing the loop defines or assigns moves out of a serial,
//     vectorized or unrolled loop that runs at least once (never out of a parallel or bound loop, whose structure the
//     runtime and the GPU launch rely on). It must be bound once in the kernel and never assigned.
// Strength reduction of induction variables (i*c → an accumulator) is left to LLVM's loop strength reduction: in
// the loop IR it would hide the affine indices that range and dependence analysis read.
SimplifyReport simplify(Kernel& kernel);

} // namespace dlc::loopir
