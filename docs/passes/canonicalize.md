# Canonicalization and Constant Folding

## Purpose
Simplify algebraic identities and evaluate constant expressions at compile time, so later passes see a normalized graph.

## Rewrites (P2, `lib/passes/Canonicalize.cpp`)
Applied until none fires (at most 64 sweeps). Every rewrite keeps the result type — when a rewrite would return a
smaller, broadcast operand, it returns `broadcast_to(operand)` instead — and is exact in IEEE arithmetic up to the sign
of zero:

- **Constant folding** of any op whose operands are all `constant` splats and whose result is again a splat:
  arithmetic, unary functions, `cmp_gt`, `select`, reductions (`sum` of a splat is the value times the count) and
  shape ops. Folding happens in double precision.
- **Identities:** `x + 0`, `0 + x`, `x − 0`, `x · 1`, `1 · x`, `x / 1`, `select` on a constant condition; transposes
  with the identity permutation; reshapes and broadcasts to the same shape.
- **Chains:** `transpose(transpose(x, p), q) = transpose(x, p∘q)`; `reshape(reshape(x))` and
  `broadcast_to(broadcast_to(x))` collapse to one op.
- **Transpose folding:** a transpose that swaps the last two axes of a `matmul` operand becomes a `transpose_a` /
  `transpose_b` flag, so the transposes autodiff and attention produce cost nothing.
- **Commutative operand order:** `add` and `mul` put their operands in a canonical order (arguments, then results in
  program order, constants last), so CSE finds `a + b` and `b + a` equal.

`x · 0 → 0` is deliberately absent: it is wrong for NaN and Inf inputs.

## Status (P2)
Unit tests in `tests/unit/test_passes.cpp` (`Canonicalize.*`), the golden test `tests/lit/canonicalize.dlc`, and
translation validation on 16 exported graphs, including the regression graph for the v1 bug where `x[4] * 0` was
rewritten to the scalar constant.


## Course link
Peephole rewriting to a fixpoint, as in LLVM's InstCombine; the IEEE caveats are the classic ones that make
`-ffast-math` an opt-in (ID5130 floating point).
