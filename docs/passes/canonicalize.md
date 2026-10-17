# Canonicalization and Constant Folding

## Purpose
Simplify algebraic identities and evaluate constant expressions at compile time, so later passes see a normalized graph.

## Rewrites implemented (v1)
- `x + 0 → x`, `0 + x → x`, `x - 0 → x`
- `x * 1 → x`, `1 * x → x`, `x / 1 → x`
- `x * 0 → 0`, `0 * x → 0` (unsafe; see Status)
- Constant folding of `add`, `sub`, `mul`, `div` when both operands are constants.

The pass makes a single sweep over the block.

## Status (v1)
- Constant folding inserts the folded constant into the op list while iterating over it, which is a heap-use-after-free
  under AddressSanitizer (`KnownBugs.DISABLED_ConstantFoldingIsMemorySafe`).
- Identity rewrites don't check result types: `x[4,4] * 0` is replaced by the scalar constant, changing the function's
  result type. `x * 0 → 0` is also wrong for NaN and Inf inputs.
- Constants created by the parser are generic ops, so nothing folds in `dlc-opt` or the lit tests.
- No fixpoint iteration and no commutative operand normalization.

## v2 (P2)
Type-preserving, NaN-safe rewrites applied to a fixpoint; folding without mutating the op list during iteration;
transposes folded into matmul `transpose_a`/`transpose_b` flags; reshape/transpose pairs cancelled; operand order
normalized for commutative ops so CSE finds more matches.
