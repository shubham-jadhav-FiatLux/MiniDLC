# Canonicalization and Constant Folding

## Purpose
Simplify algebraic identities and evaluate constant expressions at compile time, so later passes see a normalized graph.

## Rewrites implemented (v1)
- `x + 0 → x`, `0 + x → x`, `x - 0 → x`
- `x * 1 → x`, `1 * x → x`, `x / 1 → x`
- `x * 0 → 0`, `0 * x → 0` (unsafe; see Status)
- Constant folding of `add`, `sub`, `mul`, `div` when both operands are constants.

The pass makes a single sweep over the block.

## Status (P1)
- Folding walks a snapshot of the op list, so inserting the folded constant no longer invalidates the walk (it was a
  heap-use-after-free; `Canonicalize.FoldsConstantsSafely`).
- Constants are recognized by op name, so parsed IR folds too (`Canonicalize.FoldsParsedConstants`,
  `tests/lit/canonicalize.dlc`).
- Translation-validated on 16 exported graphs. Still wrong (P2): identity rewrites don't check result types, so
  `x[4] * 0` is replaced by the scalar constant and changes the function's type (the `mul_by_zero` case in
  `tests/python/test_translation_validation.py` and `KnownBugs.DISABLED_CanonicalizePreservesResultTypes`); `x * 0 → 0`
  is also wrong for NaN and Inf inputs.
- No fixpoint iteration and no commutative operand normalization.

## v2 (P2)
Type-preserving, NaN-safe rewrites applied to a fixpoint; folding without mutating the op list during iteration;
transposes folded into matmul `transpose_a`/`transpose_b` flags; reshape/transpose pairs cancelled; operand order
normalized for commutative ops so CSE finds more matches.
