# Shape Inference

## Purpose
Compute every value's shape and dtype from its operands, applying NumPy broadcasting rules, and reject programs whose
shapes don't fit together.

## Algorithm (P1)
A single forward sweep in program order. Each op's result type is recomputed by its type rule in the op registry
(`lib/ir/OpRegistry.cpp`), the same rule the parser uses for omitted types and the verifier checks against: broadcasting
for elementwise ops, 2-D, 3-D × 2-D, 3-D and 4-D matmuls, reductions over any set of axes with `keepdims`, normalization
axes, LayerNorm and cross-entropy operand shapes, permutations, reshape element counts, and gather index types. With
static shapes, SSA and a single block, one sweep reaches the fixpoint. Framed as abstract interpretation, every value
sits at a concrete shape in the lattice unknown ⊑ concrete shape ⊑ conflict, and a conflict is an error.

## Status (P1)
- Repairs wrongly declared types: `tests/lit/shape_infer.dlc` and `tests/lit/basic_relu_matmul.dlc` start from
  deliberately wrong types (with `--no-verify-input`).
- Rejects invalid programs with an error naming the op, e.g. `'matmul': matmul operand shapes don't match: f32[4,8] x
  f32[4,8]` (`Verifier.RejectsInvalidMatmulShapes`, `OpRegistry.RejectsInvalidOperandsAndAttributes`).
- Translation-validated on 16 exported graphs (`tests/python/test_translation_validation.py`).

## Course link
Abstract interpretation and forward dataflow analysis (CS6843).
