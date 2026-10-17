# Shape Inference

## Purpose
Compute every value's shape and dtype from its operands, applying NumPy broadcasting rules, and reject programs whose
shapes don't fit together.

## Algorithm (v1)
A single forward sweep in program order: for each op with a known rule (elementwise with broadcasting, 2-D/3-D/4-D
matmul, reductions over one axis, transpose, reshape, embedding, one-hot), compute the result type and overwrite the
declared one.

## Status (v1)
- Correctly propagates shapes through the MLP and Transformer ops it knows (`tests/lit/shape_infer.dlc`,
  `tests/lit/basic_relu_matmul.dlc` start from deliberately wrong types).
- Reports no errors: an incompatible broadcast or a mismatched matmul leaves the declared type in place
  (`KnownBugs.DISABLED_VerifierRejectsInvalidMatmulShapes`).
- No rule for 3-D × 2-D matmul (sequence projections); reductions take a single axis.

## v2 (P1)
Shape rules come from the op registry and are shared with the verifier; invalid programs are rejected with an error. The
analysis can be framed as forward abstract interpretation over a lattice of shapes (unknown ⊑ concrete shape ⊑
conflict).

## Course link
Abstract interpretation and forward dataflow analysis (CS6843).
