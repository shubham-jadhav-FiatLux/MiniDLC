# Shape Inference Pass

## Purpose
Infers and verifies output shapes and data types for all SSA operations in forward topological order.

## Algorithm
- Formulated as forward abstract interpretation over a shape lattice.
- Elements: static integer dimensions \([d_0, d_1, \dots]\), \(\top\) (unknown/invalid), or \(\bot\) (uninitialized).
- Evaluates broadcast compatibility using standard multidimensional expansion rules.

## Course Link
- CS6843 Program Analysis: Abstract Interpretation, Galois connections, and forward dataflow analysis.
