# Operator Fusion Pass

## Purpose
Groups fine-grained tensor operations into compound kernel groups to eliminate intermediate round-trips to main memory (DRAM).

## Fusion Patterns
1. **Elementwise Chains**: Consecutive pointwise operations (e.g. `add`, `relu`, `mul`) with identical shapes.
2. **Matmul Epilogues**: Elementwise activations and bias additions directly following a matrix multiplication accumulator writeback.
3. **Row-wise Reductions**: Reductions along inner dimensions fused into producers.

## Legality and Dependences
- Fusion must not introduce directed cycles in the dataflow DAG.
- Iteration spaces between producer and consumer must be compatible.
- Course reference: CS6843 Loop Parallelism and Dependences.
