# Operator Fusion

## Purpose
Group operations into kernels so intermediate tensors don't make round trips to memory.

## Patterns
1. **Matmul epilogue:** a matmul followed by a bias `add` and a `relu`, each with a single use.
2. **Elementwise chains:** consecutive elementwise ops that consume the group's results and have the same result shape.
3. **Row-wise kernels** (softmax, LayerNorm, cross-entropy): planned for v2.

## Legality (required)
- No cycle may be created in the dataflow graph (checked by reachability).
- Iteration spaces must be compatible.
- Every operand read by the kernel must be defined before the kernel runs.
- A value used outside the group must still be written to memory.

## Status (v1)
- The pass finds matmul epilogues (requiring single uses) and elementwise chains (requiring equal result shapes), but
  checks nothing else: an `[M,N]` residual is treated as a bias and read as `bias[j]`, a bias computed after the matmul
  is read before it is written, and intermediate values used outside an elementwise chain are never stored.
- Ops that are not matmul, elementwise or reductions are never assigned a group, so the lowering drops them when any
  group exists.
- The elementwise lowering keeps only the first binary op of a chain.
- At the MNIST layer size, fusing the epilogue gives no measurable speedup with the same GEMM kernel
  (`benchmarks/fusion_ablation.py`; `docs/perf-report.md` §3).

## v2 (P3)
Greedy fusion with the legality rules above, a cost model (FLOPs and bytes per kernel) to decide profitability, kernels
with several outputs, and row-wise reduction kernels.

## Course link
Dependence analysis and legality of reordering (CS6843).
