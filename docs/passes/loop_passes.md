# Loop Transformation Passes

## Purpose
Transform lowered loop nests to use the target's hardware: caches, SIMD, ILP and multiple cores on the CPU; shared
memory, thread blocks and warps on the GPU.

## Passes (v1)

| Pass | What it does in v1 | Status |
|---|---|---|
| `LoopTilingPass` | Sets the outer loop's step to 1; creates no tiles | Placeholder |
| `LoopVectorizationPass` | Stamps `vectorize(8)` on the innermost loop without a legality check — including strided reduction loops, such as the `k` loop of a matmul | Attribute only |
| `LoopUnrollingPass` | Stamps `unroll(4)` on the innermost loop; `dlc-opt` never runs it | Attribute only |
| `LoopParallelizationPass` | Marks every root loop `parallel` without dependence analysis | Unconditional |
| `BoundsCheckEliminationPass` | Interval range analysis and remainder peeling (`bounds_check_elim.md`) | Works; no guards to remove |

The v1 code generators read none of these attributes except the root loop's bounds and `parallel` flag, so the passes
do not change generated code.

## v2 (P4–P5)
Each loop-IR kernel gets a per-target schedule built from primitives — split/tile, reorder, bind to GPU blocks and
threads, stage in shared memory or registers, unroll, vectorize, parallel — applied only where dependence analysis
(distance/direction vectors over affine accesses) proves them legal. The emitters translate the scheduled loop IR
directly.

## Course links
Caches, ILP and SIMD (CS6600/CS6400); dependence analysis (CS6843); parallel loops and scheduling (ID5130); shared
memory and warps (CS6023).
