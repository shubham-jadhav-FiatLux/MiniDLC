# Course-Concept Map

How MiniDLC uses graduate coursework from IIT Madras, and how far each concept is implemented.

Status: **✓** implemented and tested · **◐** implemented with known defects, or not on an executed path yet · **○**
planned (phase given). The defects are detailed in [`architecture.md`](architecture.md), Part 2.

## Summary

| Concept | Where | Course | Status |
|---|---|---|---|
| SSA form, IR design, verifier, textual IR | `include/dlc/ir/`, `lib/ir/` | CS6843 Program Analysis | ◐ verifier checks only def-before-use; parser drops attributes (P1) |
| Value numbering (CSE) | `lib/passes/CSE.cpp` | CS6843 (available expressions) | ◐ key ignores result types; doubles keyed to 6 digits (P2) |
| Reverse reachability (DCE) | `lib/passes/DCE.cpp` | CS6843 (liveness) | ◐ erased ops stay in use lists (P1) |
| Abstract interpretation over shapes | `lib/passes/ShapeInference.cpp` | CS6843 | ◐ infers shapes but reports no errors (P1) |
| Reverse-mode autodiff (VJPs) | `python/minidlc/reference.py`, `lib/passes/Autodiff.cpp` | Chain rule / adjoint method | ✓ Python, all 18 ops gradient-checked; ◐ C++ VJPs wrong for ReLU, GELU, mean, LayerNorm (P2) |
| Activity analysis | — | CS6843 (dataflow) | ○ P2 |
| Dependences and fusion legality | `lib/passes/Fusion.cpp` | CS6843 (dependence analysis) | ◐ groups ops without legality checks (P3) |
| Liveness-based memory planning | `lib/passes/MemoryPlanning.cpp` | CS6843; register-allocation analogy | ◐ greedy interval offset assignment; unsafe with fusion (P3) |
| Interval range analysis, bounds-check elimination | `lib/loopir/LoopPasses.cpp` | CS6843 A2 (range analysis), A4 (bounds checks) | ◐ analysis and remainder peeling work; no guards to remove yet (P4) |
| Golden IR tests (lit + FileCheck) | `tests/lit/` | CS6843 / LLVM tooling | ✓ |
| Cache blocking, loop tiling | `lib/loopir/LoopPasses.cpp` | CS6600 Computer Architecture | ○ the tiling pass is a placeholder (P4) |
| SIMD vectorization (AVX2) | `python/minidlc/llvm_engine.py` | CS6600 / CS6400 | ✓ hand-written `<8 x float>` kernels; ○ generated (P4) |
| ILP through unrolling | `lib/loopir/LoopPasses.cpp` | CS6600 | ○ attribute only (P4) |
| Roofline model | `benchmarks/`, `docs/perf-report.md` | CS6400 / ID5130 | ✓ CPU, measured; GPU ceilings only (P5) |
| Thread-pool runtime | `include/dlc/runtime/ThreadPool.h` | ID5130 (work sharing) | ◐ mutex + condition-variable queue; not on the executed path (P4) |
| Numerical stability (log-sum-exp, NaN semantics) | `python/minidlc/reference.py`, `python/minidlc/llvm_engine.py` | ID5130 (floating point) | ✓ reference interpreter and hand-written CPU kernels |
| Shared-memory tiling, coalescing, bank conflicts | `lib/backends/cuda/CUDACodeGen.cpp` | CS6023 GPU Programming | ◐ text templates, never run; transposed tiles unpadded (P5) |
| Warp-shuffle reductions | `lib/backends/cuda/CUDACodeGen.cpp` | CS6023 (warp primitives) | ◐ the template reduces only warp 0, so rows longer than 32 are wrong (P5) |
| Atomic scatter (`atomicAdd`) | `lib/backends/cuda/CUDACodeGen.cpp` | CS6023 (atomics) | ◐ template, never run (P5) |
| PTX via LLVM NVPTX; CUDA driver API | — | CS6023 + LLVM | ○ P5 |

## By course

### CS6843 Program Analysis
- **SSA and verification.** The graph IR is in SSA form; the verifier checks that every operand is defined before use
  and that a return exists. Per-op type rules, signature checks and a parser that round-trips come in P1.
- **Dataflow.** CSE numbers values by hashing (op, operands, attributes) within the single block; DCE marks values live
  backwards from the return. Activity analysis for autodiff (forward "varied" × backward "useful") comes in P2.
- **Memory planning.** Live intervals over the schedule, then greedy offset assignment (largest first) with an
  interference check — the interval-colouring analogy to register allocation. v2 moves it to kernel granularity, after
  fusion, with parameters in a persistent region (P3).
- **Range analysis and bounds checks (A2/A4).** Interval arithmetic over loop induction variables proves affine
  accesses in bounds, and loops whose trip count is not a multiple of the vector width are split into a vector main
  loop and a scalar tail. In v2 the loop IR has explicit guards, so the analysis deletes guards it proves redundant and
  splits GPU tiles into unguarded interior and guarded boundary blocks (P4–P5).

### CS6023 GPU Programming
- The v1 CUDA emitter produces text for shared-memory tiled GEMMs, warp-shuffle reductions and an `atomicAdd` scatter,
  but none of it has been compiled or run, and the softmax template combines only warp 0's partial results.
- v2 generates GPU code from the loop IR through LLVM's NVPTX backend: thread-bound loops, shared-memory staging with
  padding against bank conflicts, warp shuffles plus a shared-memory combine across warps, and atomics, validated on a
  T4 with `compute-sanitizer` (P5).

### CS6600 / CS6400 Computer Architecture
- The MNIST kernels are hand-written AVX2 LLVM IR: `<8 x float>` loads, `llvm.fmuladd`, and `llvm.maximum` for a
  NaN-propagating ReLU. The forward GEMM vectorizes over output columns; the backward GEMMs are scalar.
- The roofline is measured on the development laptop (FMA peak, STREAM copy). Tiling, unrolling and vectorization as
  compiler transformations arrive with the v2 CPU schedules (P4).

### ID5130 Parallel Scientific Computing
- Numerics: softmax and cross-entropy subtract the row maximum before `exp`; the cross-entropy gradient is computed as
  `(softmax − onehot) / N`; ReLU propagates NaN as PyTorch does. Contract tests cover these on the reference interpreter.
- The C++ thread pool (mutex and condition-variable task queue, `parallelFor` over chunks) exists but no executed path
  uses it yet; the v2 CPU runtime dispatches parallel loops through it (P4).
