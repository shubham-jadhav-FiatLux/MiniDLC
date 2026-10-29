# Course-Concept Map

How MiniDLC uses graduate coursework from IIT Madras, and how far each concept is implemented.

Status: **✓** implemented and tested · **◐** implemented with known defects, or not on an executed path yet · **○**
planned (phase given). The defects are detailed in [`architecture.md`](architecture.md), Part 2.

## Summary

| Concept | Where | Course | Status |
|---|---|---|---|
| SSA form, IR design, verifier, textual IR | `include/dlc/ir/`, `lib/ir/` | CS6843 Program Analysis | ✓ op registry; verifier with per-op type rules, dominance and use-list checks; parser that round-trips |
| Value numbering (CSE) | `lib/passes/CSE.cpp` | CS6843 (available expressions) | ✓ keyed on operand identity, result type and exact attribute bits |
| Reverse reachability (DCE) | `lib/passes/DCE.cpp` | CS6843 (liveness) | ✓ erased ops are unlinked from their operands' use lists |
| Abstract interpretation over shapes | `lib/passes/ShapeInference.cpp`, `lib/ir/OpRegistry.cpp` | CS6843 | ✓ registry type rules; invalid programs rejected |
| Translation validation of passes | `tests/python/test_translation_validation.py` | CS6843 (semantics preservation) | ✓ every graph pass on 16 exported graphs; known miscompiles are strict expected failures |
| Reverse-mode autodiff (VJPs) | `python/minidlc/reference.py`, `lib/passes/Autodiff.cpp` | Chain rule / adjoint method | ✓ Python and C++; every C++ VJP checked against float64 finite differences |
| Activity analysis | `lib/passes/Activity.cpp` | CS6843 (dataflow) | ✓ varied ∩ useful; no gradient for the input batch |
| Optimizer lowering, buffer donation | `lib/passes/Optimizer.cpp` | — | ✓ Adam/SGD in the graph; step graph matches eager training |
| Dependences and fusion legality | `lib/passes/Fusion.cpp` | CS6843 (dependence analysis) | ✓ merges legal only if the kernel graph stays acyclic |
| Liveness-based memory planning | `lib/passes/BufferAssignment.cpp` | CS6843; register-allocation analogy | ✓ per-kernel liveness, interval colouring, in-place updates, checker |
| Roofline cost model | `lib/passes/CostModel.cpp` | CS6400 / ID5130 | ✓ FLOPs and compulsory bytes per kernel |
| Interval range analysis, bounds-check elimination | `lib/loopir/Analysis.cpp` | CS6843 A2 (range analysis), A4 (bounds checks) | ✓ branch refinement, guards for data indices, BCE |
| Dependence analysis of parallel and vector loops | `lib/loopir/Analysis.cpp` | CS6843 PAR | ✓ every parallel and vectorized loop is checked |
| Golden IR tests (lit + FileCheck) | `tests/lit/` | CS6843 / LLVM tooling | ✓ |
| Cache blocking, register tiling | `lib/loopir/Lowering.cpp` | CS6600 Computer Architecture | ✓ 4×16 register tiles; 65–95 GFLOP/s on the MNIST GEMMs |
| SIMD vectorization (AVX2) | schedules + `lib/backends/llvm/` | CS6600 / CS6400 | ✓ `vectorize(8)` loops and unrolled tiles become `<8 x float>` FMAs |
| ILP through unrolling | `lib/backends/llvm/Emitter.cpp` | CS6600 | ✓ the emitter expands unrolled loops; 64 independent accumulators per tile |
| Roofline model | `benchmarks/roofline.py`, `docs/perf-report.md` | CS6400 / ID5130 | ✓ CPU, measured on the compiled step's kernels; GPU ceilings only until the Colab run |
| Performance methodology (baselines, ablations, repeated timings) | `benchmarks/ablations.py` | CS6600 / ID5130 | ✓ medians and IQR over repeats; fusion, bounds-proof, tiling and thread ablations; PyTorch and OpenBLAS baselines |
| Thread-pool runtime | `lib/runtime/ThreadPool.cpp`, `Executable.cpp` | ID5130 (work sharing) | ✓ the outermost parallel loop of each kernel is split across threads |
| Numerical stability (log-sum-exp, NaN semantics) | `python/minidlc/reference.py`, `python/minidlc/llvm_engine.py` | ID5130 (floating point) | ✓ reference interpreter and hand-written CPU kernels |
| Shared-memory tiling, coalescing, bank conflicts | `lib/loopir/Lowering.cpp` (GPU schedules) | CS6023 GPU Programming | ◐ 64×64 block tiles staged through shared memory (A padded to 65 columns), 4×4 register tiles, column-interleaved threads for coalesced B reads; checked on the lockstep interpreter and by ptxas, not yet timed on a GPU |
| Warp-shuffle reductions | GPU schedules | CS6023 (warp primitives) | ◐ warp per row, five `shfl.sync.bfly` steps; the interpreter rejects shuffles of inactive lanes |
| Barrier semantics | `lib/loopir/Interpreter.cpp` | CS6023 (synchronization) | ✓ the lockstep interpreter rejects divergent barriers; shared memory starts as NaN, so a missing barrier shows |
| Atomic scatter (`atomicAdd`) | GPU schedules | CS6023 (atomics) | ◐ `red.global.add.f32` for scatter_add and the loss |
| PTX via LLVM NVPTX; CUDA driver API | `lib/backends/llvm/Ptx.cpp`, `python/minidlc/gpu.py` | CS6023 + LLVM | ◐ PTX for sm_75 accepted by ptxas (no spills); the driver runtime runs on the simulated device; a GPU run is pending (docs/gpu.md) |

## By course

### CS6843 Program Analysis
- **SSA and verification.** The graph IR is in SSA form. An op registry gives each op's operand count, attributes and
  type rule; the verifier checks those rules, that operands are defined before use, the return against the signature,
  and that every use list matches the actual uses. The parser and printer round-trip.
- **Translation validation.** Instead of proving each pass correct, every run is checked: each graph pass runs on
  exported graphs, and the optimized function must compute the same outputs as the original in the reference
  interpreter. Two miscompiles the suite catches (CSE merging nearby constants, `x * 0` changing a result type) are
  scheduled for P2.
- **Dataflow.** CSE numbers values by hashing (op, operand values, result type, exact attributes) within the single
  block; DCE marks values live backwards from the return. Activity analysis for autodiff intersects a forward problem
  ("varied": depends on a parameter) with a backward one ("useful": reaches the loss).
- **Memory planning.** Live intervals over the schedule, then greedy offset assignment (largest first) with an
  interference check — the interval-colouring analogy to register allocation. v2 moves it to kernel granularity, after
  fusion, with parameters in a persistent region (P3).
- **Range analysis and bounds checks (A2/A4).** Interval arithmetic over loop induction variables proves affine
  accesses in bounds, and loops whose trip count is not a multiple of the vector width are split into a vector main
  loop and a scalar tail. In v2 the loop IR has explicit guards, so the analysis deletes guards it proves redundant and
  splits GPU tiles into unguarded interior and guarded boundary blocks (P4–P5).

### CS6023 GPU Programming
- The v1 CUDA emitter produced text for shared-memory tiled GEMMs, warp-shuffle reductions and an `atomicAdd` scatter
  that was never compiled or run (its softmax combined only warp 0's partial results).
- v2 generates GPU code from the loop IR through LLVM's NVPTX backend: loops bound to blocks and threads, shared-memory
  staging with padding against bank conflicts, warp-per-row kernels with butterfly shuffles (so every lane holds the
  result and no cross-warp combine is needed), and atomics. A lockstep interpreter executes the same loop IR with
  SIMT semantics — branches split the active set, barriers must be reached by the whole block, shuffles must read
  active lanes, shared memory starts as NaN — and every op and both training models are checked on it against the
  reference. ptxas accepts the PTX with no spills. The run on a T4 (and `compute-sanitizer`) is pending: see
  `docs/gpu.md`.

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
