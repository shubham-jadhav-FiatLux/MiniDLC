# Course-Concept Map

How MiniDLC uses graduate coursework from IIT Madras, and how far each concept is implemented.

Status: **✓** implemented and tested · **◐** implemented and tested without the hardware it targets (the GPU path
has not run on a GPU yet: `docs/gpu.md`) · **○** planned. Measured numbers are in [`perf-report.md`](perf-report.md).

## Summary

| Concept | Where | Course | Status |
|---|---|---|---|
| SSA form, IR design, verifier, textual IR | `include/dlc/ir/`, `lib/ir/` | CS6843 Program Analysis | ✓ op registry; verifier with per-op type rules, dominance and use-list checks; parser that round-trips |
| Value numbering (CSE) | `lib/passes/CSE.cpp` | CS6843 (available expressions) | ✓ keyed on operand identity, result type and exact attribute bits |
| Reverse reachability (DCE) | `lib/passes/DCE.cpp` | CS6843 (liveness) | ✓ erased ops are unlinked from their operands' use lists |
| Abstract interpretation over shapes | `lib/passes/ShapeInference.cpp`, `lib/ir/OpRegistry.cpp` | CS6843 | ✓ registry type rules; invalid programs rejected |
| Translation validation of passes | `tests/python/test_translation_validation.py` | CS6843 (semantics preservation) | ✓ every graph pass on 16 exported graphs |
| Differential testing and fuzzing | `tests/python/test_pipeline_fuzzer.py`, `test_cpu_backend.py`, `test_fuzzer.py` | CS6843 (testing compilers) | ✓ random models through the whole pipeline on every target against the reference, training included; a parser/pass mutation fuzzer |
| Reverse-mode autodiff (VJPs) | `python/minidlc/reference.py`, `lib/passes/Autodiff.cpp` | Chain rule / adjoint method | ✓ Python and C++; every C++ VJP checked against float64 finite differences; whole models against PyTorch |
| Activity analysis | `lib/passes/Activity.cpp` | CS6843 (dataflow) | ✓ varied ∩ useful; no gradient for the input batch |
| Optimizer lowering, buffer donation | `lib/passes/Optimizer.cpp` | CS6843 (program transformation, aliasing) | ✓ Adam/SGD in the graph; step graph matches eager training |
| Dependences and fusion legality | `lib/passes/Fusion.cpp` | CS6843 (dependence analysis) | ✓ merges legal only if the kernel graph stays acyclic |
| Liveness-based memory planning | `lib/passes/BufferAssignment.cpp` | CS6843; register-allocation analogy | ✓ per-kernel liveness, interval colouring, in-place updates, checker |
| Roofline cost model | `lib/passes/CostModel.cpp` | CS6400 / ID5130 | ✓ FLOPs and compulsory bytes per kernel |
| Interval range analysis, bounds-check elimination | `lib/loopir/Analysis.cpp` | CS6843 A2 (range analysis), A4 (bounds checks) | ✓ branch refinement, guards for data indices, BCE |
| Loop versioning of partial tiles | `lib/loopir/Lowering.cpp` | CS6843 A4 | ✓ CPU tail tiles as branches; GPU: interior blocks run an unguarded copy (BCE deletes its guards), the k loop splits off its partial tile |
| Dependence analysis of parallel and vector loops | `lib/loopir/Analysis.cpp` | CS6843 PAR | ✓ every parallel and vectorized loop is checked |
| Loop simplification: constant propagation, range-based folding, LICM | `lib/loopir/Transforms.cpp` | CS6843 (dataflow) | ✓ on every kernel, after the checks |
| Golden IR tests (lit + FileCheck) | `tests/lit/` | CS6843 / LLVM tooling | ✓ |
| Cache blocking, register tiling | `lib/loopir/Lowering.cpp` | CS6600 Computer Architecture | ✓ 4×16 register tiles; the MNIST step's two large GEMMs at 65–100 GFLOP/s |
| SIMD vectorization (AVX2) | schedules + `lib/backends/llvm/` | CS6600 / CS6400 | ✓ `vectorize(8)` loops and unrolled tiles become `<8 x float>` FMAs |
| ILP through unrolling | `lib/backends/llvm/Emitter.cpp` | CS6600 | ✓ the emitter expands unrolled loops; 64 independent accumulators per tile |
| Roofline model | `benchmarks/roofline.py`, `docs/perf-report.md` | CS6400 / ID5130 | ✓ CPU, measured on the compiled step's kernels; GPU ceilings only until the Colab run |
| Performance methodology (baselines, ablations, repeated timings) | `benchmarks/ablations.py` | CS6600 / ID5130 | ✓ medians and IQR over repeats; fusion, bounds-proof, tiling and thread ablations; PyTorch and OpenBLAS baselines |
| Thread-pool runtime | `lib/runtime/ThreadPool.cpp`, `Executable.cpp` | ID5130 (work sharing) | ✓ the outermost parallel loop of each kernel (and a simulated launch's blocks) split across threads |
| Numerical stability (log-sum-exp, NaN semantics) | schedules, `python/minidlc/reference.py` | ID5130 (floating point) | ✓ max-subtracted softmax and log-sum-exp; NaN-propagating max; contract tests on every target |
| Shared-memory tiling | `lib/loopir/Lowering.cpp` (GPU schedules) | CS6023 GPU Programming | ◐ 64×64 block tiles staged through shared memory, 4×4 register tiles, barriers |
| Bank-conflict analysis and padding | `lib/loopir/GpuPasses.cpp` | CS6023 (shared memory banks) | ◐ a warp probe measures n-way conflicts per access; the pass pads the A tile to 66 words (16-way → conflict-free; the hand-picked 65 left 2-way) |
| Coalescing | `lib/loopir/GpuPasses.cpp` | CS6023 (memory transactions) | ◐ sectors per warp request and bytes used per access (`--print-gpu-report`) |
| Resource limits (block, grid, shared memory, registers) | `lib/loopir/GpuPasses.cpp` | CS6023 (occupancy) | ✓ checked against sm_75 at compile time |
| Warp-shuffle reductions, cross-warp combine | GPU schedules | CS6023 (warp primitives, reductions) | ◐ warp per row with five `shfl.sync.bfly` steps; rows from 1024 elements: a block, warps combined through shared memory |
| Barriers and races | `lib/loopir/Interpreter.cpp` | CS6023 (synchronization) | ✓ the lockstep interpreter rejects divergent barriers and detects races in shared memory (per barrier interval) and global memory (per launch, across blocks) |
| Atomic scatter (`atomicAdd`) | GPU schedules | CS6023 (atomics) | ◐ `red.global.add.f32` for scatter_add and the loss |
| PTX via LLVM NVPTX; CUDA driver API | `lib/backends/llvm/Ptx.cpp`, `python/minidlc/gpu.py` | CS6023 + LLVM | ◐ PTX for sm_75 accepted by ptxas (no spills); the driver runtime (CUDA events for timing) runs on the simulated device; a GPU run is pending (docs/gpu.md) |

## By course

### CS6843 Program Analysis
- **SSA and verification.** The graph IR is in SSA form. An op registry gives each op's operand count, attributes and
  type rule; the verifier checks those rules, that operands are defined before use, the return against the signature,
  and that every use list matches the actual uses. The parser and printer round-trip.
- **Translation validation and differential testing.** Instead of proving each pass correct, every run is checked:
  each graph pass runs on exported graphs, and the optimized function must compute the same outputs as the original
  in the reference interpreter; generated code is compared with the reference on every target; a fuzzer builds random
  models and trains them through the whole compiler against eager training in the reference.
- **Dataflow.** CSE numbers values by hashing (op, operand values, result type, exact attributes) within the single
  block; DCE marks values live backwards from the return. Activity analysis for autodiff intersects a forward problem
  ("varied": depends on a parameter) with a backward one ("useful": reaches the loss). The loop simplifier propagates
  constants and hoists loop-invariant `let`s.
- **Memory planning.** Live intervals over the kernels after fusion, then greedy offset assignment (largest first) —
  the interval-colouring analogy to register allocation — with parameters and optimizer state in a persistent region,
  in-place updates of donated parameters, and a checker that proves no two live buffers overlap.
- **Range analysis and bounds checks (A2/A4).** Interval arithmetic over loop variables, integer locals and branch
  conditions proves accesses in bounds; indices loaded from memory get guards; guards the analysis proves are deleted.
  Partial tiles are versioned: CPU tail tiles are separate branches, and GPU matmuls run an unguarded copy of the body
  in interior blocks, where the analysis removes every edge guard.
- **Dependence analysis.** Parallel loops must write disjoint elements; vectorized loops must not carry a dependence
  through memory. On the GPU the check is dynamic: the lockstep interpreter records the accesses of every thread and
  reports conflicting ones that no barrier orders.

### CS6023 GPU Programming
- v2 generates GPU code from the loop IR through LLVM's NVPTX backend: loops bound to blocks and threads,
  shared-memory staging, a bank-conflict pass that measures each access's conflicts with a warp probe and pads tiles
  (the matmul's transposed A tile: 66 words, conflict-free), warp-per-row kernels with butterfly shuffles and, for long
  rows, a block whose warps combine through shared memory, atomics, and a resource check against the sm_75 limits.
- A lockstep interpreter executes the same loop IR with SIMT semantics — branches split the active set, barriers must
  be reached by the whole block, shuffles must read active lanes, shared memory starts as NaN — and detects races, so a
  missing barrier is an error rather than a hidden one. Every op, both training models and the fuzzer's random models
  are checked on it against the reference. ptxas accepts every kernel with no spills.
- The runtime drives the CUDA driver through ctypes and times kernels with CUDA events. The run on a T4 (with
  `compute-sanitizer`) and the comparison with cuBLAS and PyTorch are pending: see `docs/gpu.md`.

### CS6600 / CS6400 Computer Architecture
- The CPU schedules tile matmuls into 4×16 register tiles (eight AVX2 accumulators) with in-register epilogues,
  vectorize elementwise and row kernels, and unroll tiles for ILP; LLVM turns them into `<8 x float>` FMAs.
- The roofline is measured on the development laptop (FMA peak, STREAM copy) and the compiled step's kernels are
  placed on it. Ablations measure what tiling (22×), fusion (1.6×) and static bounds proofs (2.1×) are worth.

### ID5130 Parallel Scientific Computing
- Numerics: softmax and cross-entropy subtract the row maximum before `exp`; the cross-entropy gradient is computed as
  `(softmax − onehot) / N`; ReLU propagates NaN as PyTorch does. Contract tests cover these on every target.
- The thread pool splits each kernel's outermost parallel loop into one chunk per thread; thread scaling of the MNIST
  step (1 → 8 threads: 5.29 → 1.88 ms) is in the perf report, with Amdahl's law for the serial kernels.
