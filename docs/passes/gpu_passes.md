# GPU Schedules and GPU Loop Passes

## Purpose
Turn each fused kernel into a loop nest bound to blocks and threads that uses an NVIDIA GPU well — shared memory
without bank conflicts, coalesced global accesses, warp shuffles, no divergent barriers — and check, before any GPU
runs it, that the result fits the hardware and has no races. Schedules are rules per kernel class
(`lib/loopir/Lowering.cpp`); the passes are in `lib/loopir/GpuPasses.cpp`, `lib/loopir/Transforms.cpp` and the
lockstep interpreter (`lib/loopir/Interpreter.cpp`).

## Schedules
| Kernel class | Launch | Schedule |
|---|---|---|
| Matmul (+ epilogue) | grid (⌈N/64⌉, ⌈M/64⌉, batch), block 16×16 | 64×64 output tile per block, k in 16-wide tiles staged through shared memory (A transposed, its row pitch chosen by the padding pass), a 4×4 register tile per thread over rows `ty + 16i` and columns `tx + 16j`, barriers around each stage, the epilogue in registers. **Partial tiles:** the k loop runs over the full tiles, then one guarded partial tile; when M or N is not a multiple of 64 (and some blocks are full), the body is versioned on the block index — interior blocks run a copy whose guards bounds-check elimination deletes, edge blocks the guarded copy. The condition is uniform across a block, so the barriers stay convergent. |
| Elementwise | 256 threads per block | one thread per element, guard on the tail |
| Row (softmax, log-softmax, LayerNorm, cross-entropy) | one warp per row; **a block of 256 threads per row from 1024 elements on** | each thread strides over the row; five `shfl.sync.bfly` steps reduce a warp; with a block, lane 0 of each warp writes its result to a shared array of 8 partials, a barrier, every thread combines them in warp order, a barrier before the array is reused |
| Reduction | the same for reduced suffixes (warp from 32 elements, block from 1024); otherwise a thread per output | |
| Gather, scatter, transpose, fill | one thread per element | data-dependent indices guarded; scatter and the loss with `red.global.add.f32` |

## Passes, in order (per GPU kernel, in `lowerFunction`)
1. **Bank-conflict analysis and padding** (`padSharedBuffers`). A two-dimensional shared tile whose accesses use its
   row pitch through a variable `<tile>_pitch` gets the smallest pitch, from its row length up to 31 more words, with
   the fewest bank conflicts over all its accesses; the pitch is substituted into the indices and the buffer grows.
   Conflicts come from the **warp probe** (`probeWarpAccesses`): every warp of block 0 runs on its own with symbolic
   memory and the first two iterations of each loop, recording each lane's index for every request; a request's
   n-way conflict is the most distinct words that fall in one of the 32 banks. For the matmul's transposed A tile
   (16 rows of 64), a warp stores elements `(e % 16) * pitch + e / 16` for `e = lane`: pitch 64 puts them in two banks
   (16-way), 65 leaves 2-way conflicts, **66 is conflict-free** — the pass picks 66, where the hand-written schedule
   had 65.
2. **Guard insertion and bounds-check elimination**, as on the CPU ([bounds_check_elim.md](bounds_check_elim.md)):
   inside `if (by < M/64)` range analysis knows `by ≤ M/64 − 1`, so the row guards of the interior copy are proven
   true and deleted; in the k loop over full tiles, the k guards are.
3. **Dependence check** of parallel and vectorized loops (CPU loops; GPU kernels are checked dynamically, below).
4. **Simplification** ([simplify.md](simplify.md)).
5. **Resource check** (`checkResources`): block dimensions (≤ 1024, 1024, 64; ≤ 1024 threads), grid dimensions
   (≤ 2³¹−1, 65535, 65535), static shared memory (≤ 48 KiB) and register-buffer words per thread (≤ 255, and × threads
   ≤ 64 Ki) against `GpuLimits` (sm_75). A kernel the GPU could not launch is a compile error, not a launch failure:
   e.g. a batched matmul with 70,000 batches (grid.z) — `tests/lit/errors.dlc`.
6. **Coalescing statistics** (`analyzeAccesses`, for reports): per access site, 32-byte sectors per warp request and
   the fraction of fetched bytes the lanes asked for; per shared site, the worst and mean bank conflict.

```
$ ./build/dlc-opt --target=gpu --print-gpu-report tests/lit/cuda_tiled_matmul.dlc
k0_matmul_add_relu: 12 accesses, 12 proven in bounds, 0 guarded, 4 guards eliminated; grid (2, 2, 1), block (16, 16, 1)
    simplified: 0 constants propagated, 0 divisions folded, 1 invariant lets hoisted
    resources: 256 threads, 8320 bytes of shared memory, 16 register words per thread
    padding: As0 rows 64 -> 66 words (bank conflicts 16-way -> 1-way)
    global load  %a   ((ar * 64) + ak)                 32 requests, 4.0 sectors, 100% of fetched bytes used
    shared store As0  (((e % 16) * 66) + (e / 16))     32 requests, 1-way worst, 1.00-way mean
    ...
```

## Race detection (the lockstep interpreter, `target="gpu-sim"`)
Executing a block in lockstep — every statement for all threads before the next — hides missing synchronization:
with or without a barrier between a write and another thread's read, the results are the same. So the interpreter
also tracks every access, as a GPU would be free to reorder them:
- **shared memory:** per element, the last writer and the readers since the last barrier (epoch); a read of another
  thread's write, or a write after another thread's read or write, in the same epoch is a race;
- **global memory:** per element and launch, who read, wrote and atomically added; two threads touching an element,
  one of them writing (unless both add atomically) is a race — between threads of a block, and between blocks through
  a check shared by the host threads that run the blocks (`GpuLaunchCheck`; blocks are unordered). This is stricter
  than CUDA within a block, where a barrier orders global memory too; no schedule communicates that way.

Every op test and both training models run on `gpu-sim` with these checks; the unit tests show a missing barrier,
a missing second barrier (write after read) and write-write races between blocks being caught.

## Status
All schedules are checked differentially on `gpu-sim` (odd shapes, partial tiles in M, N and K, rows of 1024 to 2048
elements, broadcasting, zero-size dimensions, NaN/Inf), against the NumPy reference. ptxas (sm_75) accepts every
kernel of the committed artifacts with no spills: the Char-GPT training step's 316 kernels use at most 79 registers.
Not yet run on a GPU: `examples/colab_gpu.py` does that, with `compute-sanitizer` memcheck/racecheck (docs/gpu.md).

## Course link
CS6023 GPU Programming: shared-memory tiling and bank conflicts (L3), warp shuffles and reductions across warps (L4),
atomics (L8), occupancy and resource limits. CS6843 Program Analysis: range analysis and bounds-check elimination
with loop versioning (A2, A4), the race checker as a dynamic dependence analysis.
