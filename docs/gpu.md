# The GPU path

MiniDLC compiles a training step for NVIDIA GPUs through the same pipeline as for the CPU, up to the loop IR. There,
the GPU schedules bind loops to blocks and threads; LLVM's NVPTX backend turns the loop IR into PTX; and a small
runtime replays the plan through the CUDA driver.

```
graph IR ─ autodiff, optimizer, fusion, buffer assignment ─┐
                                                           ▼
             loop IR with GPU schedules (lowerFunction(..., Target::GPU))
               │                                    │
               ▼                                    ▼
   lockstep GPU interpreter (gpu-sim)      NVPTX: LLVM IR → O3 → PTX (sm_75)
                                                    │
                                    plan.json + kernels.ptx (a GPU artifact)
                                                    │
                              minidlc/gpu.py: CUDA driver API via ctypes
```

## Schedules

| Kernel class | Launch | Schedule |
|---|---|---|
| Matmul (+ epilogue) | grid (⌈N/64⌉, ⌈M/64⌉, batch), block 16×16 | A and B tiles of k = 16 staged in shared memory (A transposed; its row pitch, 66 words, chosen by the bank-conflict pass), barriers around each stage, a 4×4 register tile per thread over rows `ty + 16i` and columns `tx + 16j` (a warp reads consecutive words of the B tile), the epilogue applied to the tile in registers. Partial tiles: the k loop covers the full tiles, then one guarded tile; with M or N not a multiple of 64, interior blocks run an unguarded copy of the body and edge blocks a guarded one. Transpose flags change only the tile loads' indexing |
| Elementwise (multi-output) | one thread per element, 256 per block | collapsed dimensions, broadcasting by strides |
| Row (softmax, log-softmax, layernorm, cross-entropy) | one warp (32 threads) per row; a block of 256 from 1024 elements on | each thread strides over the row; five `shfl.sync.bfly` steps leave a warp's reduction in every lane; a block's warps then combine through shared memory. Cross-entropy: one thread adds the row's share of the mean loss with an atomic (an invalid label adds NaN), after a fill kernel zeroes the result |
| Reduction | warp (block from 1024) per output if the reduced axes are a suffix of at least 32 elements, else a thread per output | |
| Gather (embedding, one-hot), transpose, fill | one thread per element | data-dependent indices guarded |
| Scatter (`scatter_add`) | fill kernel + one thread per update element | `red.global.add.f32` |

Edge guards (`if (i < n)`) appear only where the grid overshoots the iteration space; bounds-check elimination removes
those that the shapes, the k-tile split and the block versioning make redundant. The same range analysis as on the CPU
proves every remaining access in bounds. The GPU loop passes — bank-conflict padding, the resource check, coalescing
statistics, simplification — are described in [`passes/gpu_passes.md`](passes/gpu_passes.md).

## Checking it without a GPU

- **`gpu-sim`** (`dl.compile(..., target="gpu-sim")`): the plan runs on the host, each kernel in the lockstep GPU
  interpreter. Within a block, every statement executes for all active threads before the next; an `if` splits the
  active set; a barrier reached by part of the block is an error (divergent barrier), as is a shuffle that reads an
  inactive lane; shared memory starts as NaN. Lockstep execution would hide a missing barrier, so the interpreter also
  checks every access for races: two threads touching a shared element between the same two barriers, or a global
  element in the same launch (between blocks too), one of them writing — unless both add atomically — is an error.
  Every op is checked against the reference on it (`tests/python/test_cpu_backend.py` runs on `cpu`, `interp` and
  `gpu-sim`), and so are MLP and Char-GPT training (`test_cpu_training.py`) and the random models of the pipeline
  fuzzer (`test_pipeline_fuzzer.py`).
- **The resource check**: launch dimensions, shared memory and register-buffer words per block against the sm_75
  limits, at compile time.
- **ptxas**: `tests/python/test_gpu_backend.py` assembles the PTX of a Char-GPT step for sm_75 when ptxas is
  available (`pip install nvidia-cuda-nvcc-cu12` provides one; or set `PTXAS`). Every committed artifact assembles
  with no spills and 0-byte stack frames: the MNIST step's matmul blocks use 55–72 registers and 8,320 bytes of shared
  memory; the Char-GPT step's 316 kernels at most 79 registers.
- **The runtime** (`minidlc/gpu.py`) is tested on `SimDevice`, which implements the device interface with host memory
  and the lockstep interpreter: slots, launches, copies, state reset, saved artifacts, checkpoints, and the profiling
  path that CUDA events take (with a host-clock stand-in for the events).
- **Golden tests**: `tests/lit/cuda_*.dlc` check the GPU loop IR (padding, versioned partial tiles, block-per-row
  reductions), the GPU report, the NVPTX LLVM IR and the PTX.
- **Committed artifacts are current**: `test_committed_artifacts_are_current` recompiles each artifact's source and
  compares its plan and the SHA-256 of its loop IR with what was committed.

```bash
./build/dlc-opt step.dlc --target=gpu --emit-loopir       # loop IR with the GPU schedules
./build/dlc-opt step.dlc --target=gpu --print-loop-report # accesses proven / guarded, launch configurations
./build/dlc-opt step.dlc --target=gpu --print-gpu-report  # + resources, padding, bank conflicts, sectors per request
./build/dlc-opt step.dlc --target=gpu --emit-llvm-opt     # NVPTX LLVM IR after O3
./build/dlc-opt step.dlc --target=gpu --emit-ptx --arch=sm_75
```

## Running on a GPU (Colab)

The runtime needs only NumPy and the CUDA driver, so a GPU machine does not need LLVM or the compiled module: the
artifacts are built here and committed under `artifacts/gpu/` (plan, PTX, initial weights, source graph): a training
step and an evaluation function per model, and two GEMMs for the comparison with cuBLAS. The Char-GPT step is 316
kernels; ptxas accepts all of them with no spills.

1. In Colab: *Runtime → Change runtime type → T4 GPU*.
2. In a cell:

   ```bash
   !git clone https://github.com/shubham-jadhav-FiatLux/MiniDLC.git
   %cd MiniDLC
   !PYTHONPATH=python python3 examples/colab_gpu.py all --epochs=2 --steps=2000 --json=gpu_run.json --checkpoint=ckpt
   ```

   `all` runs the MNIST MLP (2 epochs), Char-GPT (d_model 128, 4 heads, 4 layers, context 64, batch 16; 2,000 steps
   with the held-out loss every 200 and the held-out bigram baseline) and the baselines; `mnist`, `chargpt` or
   `baselines` runs one.

   For each model the script checks that the artifact's initial weights are the model definition's, compares the
   first training steps' losses with the NumPy reference interpreter from the same weights and batches, trains and
   times it (wall clock per step, the loss fetched every step), evaluates (MNIST test accuracy on all 10,000 images;
   Char-GPT held-out loss), saves a checkpoint (`--checkpoint`; loadable by any target), and profiles the kernels with
   CUDA events. The baselines (PyTorch, preinstalled on Colab) train the same models from the same weights in PyTorch
   on the same GPU, and time MiniDLC's GEMM kernels against cuBLAS (`torch.matmul`) on the MNIST first layer
   (128×784×256, bias and ReLU fused) and on 1024³. Everything is printed and written to `gpu_run.json`: send that
   file to fill in the GPU section of the perf report.
3. Optional, memory and race checks: `!compute-sanitizer --tool memcheck python3 examples/colab_gpu.py mnist --epochs=1`
   (and `--tool racecheck`), with `PYTHONPATH=python`.

To rebuild the artifacts after changing the compiler: `PYTHONPATH=python python3 examples/gpu_artifacts.py`.
On a machine with both the compiled module and a GPU, `dl.compile(loss, optimizer, target="cuda")` compiles and runs
in one process (the PTX comes from the kernel cache when the same kernels were compiled before).

## Numerics

NVPTX has no libm. `exp` and `log` go through `ex2.approx.f32` and `lg2.approx.f32` (as CUDA's `__expf` and `__logf`
do; relative error about 2⁻²², growing with |x| for `exp`), `tanh` through `exp(-2|x|)`; `sqrt` and division are the
IEEE-rounded `sqrt.rn` and `div.rn`. Float atomics (`red.global.add.f32`) flush subnormals, and their order — hence
the rounding of the loss and of embedding gradients — varies between runs. The GPU-vs-reference check uses a relative
tolerance of 1e-3.

## Status

Built and tested as above; **not yet run on GPU hardware.** Until `examples/colab_gpu.py` has run on a GPU, the
P5 criterion "MNIST and Char-GPT train on a Colab T4 and match the reference" is unmet, and no GPU timings exist.

## Course link

CS6023 GPU Programming: the thread hierarchy and launch configuration, shared-memory tiling and bank conflicts,
coalescing, warp shuffles, barriers and races, atomics, occupancy limits; PTX and the driver API.
