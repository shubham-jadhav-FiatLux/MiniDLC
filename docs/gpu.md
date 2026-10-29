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
| Matmul (+ epilogue) | grid (⌈N/64⌉, ⌈M/64⌉, batch), block 16×16 | A and B tiles of k = 16 staged in shared memory (A transposed, padded to 65 columns against bank conflicts), barriers around each stage, a 4×4 register tile per thread over rows `ty + 16i` and columns `tx + 16j` (a warp reads consecutive words of the B tile), the epilogue applied to the tile in registers. Transpose flags change only the tile loads' indexing |
| Elementwise (multi-output) | one thread per element, 256 per block | collapsed dimensions, broadcasting by strides |
| Row (softmax, log-softmax, layernorm, cross-entropy) | one warp (32 threads) per row | each lane strides over the row; five `shfl.sync.bfly` steps leave the reduction in every lane. Cross-entropy: lane 0 adds the row's share of the mean loss with an atomic (an invalid label adds NaN), after a fill kernel zeroes the result |
| Reduction | warp per output if the reduced axes are a suffix of at least 32 elements, else a thread per output | |
| Gather (embedding, one-hot), transpose, fill | one thread per element | data-dependent indices guarded |
| Scatter (`scatter_add`) | fill kernel + one thread per update element | `red.global.add.f32` |

Edge guards (`if (i < n)`) appear only where the grid overshoots the iteration space; bounds-check elimination removes
the matmul guards when M and N are multiples of 64 and K of 16. The same range analysis as on the CPU proves every
remaining access in bounds.

## Checking it without a GPU

- **`gpu-sim`** (`dl.compile(..., target="gpu-sim")`): the plan runs on the host, each kernel in the lockstep GPU
  interpreter. Within a block, every statement executes for all active threads before the next; an `if` splits the
  active set; a barrier reached by part of the block is an error (divergent barrier), as is a shuffle that reads an
  inactive lane; shared memory starts as NaN, so a missing barrier or a read of an unwritten element shows in the
  results. Every op is checked against the reference on it (`tests/python/test_cpu_backend.py` runs on `cpu`,
  `interp` and `gpu-sim`), and so are MLP and Char-GPT training (`test_cpu_training.py`).
- **ptxas**: `tests/python/test_gpu_backend.py` assembles the PTX of a Char-GPT step for sm_75 when ptxas is
  available (`pip install nvidia-cuda-nvcc-cu12` provides one; or set `PTXAS`). On the MNIST step: no spills, 0-byte
  stack frames, 64 registers and 8,256 bytes of shared memory per matmul block.
- **The runtime** (`minidlc/gpu.py`) is tested on `SimDevice`, which implements the device interface with host memory
  and the lockstep interpreter: slots, launches, copies, state reset, saved artifacts.
- **Golden tests**: `tests/lit/cuda_*.dlc` check the GPU loop IR, the NVPTX LLVM IR and the PTX.

```bash
./build/dlc-opt step.dlc --target=gpu --emit-loopir       # loop IR with the GPU schedules
./build/dlc-opt step.dlc --target=gpu --print-loop-report # accesses proven / guarded, launch configurations
./build/dlc-opt step.dlc --target=gpu --emit-llvm-opt     # NVPTX LLVM IR after O3
./build/dlc-opt step.dlc --target=gpu --emit-ptx --arch=sm_75
```

## Running on a GPU (Colab)

The runtime needs only NumPy and the CUDA driver, so a GPU machine does not need LLVM or the compiled module: the
artifacts are built here and committed under `artifacts/gpu/` (plan, PTX, initial weights, source graph): a training
step and an evaluation function per model. The Char-GPT step is 316 kernels; ptxas accepts all of them with no
spills.

1. In Colab: *Runtime → Change runtime type → T4 GPU*.
2. In a cell:

   ```bash
   !git clone https://github.com/shubham-jadhav-FiatLux/MiniDLC.git
   %cd MiniDLC
   !PYTHONPATH=python python3 examples/colab_gpu.py all --epochs=2 --steps=2000 --json=gpu_run.json
   ```

   `all` runs the MNIST MLP (2 epochs) and Char-GPT (d_model 128, 4 heads, 4 layers, context 64, batch 16; 2,000
   steps with the held-out loss every 200 and the held-out bigram baseline); `mnist` or `chargpt` runs one.

   For each model the script checks that the artifact's initial weights are the model definition's, compares the
   first training steps' losses with the NumPy reference interpreter from the same weights and batches, trains and
   times it, evaluates (MNIST test accuracy; Char-GPT held-out loss), and profiles the kernels. Everything is printed
   and written to `gpu_run.json`.
3. Optional, memory and race checks: `!compute-sanitizer --tool memcheck python3 examples/colab_gpu.py mnist --epochs=1`
   (and `--tool racecheck`), with `PYTHONPATH=python`.

To rebuild the artifacts after changing the compiler: `PYTHONPATH=python python3 examples/gpu_artifacts.py`.
On a machine with both the compiled module and a GPU, `dl.compile(loss, optimizer, target="cuda")` compiles and runs
in one process.

## Numerics

NVPTX has no libm. `exp` and `log` go through `ex2.approx.f32` and `lg2.approx.f32` (as CUDA's `__expf` and `__logf`
do; relative error about 2⁻²², growing with |x| for `exp`), `tanh` through `exp(-2|x|)`; `sqrt` and division are the
IEEE-rounded `sqrt.rn` and `div.rn`. Float atomics (`red.global.add.f32`) flush subnormals, and their order — hence
the rounding of the loss and of embedding gradients — varies between runs. The GPU-vs-reference check uses a relative
tolerance of 1e-3.

## Status

Built and tested as above; **not yet run on GPU hardware.** Until `examples/colab_gpu.py` has run on a GPU, the
P5 criterion "MNIST and Char-GPT train on a Colab T4 and match the reference" is unmet, and no GPU timings exist.
