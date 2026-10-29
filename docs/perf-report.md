# MiniDLC Performance Report

Measured on the development laptop: Intel i7-8550U (4 cores / 8 threads, AVX2 + FMA, 8 MB L3), Arch Linux, LLVM 23,
pip NumPy 2.5 (OpenBLAS), PyTorch 2.15 CPU. The laptop's clock depends on its power state and on other load, so
P7's timings are medians over repeats with the spread given (median ± half the interquartile range), and the load
average is recorded with each run. Every number comes from a script in this repository, named with each table.

**What is being measured.** Since P4 the `cpu` target runs code the MiniDLC compiler generates: the training step is
one graph (forward, backward, optimizer), fused, scheduled, emitted through LLVM and JIT-compiled. No MiniDLC code has
run on a GPU yet (Section 6). The 2026-10-17 audit withdrew several earlier claims (Section 9).

## 1. Hardware baseline

| Measurement | Value | Command |
|---|---|---|
| Single-core FP32 peak (AVX2 FMA microbenchmark) | 112–124 GFLOP/s | `build/dlc_fma_peak 1` |
| All-core FP32 peak | 206–448 GFLOP/s; 229 in the run of Section 2 (power-limited: bursts reach full turbo, sustained load throttles) | `build/dlc_fma_peak 4` |
| Single-thread copy bandwidth (NumPy, 40 MB arrays) | 10–15 GB/s; 14.8 in the run of Section 2 | `python3 benchmarks/stream_bw.py` |

The "triad" bandwidth printed by `stream_bw.py` is not a property of the hardware (NumPy evaluates `b + s·c` in two
passes); use copy bandwidth. Bandwidth is single-threaded, so it understates what eight threads can draw.

## 2. The compiled MNIST step on the CPU roofline

Command: `PYTHONPATH=python python3 benchmarks/roofline.py` (writes `docs/roofline_cpu.svg`): the slowest kernels of
the compiled MNIST training step, timed by the runtime's per-kernel timers over 300 steps (8 threads), with FLOPs
and compulsory bytes from the compiler's cost model. 2026-10-29, load average 2.4.

| Kernel | Intensity (FLOP/B) | Time per step | Attained | Share of the roof |
|---|:---:|:---:|:---:|:---:|
| forward layer 1: matmul + bias + ReLU (+ the mask for backward) | 35.1 | 0.79 ms | 65 GFLOP/s | 28% of 229 GFLOP/s |
| dW1 = xᵀ·dh with W1's Adam update in the epilogue | 10.1 | 0.54 ms | 100 GFLOP/s | 66% |
| dh = dlogits·W2ᵀ, masked | 2.5 | 0.07 ms | 9 GFLOP/s | 26% |
| dW2 with W2's Adam update | 3.5 | 0.06 ms | 12 GFLOP/s | 23% |
| forward layer 2 | 4.5 | 0.05 ms | 13 GFLOP/s | 19% |
| softmax | 0.6 | 0.03 ms | 0.2 GFLOP/s | 3% |

The memory roof uses single-thread DRAM bandwidth, but these working sets (≤ 3.2 MB) sit in L3, so "memory-bound"
here means "below the ridge", not "limited by DRAM". The two large GEMMs get 65–100 GFLOP/s; the small kernels are
dominated by fixed costs (a launch, a parallel split over a few rows), which is why they sit far below either roof.

## 3. Fusion

Fusion is measured as an ablation of the whole training step (Section 8): with every op in its own kernel, the MNIST
step goes from 15 kernels to 77 and from 1.88 to 2.95 ms (1.6×). The v1 measurement this section used to hold — the
same hand-written GEMM with and without its epilogue, 0.99–1.00× — still stands for that case: at 128×784×256 the
GEMM is ~99% of the work and its 128 KB intermediate stays in cache. Fusion pays in the step because it removes many
small memory-bound kernels (the Adam updates, the backward elementwise chains), not because it speeds up the GEMM.

## 4. Memory: fusion and buffer assignment

The v1 figure (1.64 → 1.21 MB, 26.2%) could not be reproduced, and the v1 planner was unsafe (liveness on the unfused
op schedule, parameters reused). P3 rebuilt both passes; these numbers are for the **training step** — forward,
backward and Adam update in one graph, as `dlc-opt --passes="autodiff,adam{...},canonicalize,cse,dce"` builds it —
from `benchmarks/memory_plan.py` and `dlc-opt --print-cost`. They describe the plan that the generated code executes.

| Training step | Kernels | Parameters | Persistent region | Transient arena | Without reuse | Saved | In place / copied |
|---|---:|---:|---:|---:|---:|---:|---:|
| MNIST MLP (batch 128) | 15 | 795.0 KiB | 2386.2 KiB | 781.5 KiB | 798.5 KiB | 2.1% | 14 / 0 |
| Char-GPT (example config, batch 4) | 157 | 119.8 KiB | 376.2 KiB | 725.5 KiB | 1966.0 KiB | 63.1% | 113 / 0 |

- The persistent region holds the parameters, the Adam moments (3× the parameters) and two bias-correction scalars;
  every update is written **in place** (each Adam update is fused into the epilogue of the matmul that computes the
  gradient, which reads the parameter only at the element it writes).
- The transient arena is small for the MLP because its few large buffers are live at the same time (the input batch
  stays live until the first layer's weight gradient). In Char-GPT, liveness lets 157 kernels share 36% of the space.
- Fusion cuts the compulsory traffic of the MLP step from 31.4 MB (each op reading its operands and writing its result)
  to 7.6 MB (each kernel reading its inputs and writing its outputs), 75.7% less.
- The checker (`BufferAssignment::verify`) runs on every assignment: no two buffers that are live at the same time
  overlap, and every in-place update runs after all other readers of its argument.

## 5. MNIST MLP training

Model 784 → 256 → 10 with ReLU, cross-entropy, Adam (lr 1.5e-3), batch 128.

**Per step** (`benchmarks/ablations.py`: 7 repeats × 20 steps after a warm-up, the same batch; 2026-10-29, load
average 2.5 → 2.8):

| Implementation | ms per step | vs MiniDLC |
|---|:---:|:---:|
| **MiniDLC `cpu`**, generated code, 8 threads | **1.88 ± 0.10** | 1.0× |
| PyTorch 2.15 CPU, 4 threads | 2.23 ± 0.08 | 1.18× slower |
| PyTorch CPU, 1 thread | 4.11 ± 0.14 | |
| MiniDLC `cpu`, 1 thread | 5.29 ± 0.15 | |
| NumPy reference interpreter (OpenBLAS, multithreaded) | 6.12 ± 0.76 | 3.3× slower |
| v1 hand-written AVX2 kernels (one thread, scalar backward GEMM) | 64.8 ± 1.2 | 34× slower |

**Two epochs** (936 steps, data loading included; `examples/mnist_mlp.py 2 [--target=...]`; single runs, so ranges):
`cpu` 0.9–2.0 s and 96.81–96.92% test accuracy; `reference` 5.4 s, 96.84%; `handwritten` 31.6–64.8 s, 96.88%. Test
accuracy is on 9,984 of the 10,000 test images (the last partial batch is dropped).

Single-threaded, PyTorch's step is 1.3× faster than MiniDLC's (its GEMMs come from a tuned BLAS — Section 8's GEMM
ladder); with threads, MiniDLC's whole-step fusion and in-place updates make up for that.

## 6. GPU

**No MiniDLC code has run on a GPU yet.** The GPU path is built and checked without one (`docs/gpu.md`): every op and
two training models match the reference on the lockstep GPU interpreter, and ptxas (sm_75) accepts every kernel with
no spills — the MNIST matmul kernels use 64 registers and 8,256 bytes of shared memory per block. GPU timings will come
from `examples/colab_gpu.py`, which writes them to `gpu_run.json`.

`docs/roofline_gpu.svg` (from `python3 benchmarks/roofline_gpu.py`) shows theoretical ceilings only: datasheet
rooflines for the T4 and A100, and for each kernel of the MNIST training step the ceiling its arithmetic intensity
allows.

| | NVIDIA T4 (`sm_75`) | NVIDIA A100 (`sm_80`) |
|---|:---:|:---:|
| FP32 peak | 8.1 TFLOP/s | 19.5 TFLOP/s |
| Memory bandwidth | 320 GB/s | 1,555 GB/s |
| Ridge point | 25.3 FLOP/B | 12.5 FLOP/B |

## 7. Character-level Transformer

**P6 (2026-10-29): compiled, multi-head, against PyTorch.** Model: 2 pre-LayerNorm blocks, d_model 64, 4 heads of 16
channels, context 32, batch 16, Adam 3e-3. Data: TinyShakespeare, trained on the first 90%, evaluated on 40 fixed
batches of the last 10%. Both runs start from the same weights and see the same batches
(`benchmarks/chargpt_heldout.py 2000 --targets=cpu,torch`).

| Steps | MiniDLC `cpu` (held-out nats) | PyTorch 2.x CPU, float32 |
|---|---|---|
| 0 | 4.2248 | 4.2248 |
| 500 | 2.2041 | 2.2041 |
| 1,000 | 2.0555 | 2.0554 |
| 1,500 | 1.9994 | 1.9995 |
| 2,000 | **1.9306** | **1.9328** |

The curves differ by at most 0.0022 nats. Held-out baselines fitted on the training split (add-one smoothing):
unigram 3.3473, bigram 2.4819 nats, so the model beats the bigram by 0.55 nats. In float64, the reference interpreter's
loss and every parameter gradient match PyTorch's to 1e-7 relative (`tests/python/test_pytorch_crosscheck.py`); the
compiled step's Adam trajectory matches PyTorch's on `cpu` and `gpu-sim`. Step times are in Section 8.

**Before P6** (single-head attention: `MultiHeadAttention` did not split heads; reference interpreter): model of 2
blocks, d_model 32, context 32, 30,656 parameters.

| Run | Result | Command |
|---|---|---|
| 40-step demo, batch 4, Adam 8e-3 | Training loss 4.4626 → 3.1908 in 0.5 s (single-batch losses are noisy: 3.57 at step 35) | `examples/char_gpt.py` |
| 2,000 steps, batch 16, Adam 3e-3, 90/10 train/held-out split | **Held-out loss 2.1005 nats** after 101 s | `benchmarks/chargpt_heldout.py` |
| Unigram / bigram entropy of the text | 3.3128 / 2.4526 nats | printed by the same script |

Held-out loss by step: 0 → 4.1913 · 250 → 2.4953 · 500 → 2.3833 · 1,000 → 2.2296 · 1,500 → 2.1550 · 2,000 → 2.1005.

The 40-step demo ends near the unigram baseline: it has learned character frequencies, not context. The 2,000-step run
goes below the bigram baseline, so it uses context. Sample after 2,000 steps (temperature 0.8, prompt `ROMEO:`):
"Clord, dree thould a that us have teth to kreir fink, thes so eld halords, / Who here laut ilsch briver, With dime add
tho".

## 8. Baselines and ablations

Command: `PYTHONPATH=python python3 benchmarks/ablations.py --json=benchmarks/results/ablations.json`; 7 repeats × 20
steps, median ± half the interquartile range; 2026-10-29, load average 2.5 → 2.8. Every MiniDLC variant computes the
same losses as the default (checked by the script).

**Ablations of the MNIST training step:**

| Variant | ms per step | Slowdown | What it removes |
|---|:---:|:---:|---|
| MiniDLC `cpu` (8 threads) | 1.88 ± 0.10 | — | |
| no fusion: a kernel per op (77 kernels instead of 15) | 2.95 ± 0.48 | 1.6× | epilogues, elementwise chains, in-register Adam updates |
| run-time bounds checks on every access | 3.96 ± 0.27 | 2.1× | the range analysis that proves every access in bounds statically |
| untiled matmuls (a scalar dot product per output) | 42.4 ± 1.2 | 22× | register tiling, the unrolled FMAs LLVM vectorizes |
| untiled matmuls, 1 thread | 148.1 ± 0.7 | 79× | the above and parallel loops |

**Thread scaling** (same step): 1 thread 5.29 ms, 2 threads 3.08 (1.7×), 4 threads 2.61 (2.0×), 8 hardware threads
1.88 (2.8×). Four physical cores; the small kernels gain nothing from threads, and the laptop lowers its clock as more
cores run.

**GEMM ladder** — the first layer, 128×784 @ 784×256 + bias + ReLU, 51.4 MFLOP:

| Implementation | ms | GFLOP/s |
|---|:---:|:---:|
| MiniDLC, untiled, 1 thread | 95.3 ± 4.3 | 0.5 |
| MiniDLC, untiled, 8 threads | 27.7 ± 0.6 | 1.9 |
| MiniDLC, 4×16 register tiles, 1 thread | 2.54 ± 0.04 | 20.2 |
| MiniDLC, 4×16 register tiles, 8 threads | 0.92 ± 0.03 | 56.1 |
| PyTorch, 1 thread | 1.26 ± 0.03 | 41.0 |
| NumPy (OpenBLAS), multithreaded | 0.50 ± 0.03 | 102.3 |
| PyTorch, 4 threads | 0.44 ± 0.01 | 118.0 |

Register tiling is a 37× step (the untiled loop walks a column of W per output, a cache-hostile stride). The tiled
kernel reaches about half of a tuned BLAS: it does not pack panels of W for cache blocking, and at 128 rows its
parallel loop has only 32 row blocks to split.

**Char-GPT training step** (d_model 64, 4 heads, 2 layers, context 32, batch 16, Adam; 171 kernels): MiniDLC
11.06 ± 0.12 ms; PyTorch CPU 14.31 ± 0.05 ms with 4 threads and 22.59 ± 0.04 ms with 1. Here the many small ops
favour a compiler that fuses them and plans memory once over an eager framework.

## 9. Correctness tests

| Suite | Count | What it checks |
|---|:---:|---|
| GoogleTest (`tests/unit`) | 96 | IR round trips, malformed input, a mutation fuzzer, the passes, loop IR analyses, the CPU JIT, the GPU interpreter's SIMT semantics, the compiler |
| lit + FileCheck (`tests/lit`) | 19 | `dlc-opt` after passes, reports, loop IR, LLVM IR, PTX |
| Translation validation, C++ training graph (`tests/python`) | 207 | Every graph pass in float64; every C++ VJP against finite differences; step graphs vs eager training |
| Generated code (`test_cpu_backend.py`) | 63 | Every op through the generated code on `cpu`, `interp` and `gpu-sim`, against the reference |
| Compiled training (`test_cpu_training.py`) | 15 | Compiled trajectories (`cpu`, `gpu-sim`) vs eager training |
| GPU runtime and PTX (`test_gpu_backend.py`) | 8 | The CUDA runtime on the simulated device, artifacts, PTX contents, ptxas; a real-GPU test that runs when a GPU is present |
| PyTorch cross-check (`test_pytorch_crosscheck.py`) | 3 | Char-GPT loss and gradients (float64) and compiled Adam trajectories against PyTorch |

## 10. Withdrawn claims (2026-10-17 audit)

- **"2.27× faster than the reference interpreter"** (MNIST): measured against a netlib-BLAS NumPy (Section 5).
- **"2.63–2.77× kernel speedup from epilogue fusion":** compared the MiniDLC GEMM with a netlib-BLAS GEMM. Same-kernel
  fusion gives 1.00× (Section 3).
- **"26.2% peak memory reduction":** not reproducible, and the planner is unsafe (Section 4).
- **"LLVM ORC JIT", "ThreadPool multi-core dispatch", "32×64×32 tiling", "unroll ×4":** none is used on any executed path.
  The kernels are compiled by a `clang` subprocess and run single-threaded; the loop passes only set attributes.
- **GPU kernels plotted as operating points:** they were ceilings; no GPU code has run (Section 6).
- **Char-GPT "2 attention heads" and "loss drops monotonically from 4.33 to 3.15":** attention was single-head until P6; that trace
  was not monotonic and not reproducible (unseeded initialisation, now fixed) (Section 7).
- **"All figures empirically measured without theoretical assumptions":** traffic and intensity figures use a
  compulsory-traffic model.
