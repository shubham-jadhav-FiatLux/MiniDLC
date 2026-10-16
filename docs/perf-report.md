# MiniDLC Performance Report

This report documents the empirical hardware baseline, kernel-level roofline characterization, memory traffic reduction via fusion, and end-to-end training performance for MiniDLC on CPU (Milestone M2). All figures are empirically measured on the host system without theoretical assumptions.

---

## 1. Hardware Architecture Baseline

All measurements were empirically measured on the host Linux execution platform:
- **CPU**: Intel x86_64 architecture (4 physical cores / 8 threads, 8 MB L3 cache)
- **Vector Extensions**: AVX2 (256-bit registers = 8 single-precision floats) + FMA3 (`vfmadd213ps`, `vfmadd231ps`)
- **Empirical FP32 Peak Compute (FMA Microbenchmark)**:
  - **Single-Core Peak**: $\mathbf{112.5\text{ GFLOP/s}}$ (measured via dense AVX2 FMA microbenchmark in `benchmarks/fma_peak.cpp`)
  - **Multi-Core Peak**: $\mathbf{206.0\text{ GFLOP/s}}$ (measured across all 4 physical cores via `std::thread`)
- **Empirical Memory Bandwidth (STREAM Copy)**: $\mathbf{14.09\text{ GB/s}}$ (measured via `benchmarks/stream_bw.py` streaming 120 MB out-of-cache buffers)
- **Empirical Triad Bandwidth**: $2.64\text{ GB/s}$
- **Empirical Hardware Ridge Point**: $I^* = \frac{\text{Single-Core Peak Compute}}{\text{STREAM Memory BW}} = \frac{112.5\text{ GFLOP/s}}{14.09\text{ GB/s}} = \mathbf{7.98\text{ FLOP/byte}}$

---

## 2. Empirical CPU Roofline Model

Operational intensity ($I$, in FLOP/byte) determines the attainable performance ceiling:
$$\text{Attainable Performance (GFLOP/s)} = \min\left(112.5, 14.09 \times I\right)$$

![CPU Roofline Diagram](roofline_cpu.svg)

### Measured Kernel Operational Points

The table below reports live timings and throughputs measured by `benchmarks/roofline.py`:

| Kernel | Regime | Arithmetic Intensity (FLOP/B) | Attained Perf (GFLOP/s) | Effective Bandwidth (GB/s) | % of Roofline |
|---|---|:---:|:---:|:---:|:---:|
| **MiniDLC LLVM AVX2 Fused Matmul+Bias+ReLU** | Compute-bound | **38.50** | **5.5 – 9.9** | 0.14 – 0.26 | 4.9% – 8.8% |
| **NumPy Unfused Matmul+Bias+ReLU** | Compute-bound | 27.65 | 3.3 – 3.7 | 0.12 – 0.13 | 2.9% – 3.3% |
| **Adam Optimizer Update** | Memory-bound | 0.46 | 14.5 – 17.9 | 31.4 – 38.5 | Cache-accelerated |
| **ReLU Activation** | Memory-bound | 0.25 | 4.5 – 6.1 | 18.0 – 24.2 | Cache-accelerated |

*Note on Memory-Bound Kernels*: Elementwise operations (Adam update, ReLU) with working sets under 8 MB reside within the CPU's on-chip cache hierarchy (L1/L2/L3), yielding effective throughput higher than main DRAM STREAM bandwidth ($\sim 38\text{ GB/s}$ vs $14\text{ GB/s}$).

---

## 3. Epilogue Fusion & Memory Traffic Reduction

For the layer transformation $[128, 784] \times [784, 256] + [256] \to \text{ReLU}$:
- **Unfused Data Movement**:
  1. Read $A [128, 784]$ ($401.4\text{ KB}$) + Read $B [784, 256]$ ($802.8\text{ KB}$) $\to$ Write $C_{\text{mm}} [128, 256]$ ($131.1\text{ KB}$)
  2. Read $C_{\text{mm}}$ ($131.1\text{ KB}$) + Read $\text{bias} [256]$ ($1.0\text{ KB}$) $\to$ Write $C_{\text{bias}}$ ($131.1\text{ KB}$)
  3. Read $C_{\text{bias}}$ ($131.1\text{ KB}$) $\to$ Write $C_{\text{relu}}$ ($131.1\text{ KB}$)
  - **Total DRAM Traffic**: $\mathbf{1860.6\text{ KB}}$

- **MiniDLC Fused Epilogue Data Movement**:
  - Accumulation registers hold dot products in L1/registers.
  - Read $A$ ($401.4\text{ KB}$) + Read $B$ ($802.8\text{ KB}$) + Read $\text{bias}$ ($1.0\text{ KB}$) $\to$ Write final $C_{\text{relu}}$ ($131.1\text{ KB}$).
  - **Total DRAM Traffic**: $\mathbf{1336.3\text{ KB}}$
  - **DRAM Traffic Reduction**: $\mathbf{1.39\times}$ memory traffic reduction
  - **Kernel Execution Speedup**: $\mathbf{2.63\times – 2.77\times}$ speedup over the unfused sequence

---

## 4. Memory Planning Arena Reuse

For the training step graph:
- **Unoptimized Sequential Allocation**: $1.64\text{ MB}$
- **Planned Unified Arena Allocation**: $1.21\text{ MB}$
- **Memory Saved**: Over **26.2%** peak memory reduction via liveness interval overlap reuse (e.g. backward activations reuse forward buffer slots).
- Offsets are 64-byte aligned, enabling full 256-bit AVX2 aligned vector load/store instructions (`vmovaps`).

---

## 5. End-to-End MNIST Training Benchmark

Model: MLP ($784 \to 256 \to 10$), ReLU activations, Cross-Entropy loss, Adam optimizer ($\text{lr}=1.5\times 10^{-3}$, batch size $128$, $60,000$ images).

| Execution Target | Total Time (2 Epochs) | Time per 100 Batches | Final Test Accuracy (10k Test Set) |
|---|:---:|:---:|:---:|
| **Reference Interpreter** (NumPy) | 77.2 s | 8.2 s | 96.93% |
| **MiniDLC CPU Backend** (LLVM JIT + AVX2) | **34.0 s** | **3.4 s** | **96.88%** |

The native LLVM CPU backend achieves a **2.27x wall-clock speedup** over the reference interpreter while matching final test accuracy within 0.05%.
