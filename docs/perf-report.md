# MiniDLC Performance Report

This report documents the empirical hardware baseline, kernel-level roofline characterization, memory traffic reduction via fusion, and end-to-end training performance for MiniDLC on CPU (Milestone M2).

---

## 1. Hardware Architecture Baseline

All measurements were collected on the host Linux execution platform:
- **CPU**: Intel x86_64 architecture (4 physical cores / 8 threads)
- **Vector Extensions**: AVX2 (256-bit registers = 8 single-precision floats) + FMA3 (`vfmadd213ps`, `vfmadd231ps`)
- **Peak FP32 Compute**: $180.0 \text{ GFLOP/s}$ (4 cores $\times$ 16 FLOP/cycle $\times$ 2.8 GHz base/all-core)
- **Empirical Memory Bandwidth (STREAM Copy)**: $13.74 \text{ GB/s}$ ($5.82 \text{ ms}$ for 80 MB transfer)
- **Empirical Triad Bandwidth**: $2.80 \text{ GB/s}$
- **Hardware Ridge Point**: $I^* = \frac{\text{Peak Compute}}{\text{Peak Memory BW}} = \frac{180.0 \text{ GFLOP/s}}{13.74 \text{ GB/s}} = \mathbf{13.10 \text{ FLOP/byte}}$

---

## 2. Empirical CPU Roofline Model

Operational intensity determines the performance ceiling:
$$\text{Attainable Performance (GFLOP/s)} = \min\left(180.0, 13.74 \times I\right)$$

![CPU Roofline Diagram](roofline_cpu.svg)

### Measured Kernel Operational Points

| Kernel | Regime | Arithmetic Intensity (FLOP/B) | Attained Perf (GFLOP/s) | Attained BW (GB/s) | % of Roofline |
|---|---|:---:|:---:|:---:|:---:|
| **MiniDLC LLVM AVX2 Fused Matmul+Bias+ReLU** | Compute-bound | **38.50** | **9.87** | 0.26 | 5.5% |
| **NumPy Unfused Matmul+Bias+ReLU** | Compute-bound | 27.65 | 3.56 | 0.13 | 2.0% |
| **Adam Optimizer Update** | Memory-bound | 0.50 | 1.40 | 2.80 | 20.4% |
| **ReLU Activation** | Memory-bound | 0.25 | 0.95 | 3.80 | 27.7% |

---

## 3. Epilogue Fusion & Memory Traffic Reduction

For the layer transformation $[128, 784] \times [784, 256] + [256] \to \text{ReLU}$:
- **Unfused Data Movement**:
  1. Read $A [128, 784]$ ($401.4 \text{ KB}$) + Read $B [784, 256]$ ($802.8 \text{ KB}$) $\to$ Write $C_{\text{mm}} [128, 256]$ ($131.1 \text{ KB}$)
  2. Read $C_{\text{mm}}$ ($131.1 \text{ KB}$) + Read $\text{bias} [256]$ ($1.0 \text{ KB}$) $\to$ Write $C_{\text{bias}}$ ($131.1 \text{ KB}$)
  3. Read $C_{\text{bias}}$ ($131.1 \text{ KB}$) $\to$ Write $C_{\text{relu}}$ ($131.1 \text{ KB}$)
  - **Total DRAM Traffic**: $\mathbf{1860.6 \text{ KB}}$

- **MiniDLC Fused Epilogue Data Movement**:
  - Accumulation registers hold dot products in L1/registers.
  - Read $A$ ($401.4 \text{ KB}$) + Read $B$ ($802.8 \text{ KB}$) + Read $\text{bias}$ ($1.0 \text{ KB}$) $\to$ Write final $C_{\text{relu}}$ ($131.1 \text{ KB}$).
  - **Total DRAM Traffic**: $\mathbf{1336.3 \text{ KB}}$
  - **DRAM Traffic Reduction**: $\mathbf{1.39\times}$ memory traffic reduction
  - **Kernel Execution Speedup**: $\mathbf{2.77\times}$ speedup ($14.46 \text{ ms} \to 5.21 \text{ ms}$)

---

## 4. Memory Planning Arena Reuse

For the training step graph:
- **Unoptimized Sequential Allocation**: $1.64 \text{ MB}$
- **Planned Unified Arena Allocation**: $1.21 \text{ MB}$
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
