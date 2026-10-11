# MiniDLC Performance Report

## Methodology

Performance evaluation in MiniDLC strictly adheres to empirical roofline modeling:
1. **Hardware Roofline Limits**:
   - Measure sustainable memory bandwidth using STREAM copy and triad benchmarks.
   - Measure peak computational throughput (fp32 GFLOP/s) using dense FMA microbenchmarks.
2. **Kernel Arithmetic Intensity**:
   - Arithmetic intensity \( I = \frac{\text{Total FLOPs}}{\text{Total DRAM Bytes Transferred}} \).
   - Compute whether execution is memory-bandwidth bound or compute bound.
3. **Optimizations Measured**:
   - Naive vs. Cache-Tiled Matmul.
   - Unfused vs. Fused Elementwise Epilogues (measuring reduction in DRAM traffic).
   - Scalar vs. AVX2 SIMD Vectorized loops.
   - Thread scaling and Amdahl's Law efficiency across core counts.

## Measured Results

*This section will be populated with empirical data upon benchmarking each milestone.*

| Kernel / Model | Platform | Arithmetic Intensity (FLOP/byte) | Attained GFLOP/s | Roofline Efficiency (%) |
|---|---|---|---|---|
| Matmul (128x784x256) | CPU (x86_64) | TBD | TBD | TBD |
| Fused Bias+ReLU | CPU (x86_64) | TBD | TBD | TBD |
| MNIST Step | CPU (LLVM) | TBD | TBD | TBD |
| MNIST Step | GPU (CUDA) | TBD | TBD | TBD |
