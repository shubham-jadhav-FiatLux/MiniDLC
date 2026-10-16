#!/usr/bin/env python3
"""
Microbenchmarking individual neural network operations:
Unfused NumPy vs LLVM AVX2 Fused Matmul + Bias + ReLU.
Measures execution time, throughput (GFLOP/s), and memory traffic.
"""

import time
import numpy as np
import minidlc as dl
from minidlc.llvm_engine import LLVMCPUEngine

def benchmark_op(name: str, op_fn, flops: int, bytes_moved: int, warmup: int = 10, reps: int = 50):
    for _ in range(warmup):
        op_fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        op_fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    median_sec = np.median(times)
    median_ms = median_sec * 1e3
    gflops = (flops / 1e9) / median_sec
    bw_gbs = (bytes_moved / 1e9) / median_sec
    intensity = flops / bytes_moved

    print(f"{name:<35} | Time: {median_ms:6.3f} ms | {gflops:6.2f} GFLOP/s | {bw_gbs:5.2f} GB/s | Intensity: {intensity:.2f} FLOP/B")
    return {"name": name, "time_ms": median_ms, "gflops": gflops, "bw_gbs": bw_gbs, "intensity": intensity}

def main():
    print("=== MiniDLC CPU Kernel Microbenchmarks ===")
    M, K, N = 128, 784, 256
    np.random.seed(42)
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    bias = np.random.randn(N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)

    engine = LLVMCPUEngine.get()

    # Matmul FLOPs: 2 * M * K * N
    matmul_flops = 2 * M * K * N
    fused_flops = matmul_flops + 2 * M * N # bias + relu

    # Unfused bytes moved:
    # A read (M*K*4) + B read (K*N*4) + C write (M*N*4)
    # + C read (M*N*4) + bias read (N*4) + C_bias write (M*N*4)
    # + C_bias read (M*N*4) + C_relu write (M*N*4)
    bytes_unfused = (M * K + K * N + 5 * M * N + N) * 4

    # Fused bytes moved:
    # A read (M*K*4) + B read (K*N*4) + bias read (N*4) + C write (M*N*4)
    bytes_fused = (M * K + K * N + N + M * N) * 4

    print(f"Problem Shape: [{M}, {K}] @ [{K}, {N}]")
    print(f"DRAM Traffic: Unfused = {bytes_unfused / 1e3:.1f} KB, Fused = {bytes_fused / 1e3:.1f} KB (Traffic Reduction: {bytes_unfused / bytes_fused:.2f}x)\n")

    # 1. Unfused NumPy Matmul + Bias + ReLU
    def unfused_np():
        tmp = A @ B
        tmp2 = tmp + bias
        return np.maximum(tmp2, 0.0)

    res_unfused = benchmark_op("NumPy Unfused Matmul+Bias+ReLU", unfused_np, fused_flops, bytes_unfused)

    # 2. LLVM AVX2 Fused Matmul + Bias + ReLU
    def llvm_fused():
        engine.matmul_bias_relu(A.ctypes.data, B.ctypes.data, bias.ctypes.data, C.ctypes.data, M, K, N)

    res_fused = benchmark_op("MiniDLC LLVM AVX2 Fused Kernel", llvm_fused, fused_flops, bytes_fused)

    speedup = res_unfused["time_ms"] / res_fused["time_ms"]
    print(f"\nFusion Speedup: {speedup:.2f}x faster with LLVM JIT AVX2 Fused Kernel")

if __name__ == "__main__":
    main()
