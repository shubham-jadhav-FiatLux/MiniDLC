#!/usr/bin/env python3
"""
Microbenchmarking individual neural network operations (matmul, activations, layernorm).
"""

import time
import numpy as np

def benchmark_op(name: str, op_fn, warmup: int = 5, reps: int = 20):
    for _ in range(warmup):
        op_fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        op_fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    median_ms = np.median(times) * 1e3
    print(f"{name:<25} | Median Time: {median_ms:.3f} ms")

def main():
    print("Running primitive operations microbenchmarks:")
    M, K, N = 128, 784, 256
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    bias = np.random.randn(N).astype(np.float32)

    benchmark_op(f"Matmul ({M}x{K}x{N})", lambda: A @ B)
    benchmark_op("Fused Matmul+Bias+ReLU", lambda: np.maximum((A @ B) + bias, 0.0))

if __name__ == "__main__":
    main()
