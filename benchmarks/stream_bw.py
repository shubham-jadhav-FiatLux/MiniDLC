#!/usr/bin/env python3
"""
STREAM-like microbenchmark measuring sustainable memory bandwidth.
Evaluates Copy (c = a) and Triad (a = b + scalar * c).
"""

import time
import numpy as np

def run_stream_benchmark(num_elements: int = 10_000_000, trials: int = 10):
    size_bytes = num_elements * 4  # float32
    print(f"Allocating {size_bytes / 1e6:.1f} MB per array (Total ~{3 * size_bytes / 1e6:.1f} MB)...")

    a = np.ones(num_elements, dtype=np.float32)
    b = np.full(num_elements, 2.0, dtype=np.float32)
    c = np.full(num_elements, 0.5, dtype=np.float32)
    scalar = np.float32(3.0)

    # 1. Copy: c = a (reads a, writes c -> 2 transfers)
    times_copy = []
    for _ in range(trials):
        t0 = time.perf_counter()
        np.copyto(c, a)
        t1 = time.perf_counter()
        times_copy.append(t1 - t0)

    min_copy_time = min(times_copy)
    copy_bw_gb_s = (2 * size_bytes) / (min_copy_time * 1e9)
    print(f"Copy Bandwidth:  {copy_bw_gb_s:.2f} GB/s (min time: {min_copy_time * 1e3:.2f} ms)")

    # 2. Triad: a = b + scalar * c (reads b, c; writes a -> 3 transfers)
    times_triad = []
    for _ in range(trials):
        t0 = time.perf_counter()
        np.add(b, scalar * c, out=a)
        t1 = time.perf_counter()
        times_triad.append(t1 - t0)

    min_triad_time = min(times_triad)
    triad_bw_gb_s = (3 * size_bytes) / (min_triad_time * 1e9)
    print(f"Triad Bandwidth: {triad_bw_gb_s:.2f} GB/s (min time: {min_triad_time * 1e3:.2f} ms)")

if __name__ == "__main__":
    run_stream_benchmark()
