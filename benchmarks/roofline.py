#!/usr/bin/env python3
"""
Empirical Roofline Model Analysis for MiniDLC CPU Execution.
Dynamically measures peak hardware compute (AVX2 FMA microbenchmark),
memory bandwidth (STREAM benchmark), and actual kernel operational points.
Generates an SVG roofline diagram from empirical measurements.
"""

import os
import sys
import math
import time
import subprocess
import numpy as np

# Ensure minidlc is importable
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(repo_root, "python"))
sys.path.insert(0, repo_root)

from minidlc.llvm_engine import LLVMCPUEngine
from benchmarks.stream_bw import run_stream_benchmark


def get_empirical_peak_gflops() -> dict:
    """Measures single-core and multi-core peak FP32 compute via FMA microbenchmark."""
    fma_bin = os.path.join(repo_root, "build", "dlc_fma_peak")
    if not os.path.exists(fma_bin):
        fma_bin = os.path.join(repo_root, "benchmarks", "fma_peak")
    if not os.path.exists(fma_bin):
        cpp_src = os.path.join(repo_root, "benchmarks", "fma_peak.cpp")
        print("Compiling FMA peak microbenchmark...")
        subprocess.run(
            ["clang++", "-O3", "-mavx2", "-mfma", "-pthread", cpp_src, "-o", fma_bin],
            check=True
        )

    # Measure 1 thread (single core) and 4 threads (all cores)
    res_1 = subprocess.run([fma_bin, "1"], capture_output=True, text=True, check=True)
    single_gflops = float(res_1.stdout.strip())

    res_4 = subprocess.run([fma_bin, "4"], capture_output=True, text=True, check=True)
    multi_gflops = float(res_4.stdout.strip())

    return {
        "single_core_gflops": single_gflops,
        "multi_core_gflops": multi_gflops,
    }


def benchmark_kernels(engine: LLVMCPUEngine) -> list:
    """Empirically benchmarks the key neural network operations."""
    kernels = []
    np.random.seed(42)

    # 1. MiniDLC Fused Matmul + Bias + ReLU (M=128, K=784, N=256)
    M, K, N = 128, 784, 256
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    bias = np.random.randn(N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)

    flops_fused = 2 * M * K * N + 2 * M * N
    bytes_fused = (M * K + K * N + N + M * N) * 4
    intensity_fused = flops_fused / bytes_fused

    # Warmup
    for _ in range(10):
        engine.matmul_bias_relu(A.ctypes.data, B.ctypes.data, bias.ctypes.data, C.ctypes.data, M, K, N)
    times_fused = []
    for _ in range(40):
        t0 = time.perf_counter()
        engine.matmul_bias_relu(A.ctypes.data, B.ctypes.data, bias.ctypes.data, C.ctypes.data, M, K, N)
        t1 = time.perf_counter()
        times_fused.append(t1 - t0)
    med_t_fused = float(np.median(times_fused))
    gflops_fused = (flops_fused / 1e9) / med_t_fused
    bw_fused = (bytes_fused / 1e9) / med_t_fused

    kernels.append({
        "name": "Fused Matmul+Bias+ReLU",
        "intensity": intensity_fused,
        "gflops": gflops_fused,
        "bw_gbs": bw_fused,
        "time_ms": med_t_fused * 1e3,
    })

    # 2. NumPy Unfused Matmul + Bias + ReLU
    bytes_unfused = (M * K + K * N + 5 * M * N + N) * 4
    intensity_unfused = flops_fused / bytes_unfused

    for _ in range(5):
        tmp = A @ B
        tmp2 = tmp + bias
        _ = np.maximum(tmp2, 0.0)
    times_unfused = []
    for _ in range(40):
        t0 = time.perf_counter()
        tmp = A @ B
        tmp2 = tmp + bias
        _ = np.maximum(tmp2, 0.0)
        t1 = time.perf_counter()
        times_unfused.append(t1 - t0)
    med_t_unfused = float(np.median(times_unfused))
    gflops_unfused = (flops_fused / 1e9) / med_t_unfused
    bw_unfused = (bytes_unfused / 1e9) / med_t_unfused

    kernels.append({
        "name": "NumPy Unfused Matmul",
        "intensity": intensity_unfused,
        "gflops": gflops_unfused,
        "bw_gbs": bw_unfused,
        "time_ms": med_t_unfused * 1e3,
    })

    # 3. Adam Optimizer Update (AVX2 Vectorized)
    N_elem = 784 * 256
    w = np.random.randn(N_elem).astype(np.float32)
    g = np.random.randn(N_elem).astype(np.float32)
    m = np.zeros(N_elem, dtype=np.float32)
    v = np.zeros(N_elem, dtype=np.float32)
    lr, b1, b2, eps, bc1, bc2 = 0.001, 0.9, 0.999, 1e-8, 0.1, 0.01

    flops_adam = 13 * N_elem
    bytes_adam = 7 * 4 * N_elem
    intensity_adam = flops_adam / bytes_adam

    for _ in range(10):
        engine.adam_update(w.ctypes.data, g.ctypes.data, m.ctypes.data, v.ctypes.data, lr, b1, b2, eps, bc1, bc2, N_elem)
    times_adam = []
    for _ in range(50):
        t0 = time.perf_counter()
        engine.adam_update(w.ctypes.data, g.ctypes.data, m.ctypes.data, v.ctypes.data, lr, b1, b2, eps, bc1, bc2, N_elem)
        t1 = time.perf_counter()
        times_adam.append(t1 - t0)
    med_t_adam = float(np.median(times_adam))
    gflops_adam = (flops_adam / 1e9) / med_t_adam
    bw_adam = (bytes_adam / 1e9) / med_t_adam

    kernels.append({
        "name": "Adam Update (Memory)",
        "intensity": intensity_adam,
        "gflops": gflops_adam,
        "bw_gbs": bw_adam,
        "time_ms": med_t_adam * 1e3,
    })

    # 4. ReLU Activation
    N_relu = 1_000_000
    x_relu = np.random.randn(N_relu).astype(np.float32)
    y_relu = np.empty_like(x_relu)
    flops_relu = 2 * N_relu
    bytes_relu = 8 * N_relu
    intensity_relu = flops_relu / bytes_relu

    for _ in range(10):
        np.maximum(x_relu, 0.0, out=y_relu)
    times_relu = []
    for _ in range(50):
        t0 = time.perf_counter()
        np.maximum(x_relu, 0.0, out=y_relu)
        t1 = time.perf_counter()
        times_relu.append(t1 - t0)
    med_t_relu = float(np.median(times_relu))
    gflops_relu = (flops_relu / 1e9) / med_t_relu
    bw_relu = (bytes_relu / 1e9) / med_t_relu

    kernels.append({
        "name": "ReLU Activation (Memory)",
        "intensity": intensity_relu,
        "gflops": gflops_relu,
        "bw_gbs": bw_relu,
        "time_ms": med_t_relu * 1e3,
    })

    return kernels


def compute_roofline(peak_gflops: float, peak_bw_gbs: float, intensity: float) -> float:
    return min(peak_gflops, peak_bw_gbs * intensity)


def generate_svg_roofline(
    peak_gflops: float,
    peak_bw_gbs: float,
    kernels: list,
    output_path: str = "docs/roofline_cpu.svg"
):
    """Generates an empirical vector SVG roofline chart."""
    ridge_point = peak_gflops / peak_bw_gbs

    width, height = 800, 500
    margin_l, margin_r, margin_t, margin_b = 80, 50, 60, 60
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    x_min, x_max = -2.0, 2.0  # 0.01 to 100 FLOP/byte
    y_min, y_max = -1.0, 3.0  # 0.1 to 1000 GFLOP/s

    def x_to_px(val):
        log_v = math.log10(max(val, 1e-4))
        norm = (log_v - x_min) / (x_max - x_min)
        return margin_l + norm * plot_w

    def y_to_px(val):
        log_v = math.log10(max(val, 1e-4))
        norm = (log_v - y_min) / (y_max - y_min)
        return height - margin_b - norm * plot_h

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">')
    svg.append('<rect width="100%" height="100%" fill="#1a1b26"/>')

    # Title & Subtitle
    svg.append(f'<text x="{width//2}" y="30" fill="#c0caf5" font-family="sans-serif" font-size="18" font-weight="bold" text-anchor="middle">Empirical CPU Roofline Model (MiniDLC)</text>')
    svg.append(f'<text x="{width//2}" y="48" fill="#7aa2f7" font-family="sans-serif" font-size="12" text-anchor="middle">Empirical Peak: {peak_gflops:.1f} GFLOP/s | STREAM Copy BW: {peak_bw_gbs:.1f} GB/s | Ridge Point: {ridge_point:.2f} FLOP/B</text>')

    # Grid lines & ticks
    for exp in range(int(x_min), int(x_max) + 1):
        val = 10 ** exp
        px = x_to_px(val)
        svg.append(f'<line x1="{px}" y1="{margin_t}" x2="{px}" y2="{height - margin_b}" stroke="#292e42" stroke-width="1"/>')
        label = f"{val:g}" if val >= 1 else f"{val:.2f}"
        svg.append(f'<text x="{px}" y="{height - margin_b + 20}" fill="#565f89" font-family="sans-serif" font-size="11" text-anchor="middle">{label}</text>')

    for exp in range(int(y_min), int(y_max) + 1):
        val = 10 ** exp
        py = y_to_px(val)
        svg.append(f'<line x1="{margin_l}" y1="{py}" x2="{width - margin_r}" y2="{py}" stroke="#292e42" stroke-width="1"/>')
        label = f"{val:g}" if val >= 1 else f"{val:.1f}"
        svg.append(f'<text x="{margin_l - 12}" y="{py + 4}" fill="#565f89" font-family="sans-serif" font-size="11" text-anchor="end">{label}</text>')

    # Axis Labels
    svg.append(f'<text x="{margin_l + plot_w//2}" y="{height - 15}" fill="#a9b1d6" font-family="sans-serif" font-size="13" text-anchor="middle">Arithmetic Intensity (FLOP/byte)</text>')
    svg.append(f'<text x="25" y="{margin_t + plot_h//2}" fill="#a9b1d6" font-family="sans-serif" font-size="13" text-anchor="middle" transform="rotate(-90 25 {margin_t + plot_h//2})">Performance (GFLOP/s)</text>')

    # Roofline Boundary Line
    pts = []
    for log_x in np.linspace(x_min, x_max, 200):
        intensity = 10 ** log_x
        perf = compute_roofline(peak_gflops, peak_bw_gbs, intensity)
        pts.append(f"{x_to_px(intensity):.1f},{y_to_px(perf):.1f}")
    svg.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#f7768e" stroke-width="3" stroke-linecap="round"/>')

    # Ridge Point dashed line
    rx = x_to_px(ridge_point)
    ry = y_to_px(peak_gflops)
    svg.append(f'<line x1="{rx}" y1="{ry}" x2="{rx}" y2="{height - margin_b}" stroke="#bb9af7" stroke-width="1.5" stroke-dasharray="4,4"/>')
    svg.append(f'<circle cx="{rx}" cy="{ry}" r="5" fill="#bb9af7"/>')
    svg.append(f'<text x="{rx + 8}" y="{ry - 8}" fill="#bb9af7" font-family="sans-serif" font-size="11">Ridge: {ridge_point:.2f} FLOP/B</text>')

    # Kernel Data Points
    colors = ["#7aa2f7", "#9ece6a", "#e0af68", "#bb9af7"]
    for i, k in enumerate(kernels):
        kx = x_to_px(k["intensity"])
        ky = y_to_px(k["gflops"])
        col = colors[i % len(colors)]
        svg.append(f'<circle cx="{kx}" cy="{ky}" r="6" fill="{col}" stroke="#ffffff" stroke-width="1.5"/>')
        svg.append(f'<text x="{kx + 8}" y="{ky + 4}" fill="{col}" font-family="sans-serif" font-size="11" font-weight="bold">{k["name"]} ({k["gflops"]:.1f} GFLOP/s)</text>')

    svg.append('</svg>')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(svg))
    print(f"\nEmpirical Roofline SVG chart saved to {output_path}")


def main():
    print("=== Empirical Hardware Roofline Characterization ===")
    print("1. Measuring Empirical FP32 Peak Compute (FMA Microbenchmark)...")
    fma_res = get_empirical_peak_gflops()
    single_peak = fma_res["single_core_gflops"]
    multi_peak = fma_res["multi_core_gflops"]
    print(f"   Empirical Single-Core Peak Compute: {single_peak:.2f} GFLOP/s")
    print(f"   Empirical Multi-Core Peak Compute:  {multi_peak:.2f} GFLOP/s")

    print("\n2. Measuring Empirical Memory Bandwidth (STREAM Microbenchmark)...")
    stream_res = run_stream_benchmark(num_elements=10_000_000, trials=5)
    peak_bw = stream_res["copy_bw_gb_s"]

    # For single-threaded kernels, evaluate against single-core peak compute
    ridge_single = single_peak / peak_bw
    print(f"\n3. Empirical Baseline:")
    print(f"   Single-Core Peak Compute: {single_peak:.2f} GFLOP/s")
    print(f"   Peak Memory Bandwidth:    {peak_bw:.2f} GB/s")
    print(f"   Hardware Ridge Point:     {ridge_single:.2f} FLOP/byte")

    print("\n4. Running Kernel Microbenchmarks...")
    engine = LLVMCPUEngine.get()
    kernels = benchmark_kernels(engine)

    print("\n=== Kernel Empirical Results ===")
    print(f"{'Kernel':<28} | {'Intensity':>9} | {'Time':>9} | {'Attained':>10} | {'Bandwidth':>9} | {'% Roofline':>10} | {'Regime'}")
    print("-" * 100)
    for k in kernels:
        bound = "Compute-bound" if k["intensity"] >= ridge_single else "Memory-bound"
        attainable = compute_roofline(single_peak, peak_bw, k["intensity"])
        pct = 100.0 * (k["gflops"] / attainable)
        print(f"{k['name']:<28} | {k['intensity']:7.2f} F/B | {k['time_ms']:6.3f} ms | {k['gflops']:7.2f} GF/s | {k['bw_gbs']:6.2f} GB/s | {pct:8.1f}%   | {bound}")

    generate_svg_roofline(single_peak, peak_bw, kernels, "docs/roofline_cpu.svg")


if __name__ == "__main__":
    main()
