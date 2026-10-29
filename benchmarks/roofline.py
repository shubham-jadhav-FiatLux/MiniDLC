#!/usr/bin/env python3
"""
Empirical CPU roofline for MiniDLC's generated code.
Measures the FP32 peak (AVX2 FMA microbenchmark, all cores) and memory bandwidth (STREAM copy), then places the
slowest kernels of the compiled MNIST training step on the roofline, with times from the runtime's per-kernel timers
and FLOPs/bytes from the compiler's cost model. Writes docs/roofline_cpu.svg.
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


def benchmark_kernels(steps: int = 300, top: int = 6) -> list:
    """The kernels of the compiled MNIST training step (batch 128, Adam), timed by the runtime's per-kernel timers
    over `steps` steps; FLOPs and compulsory bytes come from the compiler's cost model. The `top` slowest kernels."""
    import minidlc as dl
    from minidlc import nn, optim

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 256)
            self.fc2 = nn.Linear(256, 10)

        def forward(self, x):
            return self.fc2(dl.relu(self.fc1(x)))

    np.random.seed(42)
    model = MLP()
    x = dl.placeholder((128, 784), name="x")
    y = dl.placeholder((128,), dtype="i32", name="y")
    step = dl.compile(dl.cross_entropy(model(x), y), optim.Adam(model.parameters(), lr=1e-3), target="cpu",
                      profile=True)
    rng = np.random.default_rng(0)
    xb, yb = rng.standard_normal((128, 784)).astype(np.float32), rng.integers(0, 10, 128).astype(np.int32)
    for _ in range(20):
        step(x=xb, y=yb)
    step.executable.clear_profile()
    for _ in range(steps):
        step(x=xb, y=yb)
    kernels = []
    for p in sorted(step.get_profile(), key=lambda p: -p["seconds"])[:top]:
        seconds = p["seconds"] / p["calls"]
        kernels.append({"name": p["name"], "time_ms": 1e3 * seconds, "gflops": p["flops"] / seconds / 1e9,
                        "bw_gbs": p["bytes"] / seconds / 1e9, "intensity": p["flops"] / max(p["bytes"], 1)})
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
    svg.append(f'<text x="{width//2}" y="48" fill="#7aa2f7" font-family="sans-serif" font-size="12" text-anchor="middle">All-core FMA peak: {peak_gflops:.1f} GFLOP/s | STREAM Copy BW: {peak_bw_gbs:.1f} GB/s | Ridge Point: {ridge_point:.2f} FLOP/B</text>')

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

    # The compiled kernels run on every core, so the roof is the all-core FMA peak.
    ridge = multi_peak / peak_bw
    print(f"\n3. Empirical Baseline:")
    print(f"   All-Core Peak Compute:    {multi_peak:.2f} GFLOP/s")
    print(f"   Peak Memory Bandwidth:    {peak_bw:.2f} GB/s")
    print(f"   Hardware Ridge Point:     {ridge:.2f} FLOP/byte")

    print("\n4. Profiling the compiled MNIST training step...")
    kernels = benchmark_kernels()

    print("\n=== Kernel Empirical Results ===")
    print(f"{'Kernel':<34} | {'Intensity':>9} | {'Time':>9} | {'Attained':>10} | {'Bandwidth':>9} | {'% Roofline':>10} | {'Regime'}")
    print("-" * 106)
    for k in kernels:
        bound = "Compute-bound" if k["intensity"] >= ridge else "Memory-bound"
        attainable = compute_roofline(multi_peak, peak_bw, k["intensity"])
        pct = 100.0 * (k["gflops"] / attainable)
        print(f"{k['name'][:34]:<34} | {k['intensity']:7.2f} F/B | {k['time_ms']:6.3f} ms | {k['gflops']:7.2f} GF/s | {k['bw_gbs']:6.2f} GB/s | {pct:8.1f}%   | {bound}")

    generate_svg_roofline(multi_peak, peak_bw, kernels, "docs/roofline_cpu.svg")


if __name__ == "__main__":
    main()
