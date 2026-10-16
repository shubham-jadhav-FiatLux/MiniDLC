#!/usr/bin/env python3
"""
Empirical Roofline Model Analysis for MiniDLC CPU Execution.
Computes hardware bounds and plots kernels on an SVG roofline diagram.
"""

import os
import math
import numpy as np

def compute_roofline(peak_gflops: float, peak_bw_gbs: float, intensity: float) -> float:
    return min(peak_gflops, peak_bw_gbs * intensity)

def generate_svg_roofline(
    peak_gflops: float,
    peak_bw_gbs: float,
    kernels: list,
    output_path: str = "docs/roofline_cpu.svg"
):
    """Generates a clean vector SVG roofline chart."""
    ridge_point = peak_gflops / peak_bw_gbs

    # Dimensions
    width, height = 800, 500
    margin_l, margin_r, margin_t, margin_b = 80, 50, 60, 60
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    # Axis ranges (log10 scale)
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
    svg.append(f'<text x="{width//2}" y="48" fill="#7aa2f7" font-family="sans-serif" font-size="12" text-anchor="middle">Peak Compute: {peak_gflops:.1f} GFLOP/s | STREAM BW: {peak_bw_gbs:.1f} GB/s | Ridge Point: {ridge_point:.2f} FLOP/B</text>')

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
    print(f"Roofline SVG chart saved to {output_path}")

def main():
    peak_gflops = 180.0   # 4C x 16 FLOP/cycle x 2.8 GHz
    peak_bw_gbs = 13.74   # Empirical STREAM Copy bandwidth
    ridge = peak_gflops / peak_bw_gbs

    print("=== Empirical Hardware Roofline Model ===")
    print(f"Peak Compute Bandwidth:     {peak_gflops:.1f} GFLOP/s")
    print(f"Peak Memory Bandwidth:      {peak_bw_gbs:.2f} GB/s")
    print(f"Ridge Point:                {ridge:.2f} FLOP/byte")

    kernels = [
        {"name": "Fused Matmul+Bias+ReLU", "intensity": 38.50, "gflops": 9.87},
        {"name": "NumPy Unfused Matmul",   "intensity": 27.65, "gflops": 3.56},
        {"name": "Adam Update (Memory)",   "intensity": 0.50,  "gflops": 1.40},
        {"name": "ReLU Activation (Memory)","intensity": 0.25,  "gflops": 0.95},
    ]

    print("\nKernel Operational Positions:")
    for k in kernels:
        bound = "Compute-bound" if k["intensity"] >= ridge else "Memory-bound"
        attainable = compute_roofline(peak_gflops, peak_bw_gbs, k["intensity"])
        pct = 100.0 * (k["gflops"] / attainable)
        print(f"  {k['name']:<25} | Intensity: {k['intensity']:5.2f} FLOP/B | Attained: {k['gflops']:5.2f} GFLOP/s ({pct:4.1f}% roofline) | {bound}")

    generate_svg_roofline(peak_gflops, peak_bw_gbs, kernels, "docs/roofline_cpu.svg")

if __name__ == "__main__":
    main()
