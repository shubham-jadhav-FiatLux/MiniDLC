#!/usr/bin/env python3
"""
GPU Roofline Model Analysis for MiniDLC CUDA Execution.
Calculates arithmetic intensities, data traffic, and hardware execution bounds
for training graph kernels on NVIDIA GPU architectures (NVIDIA T4 and A100).
Generates an SVG roofline diagram in docs/roofline_gpu.svg.
"""

import os
import sys
import math
import numpy as np

# GPU Architecture Profiles
GPU_PROFILES = {
    "NVIDIA T4 (Turing)": {
        "peak_gflops": 8100.0,       # 8.1 TFLOP/s FP32
        "peak_bw_gbs": 320.0,        # 320 GB/s GDDR6
        "ridge_point": 8100.0 / 320.0,  # 25.31 FLOP/B
        "color": "#f7768e",
    },
    "NVIDIA A100 (Ampere FP32)": {
        "peak_gflops": 19500.0,      # 19.5 TFLOP/s FP32 Vector
        "peak_bw_gbs": 1555.0,       # 1,555 GB/s HBM2e
        "ridge_point": 19500.0 / 1555.0,  # 12.54 FLOP/B
        "color": "#7aa2f7",
    }
}

def get_kernel_specs():
    """
    Computes exact arithmetic intensity and memory traffic for MiniDLC kernels
    on the MNIST MLP training graph (batch size 128, 784 -> 256 -> 10).
    """
    kernels = []
    
    # 1. Fused Matmul + Bias + ReLU (M=128, K=784, N=256)
    M, K, N = 128, 784, 256
    flops_fused = 2 * M * K * N + 2 * M * N
    bytes_fused = (M * K + K * N + N + M * N) * 4
    i_fused = flops_fused / bytes_fused
    kernels.append({
        "name": "Fused Matmul+Bias+ReLU (Fwd)",
        "intensity": i_fused,
        "flops": flops_fused,
        "bytes": bytes_fused,
        "regime": "Compute-bound",
        "color": "#7aa2f7"
    })

    # 2. Unfused Matmul + Bias + ReLU (baseline)
    bytes_unfused = (M * K + K * N + 5 * M * N + N) * 4
    i_unfused = flops_fused / bytes_unfused
    kernels.append({
        "name": "Unfused Matmul+Bias+ReLU",
        "intensity": i_unfused,
        "flops": flops_fused,
        "bytes": bytes_unfused,
        "regime": "Compute-bound",
        "color": "#e0af68"
    })

    # 3. Matmul Transpose A: dW1 = X^T @ dH_pre (M=784, K=128, N=256)
    M3, K3, N3 = 784, 128, 256
    flops_transA = 2 * M3 * K3 * N3
    bytes_transA = (M3 * K3 + K3 * N3 + M3 * N3) * 4
    i_transA = flops_transA / bytes_transA
    kernels.append({
        "name": "Matmul TransA (dW1)",
        "intensity": i_transA,
        "flops": flops_transA,
        "bytes": bytes_transA,
        "regime": "Compute-bound",
        "color": "#9ece6a"
    })

    # 4. Matmul Transpose B: dH = dlogits @ W2^T (M=128, K=10, N=256)
    M4, K4, N4 = 128, 10, 256
    flops_transB = 2 * M4 * K4 * N4
    bytes_transB = (M4 * K4 + K4 * N4 + M4 * N4) * 4
    i_transB = flops_transB / bytes_transB
    kernels.append({
        "name": "Matmul TransB (dH)",
        "intensity": i_transB,
        "flops": flops_transB,
        "bytes": bytes_transB,
        "regime": "Memory-bound",
        "color": "#bb9af7"
    })

    # 5. Bias Backward: sum(dlogits, axis=0) (M=128, N=256)
    flops_bias = M * N
    bytes_bias = (M * N + N) * 4
    i_bias = flops_bias / bytes_bias
    kernels.append({
        "name": "Bias Backward (Reduction)",
        "intensity": i_bias,
        "flops": flops_bias,
        "bytes": bytes_bias,
        "regime": "Memory-bound",
        "color": "#f7768e"
    })

    # 6. Fused Adam Optimizer Update: p, g, m, v for W1 (200,704 params)
    n_params = 784 * 256
    flops_adam = 11 * n_params
    bytes_adam = 6 * n_params * 4  # read p, g, m, v; write p, m, v
    i_adam = flops_adam / bytes_adam
    kernels.append({
        "name": "Fused GPU Adam Update",
        "intensity": i_adam,
        "flops": flops_adam,
        "bytes": bytes_adam,
        "regime": "Memory-bound",
        "color": "#2ac3de"
    })

    # 7. ReLU Backward (M=128, N=256)
    n_relu = 128 * 256
    flops_relu = n_relu
    bytes_relu = 3 * n_relu * 4  # read dOut, act; write dIn
    i_relu = flops_relu / bytes_relu
    kernels.append({
        "name": "ReLU Backward",
        "intensity": i_relu,
        "flops": flops_relu,
        "bytes": bytes_relu,
        "regime": "Memory-bound",
        "color": "#ff9e64"
    })

    # 8. Softmax Cross-Entropy (M=128, N=10)
    flops_sm = 128 * 10 * 5  # max, sub, exp, sum, div
    bytes_sm = (128 * 10 * 2) * 4
    i_sm = flops_sm / bytes_sm
    kernels.append({
        "name": "Row-wise Softmax Loss",
        "intensity": i_sm,
        "flops": flops_sm,
        "bytes": bytes_sm,
        "regime": "Memory-bound",
        "color": "#b4f9f8"
    })

    return kernels

def compute_roofline(peak_gflops: float, peak_bw_gbs: float, intensity: float) -> float:
    return min(peak_gflops, peak_bw_gbs * intensity)

def generate_svg_roofline_gpu(output_path: str = "docs/roofline_gpu.svg"):
    t4 = GPU_PROFILES["NVIDIA T4 (Turing)"]
    a100 = GPU_PROFILES["NVIDIA A100 (Ampere FP32)"]
    kernels = get_kernel_specs()

    width, height = 860, 560
    margin_l, margin_r, margin_t, margin_b = 90, 50, 70, 70
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    x_min, x_max = -1.5, 2.0  # 0.03 to 100 FLOP/byte
    y_min, y_max = 1.0, 4.5   # 10 to 31,622 GFLOP/s

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

    # Header
    svg.append(f'<text x="{width//2}" y="30" fill="#c0caf5" font-family="sans-serif" font-size="18" font-weight="bold" text-anchor="middle">Theoretical &amp; Attainable GPU Roofline Model (MiniDLC)</text>')
    svg.append(f'<text x="{width//2}" y="50" fill="#7aa2f7" font-family="sans-serif" font-size="12" text-anchor="middle">NVIDIA T4: 8.1 TFLOP/s FP32, 320 GB/s (Ridge: 25.3 FLOP/B) | A100: 19.5 TFLOP/s FP32, 1,555 GB/s (Ridge: 12.5 FLOP/B)</text>')

    # Grid lines & ticks - X Axis
    for exp in [-1, 0, 1, 2]:
        val = 10 ** exp
        px = x_to_px(val)
        svg.append(f'<line x1="{px}" y1="{margin_t}" x2="{px}" y2="{height - margin_b}" stroke="#292e42" stroke-width="1"/>')
        label = f"{val:g}" if val >= 1 else f"{val:.1f}"
        svg.append(f'<text x="{px}" y="{height - margin_b + 22}" fill="#565f89" font-family="sans-serif" font-size="11" text-anchor="middle">{label}</text>')

    # Grid lines & ticks - Y Axis
    for exp in [1, 2, 3, 4]:
        val = 10 ** exp
        py = y_to_px(val)
        svg.append(f'<line x1="{margin_l}" y1="{py}" x2="{width - margin_r}" y2="{py}" stroke="#292e42" stroke-width="1"/>')
        label = f"{val:g}"
        svg.append(f'<text x="{margin_l - 12}" y="{py + 4}" fill="#565f89" font-family="sans-serif" font-size="11" text-anchor="end">{label}</text>')

    # Axis Labels
    svg.append(f'<text x="{margin_l + plot_w//2}" y="{height - 18}" fill="#a9b1d6" font-family="sans-serif" font-size="13" text-anchor="middle">Arithmetic Intensity (FLOP/byte)</text>')
    svg.append(f'<text x="25" y="{margin_t + plot_h//2}" fill="#a9b1d6" font-family="sans-serif" font-size="13" text-anchor="middle" transform="rotate(-90 25 {margin_t + plot_h//2})">Performance (GFLOP/s)</text>')

    # Draw A100 Ceiling
    pts_a100 = []
    for log_x in np.linspace(x_min, x_max, 200):
        intensity = 10 ** log_x
        perf = compute_roofline(a100["peak_gflops"], a100["peak_bw_gbs"], intensity)
        pts_a100.append(f"{x_to_px(intensity):.1f},{y_to_px(perf):.1f}")
    svg.append(f'<polyline points="{" ".join(pts_a100)}" fill="none" stroke="{a100["color"]}" stroke-width="2.5" stroke-dasharray="6,4" opacity="0.75"/>')
    svg.append(f'<text x="{width - margin_r - 10}" y="{y_to_px(a100["peak_gflops"]) - 8}" fill="{a100["color"]}" font-family="sans-serif" font-size="11" text-anchor="end">A100 Peak: 19.5 TFLOP/s</text>')

    # Draw T4 Ceiling
    pts_t4 = []
    for log_x in np.linspace(x_min, x_max, 200):
        intensity = 10 ** log_x
        perf = compute_roofline(t4["peak_gflops"], t4["peak_bw_gbs"], intensity)
        pts_t4.append(f"{x_to_px(intensity):.1f},{y_to_px(perf):.1f}")
    svg.append(f'<polyline points="{" ".join(pts_t4)}" fill="none" stroke="{t4["color"]}" stroke-width="3" stroke-linecap="round"/>')
    svg.append(f'<text x="{width - margin_r - 10}" y="{y_to_px(t4["peak_gflops"]) - 8}" fill="{t4["color"]}" font-family="sans-serif" font-size="11" font-weight="bold" text-anchor="end">NVIDIA T4 Peak: 8.1 TFLOP/s</text>')

    # Ridge Point line for T4
    rx = x_to_px(t4["ridge_point"])
    ry = y_to_px(t4["peak_gflops"])
    svg.append(f'<line x1="{rx}" y1="{ry}" x2="{rx}" y2="{height - margin_b}" stroke="#bb9af7" stroke-width="1.5" stroke-dasharray="4,4"/>')
    svg.append(f'<circle cx="{rx}" cy="{ry}" r="5" fill="#bb9af7"/>')
    svg.append(f'<text x="{rx + 6}" y="{ry + 20}" fill="#bb9af7" font-family="sans-serif" font-size="11">T4 Ridge: {t4["ridge_point"]:.1f} F/B</text>')

    # Kernel Operational Points
    label_offsets = [
        (10, -10),   # Fused matmul
        (10, 15),    # Unfused matmul
        (10, -12),   # Matmul TransA
        (10, -10),   # Matmul TransB
        (10, -8),    # Bias Backward
        (10, 15),    # Fused Adam
        (10, -10),   # ReLU backward
        (10, 15),    # Softmax
    ]

    for i, k in enumerate(kernels):
        intensity = k["intensity"]
        attainable_perf = compute_roofline(t4["peak_gflops"], t4["peak_bw_gbs"], intensity)
        kx = x_to_px(intensity)
        ky = y_to_px(attainable_perf)
        col = k["color"]

        svg.append(f'<circle cx="{kx}" cy="{ky}" r="5.5" fill="{col}" stroke="#ffffff" stroke-width="1.5"/>')
        dx, dy = label_offsets[i % len(label_offsets)]
        
        perf_str = f"{attainable_perf/1000.0:.2f} TF/s" if attainable_perf >= 1000 else f"{attainable_perf:.0f} GF/s"
        svg.append(f'<text x="{kx + dx}" y="{ky + dy}" fill="{col}" font-family="sans-serif" font-size="10.5" font-weight="bold">{k["name"]} ({perf_str})</text>')

    svg.append("</svg>")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(svg))
    print(f"GPU Roofline SVG diagram successfully written to {output_path}")

def main():
    print("=== MiniDLC GPU Roofline Analysis ===")
    kernels = get_kernel_specs()
    t4 = GPU_PROFILES["NVIDIA T4 (Turing)"]
    a100 = GPU_PROFILES["NVIDIA A100 (Ampere FP32)"]

    print("\n1. GPU Hardware Architecture Baselines:")
    print(f"   - NVIDIA T4:   {t4['peak_gflops']:.0f} GFLOP/s FP32 Peak | {t4['peak_bw_gbs']:.0f} GB/s DRAM BW | Ridge Point: {t4['ridge_point']:.2f} FLOP/B")
    print(f"   - NVIDIA A100: {a100['peak_gflops']:.0f} GFLOP/s FP32 Peak | {a100['peak_bw_gbs']:.0f} GB/s HBM2e BW | Ridge Point: {a100['ridge_point']:.2f} FLOP/B")

    print("\n2. Workload Kernel Roofline Bounds on NVIDIA T4:")
    print(f"{'Kernel':<32} | {'Intensity':>10} | {'FLOPs':>10} | {'Bytes Moved':>11} | {'T4 Ceil (GFLOP/s)':>17} | {'Regime'}")
    print("-" * 98)
    for k in kernels:
        ceil_t4 = compute_roofline(t4['peak_gflops'], t4['peak_bw_gbs'], k['intensity'])
        print(f"{k['name']:<32} | {k['intensity']:8.2f} F/B | {k['flops']:10d} | {k['bytes']:11d} | {ceil_t4:15.1f} GF/s | {k['regime']}")

    generate_svg_roofline_gpu("docs/roofline_gpu.svg")

if __name__ == "__main__":
    main()
