#!/usr/bin/env python3
"""
Plots empirical roofline model curves for CPU and GPU architectures.
Computes maximum attainable performance: Attainable GFLOP/s = min(Peak GFLOP/s, Memory Bandwidth * I).
"""

import numpy as np

def compute_roofline(
    peak_gflops: float, peak_bw_gbs: float, intensities: np.ndarray
) -> np.ndarray:
    """Returns attainable GFLOP/s for given arithmetic intensities (FLOP/byte)."""
    return np.minimum(peak_gflops, peak_bw_gbs * intensities)

def main():
    # Example machine parameters (e.g. laptop or cluster node)
    peak_gflops = 300.0   # GFLOP/s
    peak_bw_gbs = 32.0    # GB/s
    ridge_point = peak_gflops / peak_bw_gbs

    print(f"Hardware Roofline Specifications:")
    print(f"  Peak Compute:     {peak_gflops:.1f} GFLOP/s")
    print(f"  Peak Memory BW:   {peak_bw_gbs:.1f} GB/s")
    print(f"  Ridge Point:      {ridge_point:.2f} FLOP/byte")

    intensities = np.logspace(-2, 2, 50)
    attainable = compute_roofline(peak_gflops, peak_bw_gbs, intensities)
    print("Roofline model calculation completed.")

if __name__ == "__main__":
    main()
