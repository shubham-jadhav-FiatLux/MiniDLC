#!/usr/bin/env python3
"""
Epilogue-fusion ablation with the same GEMM kernel.

Times MiniDLC's hand-written AVX2 matmul+bias+ReLU kernel against the identical loop nest without the epilogue,
followed by the bias add and ReLU as separate passes. Both variants share the GEMM, so the difference isolates the
effect of fusion. NumPy is timed for context only: its speed depends on the BLAS it links (see docs/perf-report.md).
"""

import ctypes
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(repo_root, "python"))

from minidlc.llvm_engine import generate_cpu_kernels_llvm_ir

EPILOGUE = """  %ptr_bias = getelementptr inbounds float, ptr %bias, i64 %j
  %vec_bias = load <8 x float>, ptr %ptr_bias, align 4
  %acc_bias = fadd <8 x float> %acc_fma, %vec_bias
  %acc_relu = call <8 x float> @llvm.maximum.v8f32(<8 x float> %acc_bias, <8 x float> zeroinitializer)
"""


def build_kernels(workdir: str) -> ctypes.CDLL:
    """Compiles the engine's kernels plus a copy of the fused GEMM with its epilogue removed."""
    ir = generate_cpu_kernels_llvm_ir()
    start = ir.index("define void @kernel_matmul_bias_relu")
    end = ir.index("}\n", start) + 2
    plain = ir[start:end].replace("@kernel_matmul_bias_relu", "@kernel_matmul_plain")
    assert EPILOGUE in plain, "fused kernel changed; update EPILOGUE"
    plain = plain.replace(EPILOGUE, "").replace("store <8 x float> %acc_relu", "store <8 x float> %acc_fma")

    ll_path = os.path.join(workdir, "kernels.ll")
    so_path = os.path.join(workdir, "kernels.so")
    with open(ll_path, "w") as f:
        f.write(ir + "\n" + plain)
    subprocess.run(["clang", "-O3", "-shared", "-mavx2", "-mfma", "-fPIC", ll_path, "-o", so_path], check=True)

    lib = ctypes.CDLL(so_path)
    for name in ("kernel_matmul_plain", "kernel_matmul_bias_relu"):
        getattr(lib, name).argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int64] * 3
        getattr(lib, name).restype = None
    return lib


def median_time(fn, warmup: int = 5, reps: int = 40) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def main():
    M, K, N = 128, 784, 256
    rng = np.random.RandomState(0)
    A = rng.randn(M, K).astype(np.float32)
    B = rng.randn(K, N).astype(np.float32)
    bias = rng.randn(N).astype(np.float32)
    C = np.empty((M, N), dtype=np.float32)
    flops = 2 * M * K * N + 2 * M * N

    with tempfile.TemporaryDirectory() as workdir:
        lib = build_kernels(workdir)

        def fused():
            lib.kernel_matmul_bias_relu(A.ctypes.data, B.ctypes.data, bias.ctypes.data, C.ctypes.data, M, K, N)

        def unfused():
            lib.kernel_matmul_plain(A.ctypes.data, B.ctypes.data, None, C.ctypes.data, M, K, N)
            np.add(C, bias, out=C)
            np.maximum(C, 0.0, out=C)

        fused()
        expected = np.maximum(A.astype(np.float64) @ B + bias, 0.0)
        rel_err = np.abs(C - expected).max() / np.abs(expected).max()

        t_fused = median_time(fused)
        t_unfused = median_time(unfused)
        t_numpy = median_time(lambda: np.maximum(A @ B + bias, 0.0))

    blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]["name"]
    print(f"Problem: [{M}, {K}] @ [{K}, {N}] + bias, ReLU   (max rel. error of fused kernel: {rel_err:.1e})")
    print(f"Same AVX2 GEMM, fused epilogue    : {t_fused * 1e3:7.3f} ms  {flops / t_fused / 1e9:6.2f} GFLOP/s")
    print(f"Same AVX2 GEMM, separate epilogue : {t_unfused * 1e3:7.3f} ms  {flops / t_unfused / 1e9:6.2f} GFLOP/s")
    print(f"Fusion speedup (same GEMM)        : {t_unfused / t_fused:.2f}x")
    print(f"NumPy, BLAS = {blas:<20}: {t_numpy * 1e3:7.3f} ms  {flops / t_numpy / 1e9:6.2f} GFLOP/s")


if __name__ == "__main__":
    main()
