#!/usr/bin/env python3
"""Baselines and ablations for P7, on the CPU, with repeated timings.

Every row is timed as R repeats of N steps after a warm-up; the table gives the median time per step and the
interquartile range across repeats, and the load average is recorded before and after (the numbers mean little on a
busy machine). All MiniDLC variants compute the same losses; the script checks that.

  - MNIST MLP training step (784-256-10, batch 128, Adam): MiniDLC, its ablations (no fusion; run-time bounds checks
    on every access instead of proofs; no loop simplification; untiled matmuls), thread scaling, and baselines: PyTorch CPU, the NumPy
    reference interpreter (with the NumPy's BLAS), and v1's hand-written AVX2 kernels.
  - A GEMM ladder for the first layer (128x784 @ 784x256 + bias + ReLU): untiled vs register-tiled, 1 thread vs all,
    against OpenBLAS (NumPy) and PyTorch.
  - Char-GPT training step (d_model 64, 4 heads, 2 layers, context 32, batch 16): MiniDLC vs PyTorch.

Usage: PYTHONPATH=python python3 benchmarks/ablations.py [--repeats=7] [--steps=20] [--json=ablations.json] [--quick]
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time

import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(repo_root, "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import minidlc as dl  # noqa: E402
from minidlc import nn, optim  # noqa: E402

ARGS = sys.argv[1:]
QUICK = "--quick" in ARGS
REPEATS = int(next((a.split("=")[1] for a in ARGS if a.startswith("--repeats=")), 3 if QUICK else 7))
STEPS = int(next((a.split("=")[1] for a in ARGS if a.startswith("--steps=")), 5 if QUICK else 20))
HW_THREADS = os.cpu_count() or 1


def timed(fn, steps=None, repeats=None, warmup=3):
    steps, repeats = steps or STEPS, repeats or REPEATS
    for _ in range(warmup):
        fn()
    per_step = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(steps):
            fn()
        per_step.append((time.perf_counter() - t0) / steps)
    q1, med, q3 = np.percentile(per_step, [25, 50, 75])
    return {"median_ms": 1e3 * med, "iqr_ms": 1e3 * (q3 - q1), "min_ms": 1e3 * min(per_step)}


def row(table, name, result, extra=""):
    table.append({"name": name, **result, "note": extra})
    print(f"  {name:44s} {result['median_ms']:9.3f} ms  ±{result['iqr_ms'] / 2:7.3f}  (min {result['min_ms']:.3f}) {extra}",
          flush=True)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        return self.fc2(dl.relu(self.fc1(x)))


def mnist_batch():
    rng = np.random.default_rng(0)
    return rng.standard_normal((128, 784)).astype(np.float32), rng.integers(0, 10, 128).astype(np.int32)


def mnist_step(target, options=None, threads=0):
    np.random.seed(0)
    model = MLP()
    x = dl.placeholder((128, 784), name="x")
    y = dl.placeholder((128,), dtype="i32", name="y")
    return dl.compile(dl.cross_entropy(model(x), y), optim.Adam(model.parameters(), lr=1e-3), target=target,
                      threads=threads, options=options), model


def torch_mlp_step(model, threads):
    import torch
    torch.set_num_threads(threads)
    w1, b1, w2, b2 = (torch.tensor(p.data, requires_grad=True) for p in model.parameters())
    adam = torch.optim.Adam([w1, b1, w2, b2], lr=1e-3)
    xb, yb = mnist_batch()
    x, y = torch.tensor(xb), torch.tensor(yb, dtype=torch.long)

    def step():
        adam.zero_grad()
        loss = torch.nn.functional.cross_entropy(torch.relu(x @ w1 + b1) @ w2 + b2, y)
        loss.backward()
        adam.step()
        return loss.item()
    return step


def mnist(table):
    print(f"\nMNIST MLP training step (batch 128, Adam); {HW_THREADS} hardware threads", flush=True)
    xb, yb = mnist_batch()
    losses = {}

    def run(name, target="cpu", options=None, threads=0, note=""):
        step, _ = mnist_step(target, options, threads)
        losses[name] = [step(x=xb, y=yb) for _ in range(2)]
        row(table, name, timed(lambda: step(x=xb, y=yb)), note)

    run("MiniDLC cpu", note="(all threads)")
    for t in (1, 2, 4):
        if t < HW_THREADS:
            run(f"MiniDLC cpu, {t} thread{'s' if t > 1 else ''}", threads=t)
    run("ablation: no fusion (a kernel per op)", options={"fuse": False})
    run("ablation: run-time checks on every access", options={"check_all": True})
    run("ablation: no loop simplification", options={"simplify": False})
    run("ablation: untiled matmuls", options={"matmul_schedule": "naive"})
    run("ablation: untiled matmuls, 1 thread", options={"matmul_schedule": "naive"}, threads=1)
    reference = losses["MiniDLC cpu"]
    for name, got in losses.items():
        assert np.allclose(got, reference, rtol=1e-5), (name, got, reference)
    try:
        _, model = mnist_step("reference")
        for t in sorted({1, HW_THREADS // 2}):
            row(table, f"PyTorch CPU, {t} thread{'s' if t > 1 else ''}", timed(torch_mlp_step(model, t)))
    except ImportError:
        print("  (PyTorch not installed)")
    step, _ = mnist_step("reference")
    blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]["name"]
    row(table, f"NumPy reference interpreter ({blas})", timed(lambda: step(x=xb, y=yb), steps=max(2, STEPS // 4)))
    step, _ = mnist_step("handwritten")
    row(table, "v1 hand-written AVX2 kernels", timed(lambda: step(x=xb, y=yb), steps=max(2, STEPS // 4)))


def gemm(table):
    print("\nGEMM ladder: 128x784 @ 784x256 + bias + ReLU (51.4 MFLOP)", flush=True)
    flops = 2 * 128 * 784 * 256
    rng = np.random.default_rng(0)
    xb = rng.standard_normal((128, 784)).astype(np.float32)

    def layer(options, threads):
        np.random.seed(0)
        model = MLP()
        x = dl.placeholder((128, 784), name="x")
        f = dl.compile(dl.relu(model.fc1(x)), target="cpu", threads=threads, options=options)
        return f, model

    def gflops(r):
        return f"{flops / (r['median_ms'] * 1e-3) / 1e9:6.1f} GFLOP/s"

    want = None
    for name, options, threads in [("untiled, 1 thread", {"matmul_schedule": "naive"}, 1),
                                   ("untiled, all threads", {"matmul_schedule": "naive"}, 0),
                                   ("4x16 register tiles, 1 thread", None, 1),
                                   ("4x16 register tiles, all threads", None, 0)]:
        f, model = layer(options, threads)
        out = f(x=xb)
        want = out if want is None else want
        assert np.allclose(out, want, rtol=1e-4, atol=1e-4)
        r = timed(lambda: f(x=xb), steps=STEPS * 5)
        row(table, f"MiniDLC {name}", r, gflops(r))
    w, b = model.fc1.weight.data, model.fc1.bias.data
    r = timed(lambda: np.maximum(xb @ w + b, 0), steps=STEPS * 5)
    row(table, "NumPy (OpenBLAS), all threads", r, gflops(r))
    try:
        import torch
        tx, tw, tb = torch.tensor(xb), torch.tensor(w), torch.tensor(b)
        for t in (1, HW_THREADS // 2):
            torch.set_num_threads(t)
            r = timed(lambda: torch.relu(tx @ tw + tb), steps=STEPS * 5)
            row(table, f"PyTorch, {t} thread{'s' if t > 1 else ''}", r, gflops(r))
    except ImportError:
        pass


def chargpt(table):
    print("\nChar-GPT training step (d_model 64, 4 heads, 2 layers, context 32, batch 16, Adam)", flush=True)
    rng = np.random.default_rng(0)
    xb, yb = (rng.integers(0, 65, (16, 32)).astype(np.int32) for _ in range(2))

    def build():
        np.random.seed(0)
        model = nn.CharGPT(vocab_size=65, block_size=32, d_model=64, num_heads=4, num_layers=2)
        x = dl.placeholder((16, 32), dtype="i32", name="x")
        y = dl.placeholder((16, 32), dtype="i32", name="y")
        return model, dl.cross_entropy(model(x), y)

    model, loss = build()
    step = dl.compile(loss, optim.Adam(model.parameters(), lr=3e-3), target="cpu")
    row(table, "MiniDLC cpu", timed(lambda: step(x=xb, y=yb)), f"({len(step.executable.info['launches'])} kernels)")
    try:
        import torch
        from torch_chargpt import TorchCharGPT
        for t in (1, HW_THREADS // 2):
            model, _ = build()
            ref = TorchCharGPT(model, threads=t)
            adam = torch.optim.Adam(list(ref.params.values()), lr=3e-3)

            def torch_step():
                adam.zero_grad()
                out = ref.loss(xb, yb)
                out.backward()
                adam.step()
            row(table, f"PyTorch CPU, {t} thread{'s' if t > 1 else ''}", timed(torch_step))
    except ImportError:
        pass


def main():
    load_before = os.getloadavg()
    print(f"{platform.processor() or platform.machine()}; load average {load_before[0]:.2f}; "
          f"{REPEATS} repeats x {STEPS} steps; median ± half the interquartile range", flush=True)
    results = {"machine": platform.platform(), "threads": HW_THREADS, "load_before": load_before}
    for name, fn in (("mnist", mnist), ("gemm", gemm), ("chargpt", chargpt)):
        table = []
        fn(table)
        results[name] = table
    results["load_after"] = os.getloadavg()
    print(f"\nload average after: {results['load_after'][0]:.2f}")
    path = next((a.split("=", 1)[1] for a in ARGS if a.startswith("--json=")), None)
    if path:
        with open(path, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
