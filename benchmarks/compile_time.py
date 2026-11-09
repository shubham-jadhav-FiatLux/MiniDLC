#!/usr/bin/env python3
"""Compile times, and what the kernel cache saves.

Compiles the MNIST and Char-GPT training steps (CPU) and the Char-GPT step for the GPU (PTX) twice each, with an empty
kernel cache in a temporary directory: the first compile runs LLVM (O3 and code generation), the second loads the
object file or PTX from the cache. The graph passes, fusion, buffer assignment and the loop passes run both times.

Usage: PYTHONPATH=python python3 benchmarks/compile_time.py [--repeats=3]
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time

import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(repo_root, "python"))
CACHE = tempfile.mkdtemp(prefix="minidlc-compile-time-")
os.environ["MINIDLC_CACHE_DIR"] = CACHE  # before the compiled module reads it

import minidlc as dl  # noqa: E402
from minidlc import nn, optim  # noqa: E402

REPEATS = int(next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--repeats=")), 3))


def mnist(target):
    np.random.seed(0)
    fc1, fc2 = nn.Linear(784, 256), nn.Linear(256, 10)
    x = dl.placeholder((128, 784), name="x")
    y = dl.placeholder((128,), dtype="i32", name="y")
    loss = dl.cross_entropy(fc2(dl.relu(fc1(x))), y)
    opt = optim.Adam(fc1.parameters() + fc2.parameters(), lr=1e-3)
    return dl.compile(loss, opt, target=target) if target != "ptx" else dl.compile_gpu(loss, opt)


def chargpt(target):
    np.random.seed(0)
    model = nn.CharGPT(vocab_size=65, block_size=32, d_model=64, num_heads=4, num_layers=2)
    x = dl.placeholder((16, 32), dtype="i32", name="x")
    y = dl.placeholder((16, 32), dtype="i32", name="y")
    loss = dl.cross_entropy(model(x), y)
    opt = optim.Adam(model.parameters(), lr=3e-3)
    return dl.compile(loss, opt, target=target) if target != "ptx" else dl.compile_gpu(loss, opt)


def timed(build, target):
    t0 = time.perf_counter()
    build(target)
    return time.perf_counter() - t0


def main():
    print(f"load average {os.getloadavg()[0]:.2f}; {REPEATS} repeats, median seconds")
    print(f"  {'step':28s} {'cold (LLVM)':>12s} {'from cache':>11s}")
    try:
        for name, build, target in [("MNIST MLP, cpu", mnist, "cpu"), ("Char-GPT, cpu", chargpt, "cpu"),
                                    ("MNIST MLP, PTX", mnist, "ptx"), ("Char-GPT, PTX", chargpt, "ptx")]:
            cold, warm = [], []
            for _ in range(REPEATS):
                dl.clear_cache()
                cold.append(timed(build, target))
                warm.append(timed(build, target))
            print(f"  {name:28s} {np.median(cold):12.3f} {np.median(warm):11.3f}")
        info = dl.cache_info()
        print(f"cache: {info['entries']} entries, {info['bytes'] / 1024:.0f} KiB; {info['hits']} hits, {info['misses']} misses")
    finally:
        shutil.rmtree(CACHE, ignore_errors=True)


if __name__ == "__main__":
    main()
