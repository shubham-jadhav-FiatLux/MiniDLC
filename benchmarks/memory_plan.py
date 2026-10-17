#!/usr/bin/env python3
"""Peak memory of the training step under buffer assignment (P3).

Builds the training-step graph of the MNIST MLP and of the Char-GPT example (autodiff + Adam + cleanup) with dlc-opt,
then reports the persistent region (parameters, Adam state), the transient arena with liveness-based reuse, the arena
without reuse, and the in-place updates. Usage:

    PYTHONPATH=python python3 benchmarks/memory_plan.py [path/to/dlc-opt]
"""
import os
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
import minidlc as dl  # noqa: E402
from minidlc import ir, nn  # noqa: E402
from mnist_mlp import MLP  # noqa: E402

DLC_OPT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "build", "dlc-opt")


def dlc_opt(text, *flags):
    return subprocess.run([DLC_OPT, *flags, "-"], input=text, capture_output=True, text=True, check=True).stdout


def report(name, model, x, y):
    loss = dl.cross_entropy(model(x), y)
    forward = ir.export(loss, name=name, module=model).to_text()
    step = dlc_opt(forward, "--passes=autodiff,adam{lr=0.001},canonicalize,cse,dce")
    mem = dlc_opt(step, "--print-mem-report")
    kernels = int(re.search(r"\((\d+) kernels", mem).group(1))
    persistent = int(re.search(r"persistent region: (\d+)", mem).group(1))
    transient = int(re.search(r"transient arena: +(\d+)", mem).group(1))
    unshared = int(re.search(r"vs (\d+) without reuse", mem).group(1))
    in_place = int(re.search(r"in-place updates: +(\d+)", mem).group(1))
    copies = int(re.search(r"copied at the end: (\d+)", mem).group(1))
    params = sum(p.data.size * 4 for p in model.parameters())
    print(f"| {name} | {kernels} | {params / 1024:.1f} KiB | {persistent / 1024:.1f} KiB | "
          f"{transient / 1024:.1f} KiB | {unshared / 1024:.1f} KiB | {100 * (1 - transient / unshared):.1f}% | "
          f"{in_place} / {copies} |")


def main():
    np.random.seed(0)
    print("| Training step | Kernels | Parameters | Persistent region | Transient arena | Without reuse | Saved | "
          "In place / copied |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    report("mnist_mlp", MLP(), dl.placeholder((128, 784), name="x"), dl.placeholder((128,), dtype="i32", name="y"))
    gpt = nn.CharGPT(vocab_size=65, block_size=32, d_model=32, num_heads=2, num_layers=2)
    report("char_gpt", gpt, dl.placeholder((4, 32), dtype="i32", name="x"),
           dl.placeholder((4, 32), dtype="i32", name="y"))


if __name__ == "__main__":
    main()
