#!/usr/bin/env python3
"""Builds the GPU artifacts: each graph compiled with the GPU schedules, to PTX through LLVM's NVPTX backend, saved as
plan.json + kernels.ptx + values.npz + meta.json. A machine with an NVIDIA GPU runs them with examples/colab_gpu.py
and needs only NumPy and the CUDA driver (no LLVM, no compiled module). The evaluation artifacts carry no weights:
the runner gives them the training step's.

    PYTHONPATH=python python3 examples/gpu_artifacts.py [mnist|chargpt|gemm|all] [--out=artifacts/gpu] [--arch=sm_75]
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import minidlc as dl  # noqa: E402
from minidlc.data import load_tinyshakespeare  # noqa: E402

import gpu_models  # noqa: E402


def save(name, artifact, out):
    path = os.path.join(out, name)
    artifact.save(path)
    print(f"  {path}: {len(artifact.kernel_names)} kernels, {len(artifact.ptx.splitlines())} lines of PTX, "
          f"{artifact.plan['persistent_bytes'] / 1024:.0f} KiB persistent + "
          f"{artifact.plan['transient_bytes'] / 1024:.0f} KiB transient")


def main():
    which = next((a for a in sys.argv[1:] if not a.startswith("--")), "all")
    out = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")), "artifacts/gpu")
    arch = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--arch=")), "sm_75")
    t0 = time.time()
    if which in ("mnist", "all"):
        print("MNIST MLP:")
        model, loss, optimizer, logits = gpu_models.mnist_graphs()
        params = model.parameters()
        save("mnist_train", dl.compile_gpu(loss, optimizer, arch=arch), out)
        save("mnist_eval", dl.compile_gpu(logits, arch=arch, parameters=params, include_params=False), out)
    if which in ("chargpt", "all"):
        print("Char-GPT:")
        _, tokenizer = load_tinyshakespeare()
        model, loss, optimizer, eval_loss = gpu_models.chargpt_graphs(tokenizer.vocab_size)
        params = model.parameters()
        save("chargpt_train", dl.compile_gpu(loss, optimizer, arch=arch), out)
        save("chargpt_eval", dl.compile_gpu(eval_loss, arch=arch, parameters=params, include_params=False), out)
    if which in ("gemm", "all"):
        print("GEMMs (against cuBLAS):")
        for name in gpu_models.GEMMS:
            save(name, dl.compile_gpu(gpu_models.gemm_graph(name), arch=arch), out)
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
