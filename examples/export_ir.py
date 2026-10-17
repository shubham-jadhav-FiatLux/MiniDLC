#!/usr/bin/env python3
"""Prints the IR text of an example model's loss graph, for dlc-opt:

    PYTHONPATH=python python3 examples/export_ir.py mlp | ./build/dlc-opt --passes=shape-infer,canonicalize,cse,dce
    PYTHONPATH=python python3 examples/export_ir.py chargpt | ./build/dlc-opt --passes=fusion,mem-plan

The function returns (loss, logits). Its parameters are named after the model's attributes, e.g.
%block_0.attn.q_proj.weight.
"""

import sys

import numpy as np

import minidlc as dl
from minidlc import ir, nn
from mnist_mlp import MLP


def export(model_name: str) -> ir.IRFunction:
    np.random.seed(0)
    if model_name == "mlp":
        model = MLP()
        x = dl.placeholder((128, 784), name="x")
        y = dl.placeholder((128,), dtype="i32", name="y")
    elif model_name == "chargpt":
        model = nn.CharGPT(vocab_size=65, block_size=32, d_model=32, num_heads=2, num_layers=2)
        x = dl.placeholder((4, 32), dtype="i32", name="x")
        y = dl.placeholder((4, 32), dtype="i32", name="y")
    else:
        raise SystemExit(f"unknown model '{model_name}' (choose mlp or chargpt)")
    logits = model(x)
    return ir.export([dl.cross_entropy(logits, y), logits], name=model_name, module=model)


if __name__ == "__main__":
    print(export(sys.argv[1] if len(sys.argv) > 1 else "mlp").to_text(), end="")
