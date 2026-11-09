"""The models behind the committed GPU artifacts (artifacts/gpu/), shared by the exporter (examples/gpu_artifacts.py)
and the runner (examples/colab_gpu.py), so that the runner can rebuild the same graphs with the same initial weights
and check the GPU's losses against the NumPy reference interpreter.
"""
from __future__ import annotations

import numpy as np

import minidlc as dl
from minidlc import nn, optim


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        return self.fc2(dl.relu(self.fc1(x)))


MNIST = {"batch": 128, "eval_batch": 1000, "lr": 1.5e-3, "seed": 42}
CHARGPT = {"batch": 16, "block": 64, "d_model": 128, "heads": 4, "layers": 4, "lr": 3e-3, "seed": 1337}


def mnist_graphs():
    """(model, training loss, optimizer, evaluation logits) for the MNIST MLP."""
    np.random.seed(MNIST["seed"])
    model = MLP()
    x = dl.placeholder((MNIST["batch"], 784), name="x")
    y = dl.placeholder((MNIST["batch"],), dtype="i32", name="y")
    loss = dl.cross_entropy(model(x), y)
    optimizer = optim.Adam(model.parameters(), lr=MNIST["lr"])
    xe = dl.placeholder((MNIST["eval_batch"], 784), name="x")
    return model, loss, optimizer, model(xe)


def chargpt_graphs(vocab_size: int):
    """(model, training loss, optimizer, evaluation loss) for Char-GPT."""
    c = CHARGPT
    np.random.seed(c["seed"])
    model = nn.CharGPT(vocab_size=vocab_size, block_size=c["block"], d_model=c["d_model"], num_heads=c["heads"],
                       num_layers=c["layers"])
    x = dl.placeholder((c["batch"], c["block"]), dtype="i32", name="x")
    y = dl.placeholder((c["batch"], c["block"]), dtype="i32", name="y")
    loss = dl.cross_entropy(model(x), y)
    optimizer = optim.Adam(model.parameters(), lr=c["lr"])
    xe = dl.placeholder((c["batch"], c["block"]), dtype="i32", name="x")
    ye = dl.placeholder((c["batch"], c["block"]), dtype="i32", name="y")
    return model, loss, optimizer, dl.cross_entropy(model(xe), ye)


# GEMMs for the comparison with cuBLAS: the MNIST first layer with its fused epilogue, and a square 1024^3 product.
GEMMS = {"gemm_layer": {"m": 128, "k": 784, "n": 256, "epilogue": True},
         "gemm_1024": {"m": 1024, "k": 1024, "n": 1024, "epilogue": False}}


def gemm_graph(name: str):
    """relu(a @ b + bias) or a @ b, every operand a placeholder (the artifact carries no data)."""
    g = GEMMS[name]
    a = dl.placeholder((g["m"], g["k"]), name="a")
    b = dl.placeholder((g["k"], g["n"]), name="b")
    out = dl.matmul(a, b)
    if g["epilogue"]:
        out = dl.relu(out + dl.placeholder((g["n"],), name="bias"))
    return out
