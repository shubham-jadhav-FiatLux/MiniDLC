"""Char-GPT against PyTorch: the same weights and batches give the same loss and gradients (reference interpreter, in
float64) and the same Adam trajectory (the compiled step on the CPU and on the simulated GPU, in float32).
Skipped without torch (`pip install torch --index-url https://download.pytorch.org/whl/cpu`).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "benchmarks"))

import minidlc as dl  # noqa: E402
from minidlc import nn, optim  # noqa: E402
from minidlc.reference import ReferenceInterpreter  # noqa: E402
from torch_chargpt import TorchCharGPT  # noqa: E402

B, T, V, D, H, L = 3, 8, 13, 16, 4, 2


def build(seed=0):
    np.random.seed(seed)
    model = nn.CharGPT(vocab_size=V, block_size=T, d_model=D, num_heads=H, num_layers=L)
    x = dl.placeholder((B, T), dtype="i32", name="x")
    y = dl.placeholder((B, T), dtype="i32", name="y")
    return model, x, y, dl.cross_entropy(model(x), y)


def batches(n, seed=1):
    rng = np.random.default_rng(seed)
    return [(rng.integers(0, V, (B, T)).astype(np.int32), rng.integers(0, V, (B, T)).astype(np.int32)) for _ in range(n)]


def test_loss_and_gradients_match_pytorch():
    model, x, y, loss = build()
    xb, yb = batches(1)[0]
    interp = ReferenceInterpreter(dtype=np.float64)
    values = interp.forward(loss, {x: xb, y: yb})
    grads = interp.backward(loss, values)
    ref = TorchCharGPT(model, dtype=torch.float64)
    got = ref.loss(xb, yb)
    got.backward()
    assert abs(got.item() - float(values[loss])) < 1e-9
    for name, p in model.named_parameters():
        np.testing.assert_allclose(grads[p], ref.params[name].grad.numpy(), rtol=1e-7, atol=1e-10, err_msg=name)


@pytest.mark.parametrize("target", ["cpu", "gpu-sim"])
def test_compiled_adam_trajectory_tracks_pytorch(target):
    pytest.importorskip("minidlc._minidlc", reason="the _minidlc module is not built")
    model, x, y, loss = build()
    ref = TorchCharGPT(model)
    adam = torch.optim.Adam(list(ref.params.values()), lr=0.01)
    step = dl.compile(loss, optimizer=optim.Adam(model.parameters(), lr=0.01), target=target)
    data = batches(8 if target == "cpu" else 4)
    want = []
    for xb, yb in data:
        adam.zero_grad()
        out = ref.loss(xb, yb)
        out.backward()
        adam.step()
        want.append(out.item())
    got = [step(x=xb, y=yb) for xb, yb in data]
    np.testing.assert_allclose(got, want, rtol=2e-4, atol=2e-4)
    step.sync_params()
    for name, p in model.named_parameters():
        if name.endswith("k_proj.bias"):
            # Softmax ignores a bias on the keys (it shifts each row of scores by q.b), so its gradient is zero up to
            # rounding, and Adam turns that noise into lr-sized steps that differ between implementations.
            continue
        np.testing.assert_allclose(p.data, ref.params[name].detach().numpy(), rtol=1e-3, atol=2e-4, err_msg=name)
