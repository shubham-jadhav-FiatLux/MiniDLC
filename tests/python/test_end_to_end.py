"""End-to-end tests: the project's training results as tests. They download MNIST and TinyShakespeare and train for
a few minutes, so they run nightly in CI (.github/workflows/nightly.yml) or with MINIDLC_E2E=1:

    MINIDLC_E2E=1 PYTHONPATH=python pytest tests/python/test_end_to_end.py

  - MNIST MLP (784-256-10, Adam, 2 epochs) compiled for the CPU: test accuracy on all 10,000 test images >= 96.5%.
  - Char-GPT (d_model 64, 4 heads, 2 layers, context 32, batch 16, Adam) compiled for the CPU, 2,000 steps (or
    MINIDLC_E2E_STEPS): the held-out loss is below the held-out bigram baseline by at least 0.3 nats and, when PyTorch
    is installed, within 0.05 nats of PyTorch trained from the same weights on the same batches.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

import minidlc as dl
from minidlc import nn, optim

pytestmark = pytest.mark.skipif(os.environ.get("MINIDLC_E2E") != "1",
                                reason="end-to-end tests run nightly (set MINIDLC_E2E=1): they download data and train")
pytest.importorskip("minidlc._minidlc", reason="the _minidlc module is not built")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "benchmarks"))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        return self.fc2(dl.relu(self.fc1(x)))


def test_mnist_mlp_reaches_96_5_percent():
    from minidlc.data import load_mnist, mnist_batches
    tr_x, tr_y, ts_x, ts_y = load_mnist()
    np.random.seed(42)
    model = MLP()
    x = dl.placeholder((128, 784), name="x")
    y = dl.placeholder((128,), dtype="i32", name="y")
    step = dl.compile(dl.cross_entropy(model(x), y), optim.Adam(model.parameters(), lr=1.5e-3), target="cpu")
    for _ in range(2):
        for xb, yb in mnist_batches(tr_x, tr_y, batch_size=128, shuffle=True):
            loss = step(x=xb, y=yb)
    assert np.isfinite(loss)
    step.sync_params()
    xe = dl.placeholder((1000, 784), name="x")
    evaluate = dl.compile(model(xe), target="cpu")
    correct = sum(int(np.sum(np.argmax(evaluate(x=ts_x[i:i + 1000]), axis=-1) == ts_y[i:i + 1000]))
                  for i in range(0, len(ts_x), 1000))
    accuracy = 100.0 * correct / len(ts_x)
    print(f"MNIST test accuracy {accuracy:.2f}% on {len(ts_x)} images")
    assert len(ts_x) == 10000
    assert accuracy >= 96.5


def test_chargpt_held_out_loss_beats_bigram_and_tracks_pytorch():
    import chargpt_heldout as h
    from minidlc.data import load_tinyshakespeare
    steps = int(os.environ.get("MINIDLC_E2E_STEPS", "2000"))
    text, tokenizer = load_tinyshakespeare()
    data = tokenizer.encode(text)
    split = int(0.9 * len(data))
    train, val = data[:split], data[split:]
    _, bigram = h.ngram_baselines(train, val, tokenizer.vocab_size)

    def held_out_after_training(target):
        step, evaluate, _ = h.trainer(target, tokenizer.vocab_size)
        rng = np.random.RandomState(1)
        for _ in range(steps):
            step(*h.batch(train, rng))
        vrng = np.random.RandomState(123)
        return float(np.mean([evaluate(*h.batch(val, vrng)) for _ in range(h.EVAL_BATCHES)]))

    ours = held_out_after_training("cpu")
    print(f"Char-GPT held-out loss after {steps} steps: {ours:.4f} (bigram {bigram:.4f})")
    assert ours < bigram - 0.3
    try:
        import torch  # noqa: F401
    except ImportError:
        return
    theirs = held_out_after_training("torch")
    print(f"PyTorch from the same weights and batches: {theirs:.4f}")
    assert abs(ours - theirs) < 0.05
