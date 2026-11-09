"""Whole-pipeline fuzzing: random models through the C++ compiler on every target, against the reference.

Each seed builds a random model in the Python frontend: a pool of [batch, width] tensors grown by random layers —
linear layers with and without bias (widths around the tile sizes: 3, 8, 17, 64, 65), ReLU, GELU, LayerNorm,
softmax and log-softmax, exp and log, division, broadcast binary ops between pool members, reductions with keepdims,
attention-like blocks (matmul with a transposed operand, softmax, matmul), reshapes into three dimensions and back;
optionally an embedding branch over a token sequence, pooled over time. A linear head and cross-entropy make the loss.

For each seed:
  - training: the compiled step (C++ autodiff, Adam or SGD lowering, canonicalization, CSE, fusion, buffer
    assignment, loop passes, code) on `cpu` and `gpu-sim` (and `interp` for some seeds) follows eager training in the
    NumPy reference interpreter (Python autodiff): losses over three steps and the parameters after them;
  - forward: the compiled function's outputs match the reference IR executor.
Needs the compiled _minidlc module; skipped otherwise.
"""
from __future__ import annotations

import numpy as np
import pytest

import minidlc as dl
from minidlc import ir, nn, optim

pytest.importorskip("minidlc._minidlc", reason="the _minidlc module is not built")

SEEDS = range(7100, 7124)


class RandomModel:
    def __init__(self, seed: int):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.B = int(rng.choice([3, 5, 8, 13]))
        self.D = int(rng.choice([4, 7, 16, 33, 70]))
        self.C = int(rng.choice([2, 5, 11]))
        self.params = []
        self.ops = []
        x = dl.placeholder((self.B, self.D), name="x")
        self.y = dl.placeholder((self.B,), dtype="i32", name="y")
        self.inputs = {"x": ((self.B, self.D), "f32")}
        pool = [x]
        if rng.random() < 0.4:  # a token sequence, embedded and pooled over time
            T, V, E = int(rng.choice([3, 6])), int(rng.choice([7, 13])), int(rng.choice([5, 8]))
            ids = dl.placeholder((self.B, T), dtype="i32", name="ids")
            self.inputs["ids"] = ((self.B, T), V)
            seq = dl.embedding(self.param((V, E), 0.5), ids)
            if rng.random() < 0.5:
                seq = dl.layernorm(seq, self.param((E,), 0.1, 1.0), self.param((E,), 0.1))
            pool.append(dl.mean(seq, axis=1))
            self.ops.append("embedding")
        for _ in range(int(rng.integers(3, 8))):
            pool.append(self.layer(pool))
        final = pool[-1]
        logits = dl.matmul(final, self.param((final.shape[1], self.C), 1.0 / np.sqrt(final.shape[1])))
        self.logits = dl.add(logits, self.param((self.C,), 0.1))
        self.loss = dl.cross_entropy(self.logits, self.y)

    def param(self, shape, scale, offset=0.0):
        p = nn.Parameter((self.rng.standard_normal(shape) * scale + offset).astype(np.float32), name=f"p{len(self.params)}")
        self.params.append(p)
        return p

    def layer(self, pool):
        rng = self.rng
        t = pool[int(rng.integers(len(pool)))]
        d = t.shape[1]
        op = str(rng.choice(["linear", "linear", "relu", "gelu", "layernorm", "softmax", "log_softmax", "exp", "log",
                             "div", "binary", "reduce", "attention", "reshape3d"]))
        self.ops.append(op)
        if op == "linear":
            w = int(rng.choice([3, 8, 17, 64, 65]))
            out = dl.matmul(t, self.param((d, w), 1.0 / np.sqrt(d)))
            return dl.add(out, self.param((w,), 0.1)) if rng.random() < 0.6 else out
        if op == "relu":
            return dl.relu(t)
        if op == "gelu":
            return dl.gelu(t)
        if op == "layernorm":
            return dl.layernorm(t, self.param((d,), 0.1, 1.0), self.param((d,), 0.1))
        if op == "softmax":
            return dl.mul(dl.softmax(t, axis=-1), float(d))
        if op == "log_softmax":
            return dl.log_softmax(t, axis=-1)
        if op == "exp":
            return dl.exp(dl.mul(t, 0.3))
        if op == "log":
            return dl.log(dl.add(dl.mul(t, t), 1.0))
        if op == "div":
            return dl.div(t, dl.add(dl.mul(t, t), 1.0))
        if op == "binary":
            same = [u for u in pool if u.shape == t.shape]
            u = same[int(rng.integers(len(same)))] if len(same) > 1 and rng.random() < 0.6 else self.param((d,), 0.5)
            kind = str(rng.choice(["add", "sub", "mul"]))
            return dl.add(t, u) if kind == "add" else dl.sub(t, u) if kind == "sub" else dl.mul(t, dl.mul(u, 0.5))
        if op == "reduce":
            s = dl.sum(t, axis=1, keepdims=True) if rng.random() < 0.5 else dl.mean(t, axis=1, keepdims=True)
            return dl.add(t, dl.mul(s, 0.1))
        if op == "attention":  # softmax(t t^T / sqrt(d)) t
            scores = dl.mul(dl.matmul(t, dl.transpose(t, [1, 0])), 1.0 / np.sqrt(d))
            return dl.matmul(dl.softmax(scores, axis=-1), t)
        # reshape3d: [B, d] -> [B, d/k, k], softmax over the middle axis, back to [B, d]
        k = next((k for k in (4, 3, 2) if d % k == 0 and d // k > 1), None)
        if k is None:
            return dl.relu(t)
        three = dl.softmax(dl.reshape(t, (self.B, d // k, k)), axis=1)
        return dl.mul(dl.reshape(three, (self.B, d)), float(d // k))

    def feeds(self, rng):
        out = {"x": rng.standard_normal((self.B, self.D)).astype(np.float32),
               "y": rng.integers(0, self.C, self.B).astype(np.int32)}
        if "ids" in self.inputs:
            shape, vocab = self.inputs["ids"]
            out["ids"] = rng.integers(0, vocab, shape).astype(np.int32)
        return out


def optimizer_for(seed, params):
    return optim.Adam(params, lr=1e-2) if seed % 2 == 0 else optim.SGD(params, lr=0.05, momentum=0.9)


@pytest.mark.parametrize("seed", SEEDS)
def test_random_model_trains_like_the_reference(seed):
    data_rng = np.random.default_rng(seed + 1)
    probe = RandomModel(seed)
    batches = [probe.feeds(data_rng) for _ in range(3)]

    ref = RandomModel(seed)
    step = dl.compile(ref.loss, optimizer_for(seed, ref.params), target="reference")
    want = [step(**b) for b in batches]
    want_params = [p.data.copy() for p in ref.params]

    targets = ["cpu", "gpu-sim"] + (["interp"] if seed % 4 == 0 else [])
    for target in targets:
        model = RandomModel(seed)
        compiled = dl.compile(model.loss, optimizer_for(seed, model.params), target=target)
        got = [compiled(**b) for b in batches]
        context = f"seed {seed}, target {target}, ops {model.ops}"
        np.testing.assert_allclose(got, want, rtol=3e-4, atol=1e-5, err_msg=context)
        compiled.sync_params()
        for p, w in zip(model.params, want_params):
            np.testing.assert_allclose(p.data, w, rtol=2e-3, atol=2e-4, err_msg=f"{context}, parameter {p.name}")


@pytest.mark.parametrize("seed", SEEDS)
def test_random_model_forward_matches_the_reference(seed):
    model = RandomModel(seed)
    feeds = model.feeds(np.random.default_rng(seed + 2))
    func = ir.export([model.logits, model.loss], name=f"fuzz_{seed}")
    want = ir.execute(func, {k: v for k, v in feeds.items() if k != "y"} | {"y": feeds["y"]})
    for target in ["cpu", "gpu-sim"]:
        f = dl.compile([model.logits, model.loss], target=target)
        got = f(**feeds)
        for g, w in zip(got, want):
            np.testing.assert_allclose(g, w, rtol=1e-4, atol=1e-5, err_msg=f"seed {seed}, target {target}, ops {model.ops}")
