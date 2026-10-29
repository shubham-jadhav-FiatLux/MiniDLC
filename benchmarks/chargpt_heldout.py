#!/usr/bin/env python3
"""
Char-GPT held-out evaluation on TinyShakespeare.

Trains Char-GPT (multi-head attention) with a fixed seed on the first 90% of the text and reports the mean
cross-entropy over 40 fixed batches from the last 10%, next to n-gram baselines: the unigram and bigram models fitted
on the training split and evaluated on the held-out split (add-one smoothing). With several targets, each trains from
the same initial weights on the same batches, so the curves can be compared directly.

Usage: PYTHONPATH=python python3 benchmarks/chargpt_heldout.py [steps] [--targets=cpu,torch,reference]
       (default 2000 steps, target cpu; "torch" needs PyTorch, "reference" is the NumPy interpreter — slow)
"""

import os
import sys
import time

import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(repo_root, "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import minidlc as dl  # noqa: E402
from minidlc import nn, ops, optim  # noqa: E402
from minidlc.data import load_tinyshakespeare  # noqa: E402
from minidlc.reference import ReferenceInterpreter  # noqa: E402

BATCH, BLOCK, D_MODEL, HEADS, LAYERS, LR = 16, 32, 64, 4, 2, 3e-3
EVAL_BATCHES = 40


def ngram_baselines(train: np.ndarray, val: np.ndarray, vocab: int):
    """Held-out cross-entropy (nats) of add-one-smoothed unigram and bigram models fitted on `train`."""
    p1 = (np.bincount(train, minlength=vocab) + 1.0) / (len(train) + vocab)
    unigram = float(-np.log(p1[val]).mean())
    counts = np.ones((vocab, vocab))
    np.add.at(counts, (train[:-1], train[1:]), 1)
    p2 = counts / counts.sum(axis=1, keepdims=True)
    bigram = float(-np.log(p2[val[:-1], val[1:]]).mean())
    return unigram, bigram


def batch(src, rng):
    ix = rng.randint(0, len(src) - BLOCK - 1, size=BATCH)
    xb = np.stack([src[i:i + BLOCK] for i in ix]).astype(np.int32)
    yb = np.stack([src[i + 1:i + 1 + BLOCK] for i in ix]).astype(np.int32)
    return xb, yb


def build(vocab):
    np.random.seed(0)
    model = nn.CharGPT(vocab_size=vocab, block_size=BLOCK, d_model=D_MODEL, num_heads=HEADS, num_layers=LAYERS)
    x = dl.placeholder((BATCH, BLOCK), dtype="i32", name="x")
    y = dl.placeholder((BATCH, BLOCK), dtype="i32", name="y")
    return model, x, y, ops.cross_entropy(model(x), y)


def trainer(target, vocab):
    """(train_step(x, y) -> loss, eval_loss(x, y) -> loss, model) for one target."""
    model, x, y, loss = build(vocab)
    if target == "torch":
        import torch
        from torch_chargpt import TorchCharGPT
        ref = TorchCharGPT(model)
        adam = torch.optim.Adam(list(ref.params.values()), lr=LR)

        def train(xb, yb):
            adam.zero_grad()
            out = ref.loss(xb, yb)
            out.backward()
            adam.step()
            return out.item()

        def evaluate(xb, yb):
            with torch.no_grad():
                return ref.loss(xb, yb).item()
        return train, evaluate, model
    step = dl.compile(loss, optimizer=optim.Adam(model.parameters(), lr=LR), target=target)
    if target == "reference":
        interp = ReferenceInterpreter()
        return (lambda xb, yb: step(x=xb, y=yb)), (lambda xb, yb: float(interp.forward(loss, {x: xb, y: yb})[loss])), model
    # Compiled: evaluation is a second compiled function that receives the step's current weights.
    xe = dl.placeholder((BATCH, BLOCK), dtype="i32", name="x")
    ye = dl.placeholder((BATCH, BLOCK), dtype="i32", name="y")
    eval_fn = dl.compile(ops.cross_entropy(model(xe), ye), target=target)
    synced = {"step": -1}
    count = {"step": 0}

    def train(xb, yb):
        count["step"] += 1
        return step(x=xb, y=yb)

    def evaluate(xb, yb):
        if synced["step"] != count["step"]:
            step.sync_params()
            for name, tensor in eval_fn._params.items():
                eval_fn.executable.set(name, np.asarray(tensor.data, dtype=np.float32))
            synced["step"] = count["step"]
        return float(eval_fn(x=xb, y=yb))
    return train, evaluate, model


def main():
    steps = next((int(a) for a in sys.argv[1:] if a.isdigit()), 2000)
    targets = next((a.split("=", 1)[1].split(",") for a in sys.argv[1:] if a.startswith("--targets=")), ["cpu"])
    text, tokenizer = load_tinyshakespeare()
    data = tokenizer.encode(text)
    split = int(0.9 * len(data))
    train, val = data[:split], data[split:]
    unigram, bigram = ngram_baselines(train, val, tokenizer.vocab_size)
    print(f"{len(text):,} characters, vocab {tokenizer.vocab_size}; Char-GPT d={D_MODEL}, {HEADS} heads, {LAYERS} layers, "
          f"context {BLOCK}, batch {BATCH}, Adam lr {LR}")
    print(f"held-out baselines: unigram {unigram:.4f}, bigram {bigram:.4f} nats")

    results = {}
    for target in targets:
        step, evaluate, model = trainer(target, tokenizer.vocab_size)

        def val_loss():
            rng = np.random.RandomState(123)
            return float(np.mean([evaluate(*batch(val, rng)) for _ in range(EVAL_BATCHES)]))

        curve = [(0, val_loss())]
        print(f"\n[{target}] step     0  held-out loss {curve[-1][1]:.4f}")
        rng = np.random.RandomState(1)
        train_seconds = 0.0
        for s in range(1, steps + 1):
            xb, yb = batch(train, rng)
            t0 = time.perf_counter()
            step(xb, yb)
            train_seconds += time.perf_counter() - t0
            if s % 250 == 0 or s == steps:
                curve.append((s, val_loss()))
                print(f"[{target}] step {s:5d}  held-out loss {curve[-1][1]:.4f}  ({train_seconds:.1f} s training)")
        results[target] = (curve, train_seconds / steps)
        if target in ("cpu", "reference"):  # the last evaluation synced the trained weights into the model
            sample = model.generate(ReferenceInterpreter(), tokenizer.encode("ROMEO:\n"), max_new_tokens=120,
                                    temperature=0.8)
            print(f"[{target}] sample: {tokenizer.decode(sample)!r}")

    print(f"\n{'target':10s} {'ms/step':>8s} {'final held-out':>15s} {'vs bigram':>10s}")
    for target, (curve, per_step) in results.items():
        print(f"{target:10s} {1e3 * per_step:8.2f} {curve[-1][1]:15.4f} {curve[-1][1] - bigram:+10.4f}")
    if len(results) > 1:
        names = list(results)
        a, b = results[names[0]][0], results[names[1]][0]
        print(f"largest held-out difference {names[0]} vs {names[1]}: {max(abs(p[1] - q[1]) for p, q in zip(a, b)):.4f}")


if __name__ == "__main__":
    main()
