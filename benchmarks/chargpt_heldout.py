#!/usr/bin/env python3
"""
Char-GPT held-out evaluation on TinyShakespeare (reference interpreter).

Trains examples/char_gpt.py's model with a fixed seed on the first 90% of the text and reports the mean
cross-entropy over 40 fixed batches from the last 10%, next to the unigram and bigram baselines of the same text.
Usage: PYTHONPATH=python python3 benchmarks/chargpt_heldout.py [steps]   (default 2000; ~2 min on a laptop CPU)
"""

import os
import sys
import time

import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(repo_root, "python"))

import minidlc as dl
from minidlc import nn, ops, optim
from minidlc.data import load_tinyshakespeare
from minidlc.reference import ReferenceInterpreter

BATCH, BLOCK, D_MODEL, HEADS, LAYERS, LR = 16, 32, 32, 2, 2, 3e-3


def ngram_baselines(data: np.ndarray, vocab: int):
    """Unigram entropy and bigram conditional entropy (nats) of a token sequence."""
    p1 = np.bincount(data, minlength=vocab) / len(data)
    unigram = float(-(p1[p1 > 0] * np.log(p1[p1 > 0])).sum())
    counts = np.zeros((vocab, vocab))
    np.add.at(counts, (data[:-1], data[1:]), 1)
    rows = counts.sum(axis=1, keepdims=True)
    nz = counts > 0
    bigram = float(-(counts[nz] * np.log((counts / np.maximum(rows, 1))[nz])).sum() / counts.sum())
    return unigram, bigram


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    np.random.seed(0)
    text, tokenizer = load_tinyshakespeare()
    data = tokenizer.encode(text)
    split = int(0.9 * len(data))
    train, val = data[:split], data[split:]

    model = nn.CharGPT(vocab_size=tokenizer.vocab_size, block_size=BLOCK, d_model=D_MODEL,
                       num_heads=HEADS, num_layers=LAYERS)
    x = dl.placeholder((BATCH, BLOCK), dtype="i32", name="x")
    y = dl.placeholder((BATCH, BLOCK), dtype="i32", name="y")
    loss = ops.cross_entropy(model(x), y)
    step = dl.compile(loss, optimizer=optim.Adam(model.parameters(), lr=LR), target="reference")
    interp = ReferenceInterpreter()

    def batch(src, rng):
        ix = rng.randint(0, len(src) - BLOCK - 1, size=BATCH)
        xb = np.stack([src[i:i + BLOCK] for i in ix]).astype(np.int32)
        yb = np.stack([src[i + 1:i + 1 + BLOCK] for i in ix]).astype(np.int32)
        return xb, yb

    def val_loss(n_batches=40):
        rng = np.random.RandomState(123)
        return float(np.mean([interp.forward(loss, dict(zip((x, y), batch(val, rng))))[loss]
                              for _ in range(n_batches)]))

    unigram, bigram = ngram_baselines(data, tokenizer.vocab_size)
    print(f"{len(text):,} characters, vocab {tokenizer.vocab_size}; baselines: unigram {unigram:.4f}, "
          f"bigram {bigram:.4f} nats")
    print(f"step     0  held-out loss {val_loss():.4f}")
    rng = np.random.RandomState(1)
    t0 = time.time()
    for s in range(1, steps + 1):
        step(**dict(zip(("x", "y"), batch(train, rng))))
        if s % 250 == 0 or s == steps:
            print(f"step {s:5d}  held-out loss {val_loss():.4f}  ({time.time() - t0:.0f} s)")

    sample = model.generate(interp, tokenizer.encode("ROMEO:\n"), max_new_tokens=120, temperature=0.8)
    print("sample:", repr(tokenizer.decode(sample)))


if __name__ == "__main__":
    main()
