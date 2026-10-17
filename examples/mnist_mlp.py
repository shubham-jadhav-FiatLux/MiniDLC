#!/usr/bin/env python3
"""
Example: MLP training on MNIST using MiniDLC.
Trains with target "cpu" (hand-written LLVM IR kernels for this fixed MLP, the default) or "reference" (the NumPy
interpreter with reverse-mode autodiff), then evaluates test accuracy on the MNIST test set.
"""

import sys
import time
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

def main():
    epochs = 2
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        epochs = int(sys.argv[1])

    print("Loading MNIST dataset...")
    tr_x, tr_y, ts_x, ts_y = dl.data.load_mnist()
    print(f"Loaded: {tr_x.shape[0]} train samples, {ts_x.shape[0]} test samples.")

    print("Initializing MNIST MLP model (784 -> 256 -> 10)...")
    np.random.seed(42)
    model = MLP()

    # Symbolic inputs with static shapes
    batch_size = 128
    x = dl.placeholder((batch_size, 784), dtype="f32", name="x")
    y = dl.placeholder((batch_size,), dtype="i32", name="y")

    # Define loss graph
    logits = model(x)
    loss = dl.cross_entropy(logits, y)

    target = "cpu"
    profile = "--profile" in sys.argv
    for arg in sys.argv[1:]:
        if arg.startswith("--target="):
            target = arg.split("=")[1]
        elif arg in ("reference", "cpu", "cuda"):
            target = arg

    # Compile training step
    print(f"Compiling training graph to target='{target}' (profile={profile})...")
    optimizer = optim.Adam(model.parameters(), lr=1.5e-3)
    train_step = dl.compile(loss, optimizer=optimizer, target=target, profile=profile)


    print(f"Beginning training iterations ({epochs} epoch(s), batch_size={batch_size})...")
    n_batches = len(tr_x) // batch_size
    t_start = time.time()

    for ep in range(epochs):
        for b, (xb, yb) in enumerate(dl.data.mnist_batches(tr_x, tr_y, batch_size=batch_size, shuffle=True)):
            cur_loss = train_step(x=xb, y=yb)
            if (b + 1) % 100 == 0 or b == n_batches - 1:
                elapsed = time.time() - t_start
                print(f"Epoch {ep+1}/{epochs} | Batch {b+1:03d}/{n_batches} | Loss: {cur_loss:.4f} | Time: {elapsed:.1f}s")

    print("\nEvaluating test set accuracy...")
    interp = dl.ReferenceInterpreter()
    test_logits = model(x)
    correct = 0
    total = 0

    for xb, yb in dl.data.mnist_batches(ts_x, ts_y, batch_size=batch_size, shuffle=False):
        vals = interp.forward(test_logits, {x: xb})
        preds = np.argmax(vals[test_logits], axis=-1)
        correct += int(np.sum(preds == yb))
        total += len(yb)

    test_acc = 100.0 * correct / total
    print(f"Final Test Accuracy: {test_acc:.2f}% ({correct}/{total})")
    print("MNIST MLP training completed successfully.")

    if profile and hasattr(train_step, "print_profile"):
        train_step.print_profile()

if __name__ == "__main__":
    main()
