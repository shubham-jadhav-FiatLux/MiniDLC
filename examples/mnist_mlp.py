#!/usr/bin/env python3
"""
Example: MLP training on MNIST using MiniDLC.
Compiles forward + backward (autodiff) + optimizer update into a unified training step.
"""

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
    print("Initializing MNIST MLP model...")
    model = MLP()

    # Symbolic inputs with static shapes
    x = dl.placeholder((128, 784), dtype="f32", name="x")
    y = dl.placeholder((128,), dtype="i32", name="y")

    # Define loss graph
    logits = model(x)
    loss = dl.cross_entropy(logits, y)

    # Compile training step
    print("Compiling training graph to target='reference'...")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train_step = dl.compile(loss, optimizer=optimizer, target="reference")

    print("Beginning training iterations...")
    for step, (xb, yb) in enumerate(dl.data.synthetic_mnist_batches(batch_size=128, steps=5)):
        cur_loss = train_step(x=xb, y=yb)
        print(f"Step {step+1:02d} | Loss: {cur_loss:.4f}")

    print("MNIST MLP training test completed.")

if __name__ == "__main__":
    main()
