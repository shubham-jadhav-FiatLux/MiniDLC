#!/usr/bin/env python3
"""
Example: Autoregressive Character-Level Transformer trained on TinyShakespeare using MiniDLC.
Demonstrates:
  1. Learned token and position embeddings.
  2. Causal self-attention with a triangular mask (single-head for now; head splitting arrives in v2, phase P6).
  3. Pre-LN Transformer blocks with residual connections and GELU feed-forward MLPs.
  4. Automatic differentiation with cross-entropy loss and Adam optimization.
  5. Training through dl.compile(loss, optimizer, target='reference'), i.e. the NumPy reference interpreter.
  6. Autoregressive character generation given prompts.
"""

import sys
import time
import numpy as np
import minidlc as dl
from minidlc import ops, nn, optim
from minidlc.data import load_tinyshakespeare, tinyshakespeare_batches
from minidlc.reference import ReferenceInterpreter

def main():
    print("=" * 60)
    print(" MiniDLC: Autoregressive Char-GPT on TinyShakespeare")
    print("=" * 60)

    # 1. Dataset Loading
    print("\n[1/4] Loading TinyShakespeare dataset...")
    text, tokenizer = load_tinyshakespeare()
    vocab_size = tokenizer.vocab_size
    data = tokenizer.encode(text)
    print(f"  Total characters : {len(text):,}")
    print(f"  Vocabulary size  : {vocab_size} unique characters")
    print(f"  Vocabulary sample: {repr(''.join(tokenizer.chars[:20]))} ...")

    # 2. Model Architecture Hyperparameters (configurable via CLI)
    batch_size = 4
    block_size = 32
    d_model = 32
    num_heads = 2
    num_layers = 2
    lr = 0.008
    num_steps = 40

    for arg in sys.argv[1:]:
        if arg.startswith("--block-size="):
            block_size = int(arg.split("=")[1])
        elif arg.startswith("--d-model="):
            d_model = int(arg.split("=")[1])
        elif arg.startswith("--num-heads="):
            num_heads = int(arg.split("=")[1])
        elif arg.startswith("--num-layers="):
            num_layers = int(arg.split("=")[1])
        elif arg.startswith("--steps="):
            num_steps = int(arg.split("=")[1])
        elif arg.startswith("--lr="):
            lr = float(arg.split("=")[1])

    np.random.seed(1337)  # before model init, so weights and batches are reproducible
    print(f"\n[2/4] Initializing Transformer model...")
    print(f"  Block size (context) : {block_size}")
    print(f"  Model dimension (d)  : {d_model}")
    print(f"  Attention heads      : {num_heads}")
    print(f"  Transformer layers   : {num_layers}")

    model = nn.CharGPT(
        vocab_size=vocab_size,
        block_size=block_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
    )
    total_params = sum(p.data.size for p in model.parameters() if p.data is not None)
    print(f"  Total parameters     : {total_params:,}")

    # 3. Setup Computation Graph & Compile Training Step
    x_node = dl.placeholder((batch_size, block_size), dtype="i32", name="x")
    y_node = dl.placeholder((batch_size, block_size), dtype="i32", name="y")

    logits_node = model(x_node)
    loss_node = ops.cross_entropy(logits_node, y_node)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"\n[3/4] Compiling end-to-end training step (dl.compile target='reference')...")
    train_step = dl.compile(loss_node, optimizer=optimizer, target="reference")

    print(f"  Training for {num_steps} iterations (Adam lr={lr})...")
    batches = tinyshakespeare_batches(data, batch_size=batch_size, block_size=block_size, steps=num_steps)

    t0 = time.time()
    losses = []
    for step in range(num_steps):
        bx, by = next(batches)
        step_loss = train_step(x=bx, y=by)
        losses.append(step_loss)

        if (step + 1) % 5 == 0 or step == 0:
            elapsed = time.time() - t0
            print(f"  Step {step + 1:2d}/{num_steps:2d} | Loss: {step_loss:.4f} | Elapsed: {elapsed:.2f}s")

    print(f"\nTraining complete. Loss decreased from {losses[0]:.4f} to {losses[-1]:.4f}")

    # 4. Text Generation Demonstration
    print(f"\n[4/4] Autoregressive Text Generation Demonstration:")
    interp = ReferenceInterpreter()
    prompts = ["ROMEO:\n", "First Citizen:\n"]
    for prompt in prompts:
        prompt_tokens = tokenizer.encode(prompt)
        gen_tokens = model.generate(interp, prompt_tokens, max_new_tokens=40, temperature=0.7)
        output_str = tokenizer.decode(gen_tokens)
        print("-" * 50)
        print(f"Prompt: {repr(prompt)}")
        print(f"Generated text:\n{output_str}")
    print("-" * 50)

if __name__ == "__main__":
    main()
