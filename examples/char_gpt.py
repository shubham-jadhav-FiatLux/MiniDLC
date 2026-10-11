#!/usr/bin/env python3
"""
Example: Character-Level GPT trained on TinyShakespeare using MiniDLC.
Implements token and position embeddings, multi-head causal attention blocks, and linear projection.
"""

import numpy as np
import minidlc as dl
from minidlc import nn, optim

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiHeadAttention(d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp_fc1 = nn.Linear(d_model, 4 * d_model)
        self.mlp_fc2 = nn.Linear(4 * d_model, d_model)

    def forward(self, x: dl.Tensor) -> dl.Tensor:
        # Pre-LN residual attention
        x_norm = self.ln1(x)
        x = x + self.attn(x_norm)
        # Pre-LN residual MLP
        m = dl.gelu(self.mlp_fc1(self.ln2(x)))
        x = x + self.mlp_fc2(m)
        return x

class CharGPT(nn.Module):
    def __init__(self, vocab_size: int = 65, block_size: int = 64, d_model: int = 64, num_heads: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.block = TransformerBlock(d_model, num_heads)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: dl.Tensor, pos: dl.Tensor) -> dl.Tensor:
        tok_embeddings = self.token_emb(idx)
        pos_embeddings = self.pos_emb(pos)
        x = tok_embeddings + pos_embeddings
        x = self.block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

def main():
    print("Char-GPT architecture instantiated.")
    model = CharGPT(vocab_size=65, block_size=32, d_model=32, num_heads=2)
    print(f"Total model parameters: {len(model.parameters())}")

if __name__ == "__main__":
    main()
