"""Char-GPT in PyTorch, with MiniDLC's weights: the external reference for P6.

`TorchCharGPT(model)` copies the parameters of a `minidlc.nn.CharGPT` into PyTorch tensors (by their MiniDLC names)
and computes the same function with PyTorch ops — tanh-approximated GELU, LayerNorm with the biased variance, the
additive causal mask, heads split by reshape and transpose — so that losses, gradients and Adam trajectories can be
compared op for op. Used by tests/python/test_pytorch_crosscheck.py and benchmarks/chargpt_heldout.py.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F


class TorchCharGPT:
    def __init__(self, model, dtype=torch.float32, threads: int = 0):
        if threads:
            torch.set_num_threads(threads)
        self.dtype = dtype
        self.layers = len(model.blocks)
        self.heads = model.blocks[0].attn.num_heads
        self.params: Dict[str, torch.Tensor] = {
            name: torch.tensor(np.asarray(p.data), dtype=dtype, requires_grad=True) for name, p in model.named_parameters()}
        self.mask = torch.tensor(np.asarray(_causal_mask(model.block_size)), dtype=dtype)

    def logits(self, idx: np.ndarray) -> torch.Tensor:
        P = self.params
        idx = torch.as_tensor(np.asarray(idx, dtype=np.int64))
        b, t = idx.shape
        x = P["token_emb.weight"][idx] + P["pos_emb.weight"][:t]
        for i in range(self.layers):
            pre = f"block_{i}."
            x = x + self._attention(self._ln(x, pre + "ln1"), pre + "attn.")
            h = F.gelu(self._linear(self._ln(x, pre + "ln2"), pre + "mlp_fc1"), approximate="tanh")
            x = x + self._linear(h, pre + "mlp_fc2")
        return self._ln(x, "ln_f") @ P["head.weight"]

    def loss(self, x: np.ndarray, y: np.ndarray) -> torch.Tensor:
        logits = self.logits(x)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), torch.as_tensor(np.asarray(y, dtype=np.int64)).reshape(-1))

    def _ln(self, x, name):
        return F.layer_norm(x, x.shape[-1:], self.params[name + ".gamma"], self.params[name + ".beta"], eps=1e-5)

    def _linear(self, x, name):
        out = x @ self.params[name + ".weight"]
        bias = self.params.get(name + ".bias")
        return out if bias is None else out + bias

    def _attention(self, x, pre):
        b, t, d = x.shape
        dh = d // self.heads
        split = lambda v: v.reshape(b, t, self.heads, dh).transpose(1, 2)
        q, k, v = (split(self._linear(x, pre + n)) for n in ("q_proj", "k_proj", "v_proj"))
        scores = q @ k.transpose(-1, -2) * (1.0 / np.sqrt(dh)) + self.mask[:t, :t]
        context = torch.softmax(scores, dim=-1) @ v
        return self._linear(context.transpose(1, 2).reshape(b, t, d), pre + "out_proj")


def _causal_mask(n: int) -> np.ndarray:
    # The frontend's triu_mask: 0 on and below the diagonal, -1e4 above.
    return np.triu(np.full((n, n), -1e4, dtype=np.float32), 1)
