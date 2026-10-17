from __future__ import annotations
from typing import List, Dict, Any, Optional, Union
import numpy as np
from .tensor import Tensor
from . import ops

class Parameter(Tensor):
    """Trainable weight tensor."""
    def __init__(self, data: np.ndarray, name: Optional[str] = None):
        super().__init__(
            shape=data.shape,
            dtype="f32",
            op="parameter",
            data=data.astype(np.float32),
            name=name,
        )

class Module:
    """Base class for all neural network modules."""
    def __init__(self) -> None:
        self._modules: Dict[str, Module] = {}
        self._parameters: Dict[str, Parameter] = {}

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Parameter):
            if not hasattr(self, "_parameters"):
                super().__setattr__("_parameters", {})
            self._parameters[name] = value
        elif isinstance(value, Module):
            if not hasattr(self, "_modules"):
                super().__setattr__("_modules", {})
            self._modules[name] = value
        super().__setattr__(name, value)

    def parameters(self) -> List[Parameter]:
        params = list(self._parameters.values())
        for m in self._modules.values():
            params.extend(m.parameters())
        return params

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # He initialization
        bound = 1.0 / np.sqrt(in_features)
        w_data = np.random.uniform(-bound, bound, (in_features, out_features)).astype(np.float32)
        self.weight = Parameter(w_data, name="weight")
        if bias:
            b_data = np.zeros((out_features,), dtype=np.float32)
            self.bias: Optional[Parameter] = Parameter(b_data, name="bias")
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        out = ops.matmul(x, self.weight)
        if self.bias is not None:
            out = ops.add(out, self.bias)
        return out

class LayerNorm(Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = Parameter(np.ones((normalized_shape,), dtype=np.float32), name="gamma")
        self.beta = Parameter(np.zeros((normalized_shape,), dtype=np.float32), name="beta")

    def forward(self, x: Tensor) -> Tensor:
        return ops.layernorm(x, self.gamma, self.beta, eps=self.eps)

class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        weight_data = np.random.normal(0.0, 0.02, (num_embeddings, embedding_dim)).astype(np.float32)
        self.weight = Parameter(weight_data, name="emb_weight")

    def forward(self, indices: Tensor) -> Tensor:
        return ops.embedding(self.weight, indices)

class Sequential(Module):
    def __init__(self, *layers: Module) -> None:
        super().__init__()
        for i, layer in enumerate(layers):
            setattr(self, f"layer_{i}", layer)
        self.layers = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

class MultiHeadAttention(Module):
    """Causal scaled dot-product attention with q/k/v/output projections.

    Heads are not split yet: this computes single-head attention over all d_model channels, scaled by
    1/sqrt(d_model / num_heads). Real head splitting (reshape + transpose + 4-D matmul) arrives in v2, phase P6.
    """
    def __init__(self, d_model: int, num_heads: int, causal: bool = True) -> None:
        super().__init__()
        assert d_model % num_heads == 0, f"d_model {d_model} must be divisible by num_heads {num_heads}"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.causal = causal

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.out_proj = Linear(d_model, d_model)

    def forward(self, x: Tensor) -> Tensor:
        # x is [B, T, D]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Transpose k for dot product: [B, T, D] -> [B, D, T]
        kT = ops.transpose(k, perm=[0, 2, 1])
        scores = ops.matmul(q, kT) * (1.0 / np.sqrt(self.d_k))  # [B, T, T]

        if self.causal:
            seq_len = x.shape[1]
            mask = ops.triu_mask(seq_len)  # [T, T]
            scores = ops.add(scores, mask)

        attn = ops.softmax(scores, axis=-1)  # [B, T, T]
        context = ops.matmul(attn, v)        # [B, T, D]
        return self.out_proj(context)

class TransformerBlock(Module):
    def __init__(self, d_model: int, num_heads: int, causal: bool = True) -> None:
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, causal=causal)
        self.ln2 = LayerNorm(d_model)
        self.mlp_fc1 = Linear(d_model, 4 * d_model)
        self.mlp_fc2 = Linear(4 * d_model, d_model)

    def forward(self, x: Tensor) -> Tensor:
        # Pre-LN residual self-attention
        x = x + self.attn(self.ln1(x))
        # Pre-LN residual feed-forward MLP with GELU
        m = ops.gelu(self.mlp_fc1(self.ln2(x)))
        x = x + self.mlp_fc2(m)
        return x

class CharGPT(Module):
    """Autoregressive character-level Transformer for TinyShakespeare."""
    def __init__(
        self,
        vocab_size: int = 65,
        block_size: int = 64,
        d_model: int = 64,
        num_heads: int = 2,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.d_model = d_model

        self.token_emb = Embedding(vocab_size, d_model)
        self.pos_emb = Embedding(block_size, d_model)

        self.blocks: List[TransformerBlock] = []
        for i in range(num_layers):
            block = TransformerBlock(d_model, num_heads, causal=True)
            setattr(self, f"block_{i}", block)
            self.blocks.append(block)

        self.ln_f = LayerNorm(d_model)
        self.head = Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: Tensor, pos: Optional[Tensor] = None) -> Tensor:
        # idx: [B, T]
        B_sz, T_sz = idx.shape
        tok_embed = self.token_emb(idx)  # [B, T, D]

        if pos is None:
            pos_indices = np.arange(T_sz, dtype=np.int32).reshape(1, T_sz)
            pos_tensor = Tensor(shape=(1, T_sz), dtype="i32", op="constant", data=pos_indices)
            pos_embed = self.pos_emb(pos_tensor)  # [1, T, D]
        else:
            pos_embed = self.pos_emb(pos)

        x = ops.add(tok_embed, pos_embed)  # [B, T, D]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)  # [B, T, V]
        return logits

    def generate(
        self,
        interp: Any,
        prompt_tokens: Union[np.ndarray, List[int]],
        max_new_tokens: int = 50,
        temperature: float = 0.8,
    ) -> List[int]:
        """Autoregressively generates continuation tokens given a prompt."""
        from .tensor import placeholder
        tokens = list(prompt_tokens)
        for _ in range(max_new_tokens):
            curr = np.array(tokens[-self.block_size:], dtype=np.int32).reshape(1, -1)
            T_curr = curr.shape[1]
            pos = np.arange(T_curr, dtype=np.int32).reshape(1, T_curr)

            idx_node = placeholder(shape=curr.shape, dtype="i32", name="gen_x")
            pos_node = placeholder(shape=pos.shape, dtype="i32", name="gen_pos")
            logits_node = self.forward(idx_node, pos_node)

            vals = interp.forward(logits_node, {idx_node: curr, pos_node: pos})
            logits = vals[logits_node][0, -1, :]
            if temperature > 0:
                logits = logits / temperature
                shift = logits - np.max(logits)
                probs = np.exp(shift) / np.sum(np.exp(shift))
                probs = probs / np.sum(probs)
                next_tok = int(np.random.choice(len(probs), p=probs))
            else:
                next_tok = int(np.argmax(logits))
            tokens.append(next_tok)
        return tokens
