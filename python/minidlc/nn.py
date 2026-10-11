from __future__ import annotations
from typing import List, Dict, Any, Optional
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
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.out_proj = Linear(d_model, d_model)

    def forward(self, x: Tensor) -> Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # Scaled dot-product attention
        scores = ops.matmul(q, ops.transpose(k)) * (1.0 / np.sqrt(self.d_k))
        attn = ops.softmax(scores, axis=-1)
        context = ops.matmul(attn, v)
        return self.out_proj(context)
