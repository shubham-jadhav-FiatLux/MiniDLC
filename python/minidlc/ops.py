from __future__ import annotations
from typing import Tuple, List, Optional, Union
import numpy as np
from .tensor import Tensor

def _broadcast_shape(s1: Tuple[int, ...], s2: Tuple[int, ...]) -> Tuple[int, ...]:
    out = []
    r1, r2 = len(s1), len(s2)
    max_r = max(r1, r2)
    for i in range(max_r):
        d1 = s1[r1 - 1 - i] if i < r1 else 1
        d2 = s2[r2 - 1 - i] if i < r2 else 1
        if d1 == d2:
            out.append(d1)
        elif d1 == 1:
            out.append(d2)
        elif d2 == 1:
            out.append(d1)
        else:
            raise ValueError(f"Incompatible shapes for broadcasting: {s1} and {s2}")
    return tuple(reversed(out))

def matmul(a: Tensor, b: Tensor) -> Tensor:
    if len(a.shape) == 2 and len(b.shape) == 2:
        if a.shape[1] != b.shape[0]:
            raise ValueError(f"Shape mismatch in matmul: {a.shape} and {b.shape}")
        out_shape = (a.shape[0], b.shape[1])
    elif len(a.shape) == 3 and len(b.shape) == 3:
        if a.shape[0] != b.shape[0] or a.shape[2] != b.shape[1]:
            raise ValueError(f"Shape mismatch in batched matmul: {a.shape} and {b.shape}")
        out_shape = (a.shape[0], a.shape[1], b.shape[2])
    else:
        raise ValueError(f"Unsupported shapes for matmul: {a.shape} and {b.shape}")
    return Tensor(shape=out_shape, dtype=a.dtype, op="matmul", inputs=[a, b])

def add(a: Union[Tensor, float, int], b: Union[Tensor, float, int]) -> Tensor:
    if not isinstance(a, Tensor):
        a = Tensor(shape=(), dtype="f32", op="constant", data=np.array(a, dtype=np.float32))
    if not isinstance(b, Tensor):
        b = Tensor(shape=(), dtype="f32", op="constant", data=np.array(b, dtype=np.float32))
    out_shape = _broadcast_shape(a.shape, b.shape)
    return Tensor(shape=out_shape, dtype=a.dtype, op="add", inputs=[a, b])

def sub(a: Union[Tensor, float, int], b: Union[Tensor, float, int]) -> Tensor:
    if not isinstance(a, Tensor):
        a = Tensor(shape=(), dtype="f32", op="constant", data=np.array(a, dtype=np.float32))
    if not isinstance(b, Tensor):
        b = Tensor(shape=(), dtype="f32", op="constant", data=np.array(b, dtype=np.float32))
    out_shape = _broadcast_shape(a.shape, b.shape)
    return Tensor(shape=out_shape, dtype=a.dtype, op="sub", inputs=[a, b])

def mul(a: Union[Tensor, float, int], b: Union[Tensor, float, int]) -> Tensor:
    if not isinstance(a, Tensor):
        a = Tensor(shape=(), dtype="f32", op="constant", data=np.array(a, dtype=np.float32))
    if not isinstance(b, Tensor):
        b = Tensor(shape=(), dtype="f32", op="constant", data=np.array(b, dtype=np.float32))
    out_shape = _broadcast_shape(a.shape, b.shape)
    return Tensor(shape=out_shape, dtype=a.dtype, op="mul", inputs=[a, b])

def div(a: Union[Tensor, float, int], b: Union[Tensor, float, int]) -> Tensor:
    if not isinstance(a, Tensor):
        a = Tensor(shape=(), dtype="f32", op="constant", data=np.array(a, dtype=np.float32))
    if not isinstance(b, Tensor):
        b = Tensor(shape=(), dtype="f32", op="constant", data=np.array(b, dtype=np.float32))
    out_shape = _broadcast_shape(a.shape, b.shape)
    return Tensor(shape=out_shape, dtype=a.dtype, op="div", inputs=[a, b])

def relu(x: Tensor) -> Tensor:
    return Tensor(shape=x.shape, dtype=x.dtype, op="relu", inputs=[x])

def gelu(x: Tensor) -> Tensor:
    return Tensor(shape=x.shape, dtype=x.dtype, op="gelu", inputs=[x])

def exp(x: Tensor) -> Tensor:
    return Tensor(shape=x.shape, dtype=x.dtype, op="exp", inputs=[x])

def log(x: Tensor) -> Tensor:
    return Tensor(shape=x.shape, dtype=x.dtype, op="log", inputs=[x])

def softmax(x: Tensor, axis: int = -1) -> Tensor:
    return Tensor(shape=x.shape, dtype=x.dtype, op="softmax", inputs=[x], attrs={"axis": axis})

def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    return Tensor(shape=x.shape, dtype=x.dtype, op="log_softmax", inputs=[x], attrs={"axis": axis})

def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    return Tensor(shape=(), dtype="f32", op="cross_entropy", inputs=[logits, targets])

def layernorm(x: Tensor, gamma: Tensor, beta: Tensor, eps: float = 1e-5) -> Tensor:
    return Tensor(shape=x.shape, dtype=x.dtype, op="layernorm", inputs=[x, gamma, beta], attrs={"eps": eps})

def sum(x: Tensor, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
    if axis is None:
        out_shape = (1,) if keepdims else ()
    else:
        ax = axis if axis >= 0 else len(x.shape) + axis
        s = list(x.shape)
        if keepdims:
            s[ax] = 1
            out_shape = tuple(s)
        else:
            s.pop(ax)
            out_shape = tuple(s)
    return Tensor(shape=out_shape, dtype=x.dtype, op="sum", inputs=[x], attrs={"axis": axis, "keepdims": keepdims})

def mean(x: Tensor, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
    if axis is None:
        out_shape = (1,) if keepdims else ()
    else:
        ax = axis if axis >= 0 else len(x.shape) + axis
        s = list(x.shape)
        if keepdims:
            s[ax] = 1
            out_shape = tuple(s)
        else:
            s.pop(ax)
            out_shape = tuple(s)
    return Tensor(shape=out_shape, dtype=x.dtype, op="mean", inputs=[x], attrs={"axis": axis, "keepdims": keepdims})

def transpose(x: Tensor, perm: Optional[List[int]] = None) -> Tensor:
    if perm is None:
        perm = list(reversed(range(len(x.shape))))
    out_shape = tuple(x.shape[p] for p in perm)
    return Tensor(shape=out_shape, dtype=x.dtype, op="transpose", inputs=[x], attrs={"perm": perm})

def reshape(x: Tensor, new_shape: Tuple[int, ...]) -> Tensor:
    return Tensor(shape=tuple(new_shape), dtype=x.dtype, op="reshape", inputs=[x], attrs={"shape": new_shape})

def embedding(table: Tensor, indices: Tensor) -> Tensor:
    out_shape = tuple(indices.shape) + (table.shape[1],)
    return Tensor(shape=out_shape, dtype=table.dtype, op="embedding", inputs=[table, indices])
