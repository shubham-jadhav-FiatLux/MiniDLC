from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from .tensor import Tensor

def _unbroadcast(grad: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """Sum out broadcast dimensions to match target shape."""
    grad_shape = grad.shape
    extra_dims = len(grad_shape) - len(target_shape)
    for _ in range(extra_dims):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(target_shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad

class ReferenceInterpreter:
    """NumPy-based reference execution engine and VJP backward evaluator."""
    def __init__(self, dtype: np.dtype = np.float32):
        self.dtype = dtype

    def forward(self, root: Tensor, feed_dict: Dict[Tensor, np.ndarray]) -> Dict[Tensor, np.ndarray]:
        values: Dict[Tensor, np.ndarray] = {}

        # Topologically sort nodes
        topo: List[Tensor] = []
        visited = set()

        def build_topo(node: Tensor) -> None:
            if node not in visited:
                visited.add(node)
                for inp in node.inputs:
                    build_topo(inp)
                topo.append(node)

        build_topo(root)

        for node in topo:
            if node in feed_dict:
                values[node] = np.asarray(feed_dict[node], dtype=self.dtype)
            elif node.data is not None:
                values[node] = np.asarray(node.data, dtype=self.dtype)
            elif node.op == "matmul":
                a, b = values[node.inputs[0]], values[node.inputs[1]]
                values[node] = a @ b
            elif node.op == "add":
                values[node] = values[node.inputs[0]] + values[node.inputs[1]]
            elif node.op == "sub":
                values[node] = values[node.inputs[0]] - values[node.inputs[1]]
            elif node.op == "mul":
                values[node] = values[node.inputs[0]] * values[node.inputs[1]]
            elif node.op == "div":
                values[node] = values[node.inputs[0]] / values[node.inputs[1]]
            elif node.op == "relu":
                x = values[node.inputs[0]]
                values[node] = np.maximum(x, 0.0)
            elif node.op == "gelu":
                x = values[node.inputs[0]]
                t = np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * (x ** 3)))
                values[node] = 0.5 * x * (1.0 + t)
            elif node.op == "exp":
                values[node] = np.exp(values[node.inputs[0]])
            elif node.op == "log":
                values[node] = np.log(values[node.inputs[0]])
            elif node.op == "softmax":
                x = values[node.inputs[0]]
                shift_x = x - np.max(x, axis=-1, keepdims=True)
                exps = np.exp(shift_x)
                values[node] = exps / np.sum(exps, axis=-1, keepdims=True)
            elif node.op == "log_softmax":
                x = values[node.inputs[0]]
                shift_x = x - np.max(x, axis=-1, keepdims=True)
                lse = np.log(np.sum(np.exp(shift_x), axis=-1, keepdims=True))
                values[node] = shift_x - lse
            elif node.op == "cross_entropy":
                logits = values[node.inputs[0]]
                targets = values[node.inputs[1]].astype(np.int64)
                # Stable cross entropy via log-sum-exp
                shift_z = logits - np.max(logits, axis=-1, keepdims=True)
                lse = np.log(np.sum(np.exp(shift_z), axis=-1, keepdims=True)) + np.max(logits, axis=-1, keepdims=True)
                N = logits.shape[0]
                correct_logits = logits[np.arange(N), targets]
                values[node] = np.mean(lse.squeeze(-1) - correct_logits)
            elif node.op == "layernorm":
                x = values[node.inputs[0]]
                gamma = values[node.inputs[1]]
                beta = values[node.inputs[2]]
                mu = np.mean(x, axis=-1, keepdims=True)
                var = np.var(x, axis=-1, keepdims=True)
                rstd = 1.0 / np.sqrt(var + 1e-5)
                x_hat = (x - mu) * rstd
                values[node] = gamma * x_hat + beta
            elif node.op == "sum":
                values[node] = np.sum(values[node.inputs[0]])
            elif node.op == "mean":
                values[node] = np.mean(values[node.inputs[0]])
            elif node.op == "transpose":
                values[node] = np.transpose(values[node.inputs[0]])
            elif node.op == "reshape":
                values[node] = np.reshape(values[node.inputs[0]], node.shape)
            elif node.op == "embedding":
                table = values[node.inputs[0]]
                idx = values[node.inputs[1]].astype(np.int64)
                values[node] = table[idx]
            else:
                raise NotImplementedError(f"Unsupported op: {node.op}")

        return values

    def backward(self, root: Tensor, values: Dict[Tensor, np.ndarray]) -> Dict[Tensor, np.ndarray]:
        grads: Dict[Tensor, np.ndarray] = {root: np.array(1.0, dtype=self.dtype)}

        topo: List[Tensor] = []
        visited = set()

        def build_topo(node: Tensor) -> None:
            if node not in visited:
                visited.add(node)
                for inp in node.inputs:
                    build_topo(inp)
                topo.append(node)

        build_topo(root)

        for node in reversed(topo):
            if node not in grads:
                continue
            dY = grads[node]

            if node.op == "matmul":
                A, B = node.inputs[0], node.inputs[1]
                valA, valB = values[A], values[B]
                dA = dY @ np.swapaxes(valB, -1, -2)
                dB = np.swapaxes(valA, -1, -2) @ dY
                grads[A] = grads.get(A, 0.0) + _unbroadcast(dA, A.shape)
                grads[B] = grads.get(B, 0.0) + _unbroadcast(dB, B.shape)
            elif node.op == "add":
                A, B = node.inputs[0], node.inputs[1]
                grads[A] = grads.get(A, 0.0) + _unbroadcast(dY, A.shape)
                grads[B] = grads.get(B, 0.0) + _unbroadcast(dY, B.shape)
            elif node.op == "sub":
                A, B = node.inputs[0], node.inputs[1]
                grads[A] = grads.get(A, 0.0) + _unbroadcast(dY, A.shape)
                grads[B] = grads.get(B, 0.0) + _unbroadcast(-dY, B.shape)
            elif node.op == "mul":
                A, B = node.inputs[0], node.inputs[1]
                valA, valB = values[A], values[B]
                grads[A] = grads.get(A, 0.0) + _unbroadcast(dY * valB, A.shape)
                grads[B] = grads.get(B, 0.0) + _unbroadcast(dY * valA, B.shape)
            elif node.op == "relu":
                X = node.inputs[0]
                valX = values[X]
                dX = dY * (valX > 0.0)
                grads[X] = grads.get(X, 0.0) + dX
            elif node.op == "cross_entropy":
                logits = node.inputs[0]
                targets = node.inputs[1]
                valLogits = values[logits]
                valTargets = values[targets].astype(np.int64)
                shift_z = valLogits - np.max(valLogits, axis=-1, keepdims=True)
                exps = np.exp(shift_z)
                softmax_vals = exps / np.sum(exps, axis=-1, keepdims=True)
                N = valLogits.shape[0]
                one_hot = np.zeros_like(valLogits)
                one_hot[np.arange(N), valTargets] = 1.0
                dLogits = (softmax_vals - one_hot) / N
                grads[logits] = grads.get(logits, 0.0) + dY * dLogits
            elif node.op == "sum":
                inp = node.inputs[0]
                grads[inp] = grads.get(inp, 0.0) + np.broadcast_to(dY, inp.shape)
            elif node.op == "mean":
                inp = node.inputs[0]
                n_elem = np.prod(inp.shape)
                grads[inp] = grads.get(inp, 0.0) + (np.broadcast_to(dY, inp.shape) / n_elem)

        return grads
