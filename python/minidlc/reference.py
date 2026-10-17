from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from .tensor import Tensor

def _unbroadcast(grad: Any, target_shape: Tuple[int, ...]) -> np.ndarray:
    """Sum out broadcast dimensions to match target shape."""
    if not isinstance(grad, np.ndarray):
        grad = np.asarray(grad)
    grad_shape = grad.shape
    extra_dims = len(grad_shape) - len(target_shape)
    for _ in range(extra_dims):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(target_shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(target_shape)

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
                axis = node.attrs.get("axis", -1)
                shift_x = x - np.max(x, axis=axis, keepdims=True)
                exps = np.exp(shift_x)
                values[node] = exps / np.sum(exps, axis=axis, keepdims=True)
            elif node.op == "log_softmax":
                x = values[node.inputs[0]]
                axis = node.attrs.get("axis", -1)
                shift_x = x - np.max(x, axis=axis, keepdims=True)
                lse = np.log(np.sum(np.exp(shift_x), axis=axis, keepdims=True))
                values[node] = shift_x - lse
            elif node.op == "cross_entropy":
                logits = values[node.inputs[0]]
                targets = values[node.inputs[1]].astype(np.int64)
                if len(logits.shape) == 3:
                    B_sz, T_sz, V_sz = logits.shape
                    logits_flat = logits.reshape(B_sz * T_sz, V_sz)
                    targets_flat = targets.reshape(B_sz * T_sz)
                    shift_z = logits_flat - np.max(logits_flat, axis=-1, keepdims=True)
                    lse = np.log(np.sum(np.exp(shift_z), axis=-1, keepdims=True)) + np.max(logits_flat, axis=-1, keepdims=True)
                    correct_logits = logits_flat[np.arange(B_sz * T_sz), targets_flat]
                    values[node] = np.mean(lse.squeeze(-1) - correct_logits)
                else:
                    shift_z = logits - np.max(logits, axis=-1, keepdims=True)
                    lse = np.log(np.sum(np.exp(shift_z), axis=-1, keepdims=True)) + np.max(logits, axis=-1, keepdims=True)
                    N = logits.shape[0]
                    correct_logits = logits[np.arange(N), targets]
                    values[node] = np.mean(lse.squeeze(-1) - correct_logits)
            elif node.op == "layernorm":
                x = values[node.inputs[0]]
                gamma = values[node.inputs[1]]
                beta = values[node.inputs[2]]
                eps = node.attrs.get("eps", 1e-5)
                mu = np.mean(x, axis=-1, keepdims=True)
                var = np.var(x, axis=-1, keepdims=True)
                rstd = 1.0 / np.sqrt(var + eps)
                x_hat = (x - mu) * rstd
                values[node] = gamma * x_hat + beta
            elif node.op == "sum":
                axis = node.attrs.get("axis", None)
                keepdims = node.attrs.get("keepdims", False)
                values[node] = np.sum(values[node.inputs[0]], axis=axis, keepdims=keepdims)
            elif node.op == "mean":
                axis = node.attrs.get("axis", None)
                keepdims = node.attrs.get("keepdims", False)
                values[node] = np.mean(values[node.inputs[0]], axis=axis, keepdims=keepdims)
            elif node.op == "transpose":
                perm = node.attrs.get("perm", None)
                values[node] = np.transpose(values[node.inputs[0]], axes=perm)
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
        if root not in values:
            values = self.forward(root, values)
        root_val = values.get(root, np.array(1.0, dtype=self.dtype))
        grads: Dict[Tensor, np.ndarray] = {
            root: np.ones(root_val.shape, dtype=self.dtype) if root_val.shape else np.array(1.0, dtype=self.dtype)
        }

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
                if len(A.shape) == 3 and len(B.shape) == 2:
                    dA = dY @ valB.T
                    B_sz, T_sz, D_sz = A.shape
                    H_sz = B.shape[1]
                    A_flat = valA.reshape(B_sz * T_sz, D_sz)
                    dY_flat = dY.reshape(B_sz * T_sz, H_sz)
                    dB = A_flat.T @ dY_flat
                    grads[A] = grads.get(A, 0.0) + _unbroadcast(dA, A.shape)
                    grads[B] = grads.get(B, 0.0) + _unbroadcast(dB, B.shape)
                else:
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
            elif node.op == "div":
                A, B = node.inputs[0], node.inputs[1]
                valA, valB = values[A], values[B]
                dA = dY / valB
                dB = -dY * valA / (valB ** 2)
                grads[A] = grads.get(A, 0.0) + _unbroadcast(dA, A.shape)
                grads[B] = grads.get(B, 0.0) + _unbroadcast(dB, B.shape)
            elif node.op == "relu":
                X = node.inputs[0]
                valX = values[X]
                dX = dY * (valX > 0.0)
                grads[X] = grads.get(X, 0.0) + dX
            elif node.op == "gelu":
                X = node.inputs[0]
                valX = values[X]
                t = np.tanh(np.sqrt(2.0 / np.pi) * (valX + 0.044715 * (valX ** 3)))
                c = np.sqrt(2.0 / np.pi) * (1.0 + 3.0 * 0.044715 * (valX ** 2))
                dX = dY * (0.5 * (1.0 + t) + 0.5 * valX * (1.0 - t ** 2) * c)
                grads[X] = grads.get(X, 0.0) + dX
            elif node.op == "exp":
                X = node.inputs[0]
                grads[X] = grads.get(X, 0.0) + dY * values[node]
            elif node.op == "log":
                X = node.inputs[0]
                grads[X] = grads.get(X, 0.0) + dY / values[X]
            elif node.op == "softmax":
                X = node.inputs[0]
                S = values[node]
                axis = node.attrs.get("axis", -1)
                dX = S * (dY - np.sum(dY * S, axis=axis, keepdims=True))
                grads[X] = grads.get(X, 0.0) + dX
            elif node.op == "log_softmax":
                X = node.inputs[0]
                axis = node.attrs.get("axis", -1)
                val_log = values[node]
                S = np.exp(val_log)
                dX = dY - S * np.sum(dY, axis=axis, keepdims=True)
                grads[X] = grads.get(X, 0.0) + dX
            elif node.op == "cross_entropy":
                logits = node.inputs[0]
                targets = node.inputs[1]
                valLogits = values[logits]
                valTargets = values[targets].astype(np.int64)
                if len(valLogits.shape) == 3:
                    B_sz, T_sz, V_sz = valLogits.shape
                    logits_flat = valLogits.reshape(B_sz * T_sz, V_sz)
                    targets_flat = valTargets.reshape(B_sz * T_sz)
                    shift_z = logits_flat - np.max(logits_flat, axis=-1, keepdims=True)
                    exps = np.exp(shift_z)
                    softmax_vals = exps / np.sum(exps, axis=-1, keepdims=True)
                    N = B_sz * T_sz
                    one_hot = np.zeros_like(logits_flat)
                    one_hot[np.arange(N), targets_flat] = 1.0
                    dLogits = ((softmax_vals - one_hot) / float(N)).reshape(B_sz, T_sz, V_sz)
                    grads[logits] = grads.get(logits, 0.0) + dY * dLogits
                else:
                    shift_z = valLogits - np.max(valLogits, axis=-1, keepdims=True)
                    exps = np.exp(shift_z)
                    softmax_vals = exps / np.sum(exps, axis=-1, keepdims=True)
                    N = valLogits.shape[0]
                    one_hot = np.zeros_like(valLogits)
                    one_hot[np.arange(N), valTargets] = 1.0
                    dLogits = (softmax_vals - one_hot) / float(N)
                    grads[logits] = grads.get(logits, 0.0) + dY * dLogits
            elif node.op == "layernorm":
                x, gamma, beta = node.inputs[0], node.inputs[1], node.inputs[2]
                val_x = values[x]
                val_gamma = values[gamma]
                eps = node.attrs.get("eps", 1e-5)
                mu = np.mean(val_x, axis=-1, keepdims=True)
                var = np.var(val_x, axis=-1, keepdims=True)
                rstd = 1.0 / np.sqrt(var + eps)
                x_hat = (val_x - mu) * rstd
                reduce_axes = tuple(range(len(dY.shape) - 1)) if len(dY.shape) > 1 else ()
                dbeta = np.sum(dY, axis=reduce_axes) if reduce_axes else dY
                dgamma = np.sum(dY * x_hat, axis=reduce_axes) if reduce_axes else dY * x_hat
                g = dY * val_gamma
                D = val_x.shape[-1]
                sum_g = np.sum(g, axis=-1, keepdims=True)
                sum_g_xhat = np.sum(g * x_hat, axis=-1, keepdims=True)
                dx = (rstd / D) * (D * g - sum_g - x_hat * sum_g_xhat)
                grads[beta] = grads.get(beta, 0.0) + _unbroadcast(dbeta, beta.shape)
                grads[gamma] = grads.get(gamma, 0.0) + _unbroadcast(dgamma, gamma.shape)
                grads[x] = grads.get(x, 0.0) + dx
            elif node.op == "sum":
                inp = node.inputs[0]
                axis = node.attrs.get("axis", None)
                keepdims = node.attrs.get("keepdims", False)
                if axis is None:
                    expanded_dY = dY
                else:
                    expanded_dY = dY if keepdims else np.expand_dims(dY, axis=axis)
                grads[inp] = grads.get(inp, 0.0) + np.broadcast_to(expanded_dY, inp.shape)
            elif node.op == "mean":
                inp = node.inputs[0]
                axis = node.attrs.get("axis", None)
                keepdims = node.attrs.get("keepdims", False)
                n_elem = inp.shape[axis] if axis is not None else np.prod(inp.shape)
                if axis is None:
                    expanded_dY = dY
                else:
                    expanded_dY = dY if keepdims else np.expand_dims(dY, axis=axis)
                grads[inp] = grads.get(inp, 0.0) + (np.broadcast_to(expanded_dY, inp.shape) / n_elem)
            elif node.op == "transpose":
                inp = node.inputs[0]
                perm = node.attrs.get("perm", None)
                if perm is None:
                    inv_perm = list(reversed(range(len(inp.shape))))
                else:
                    inv_perm = [0] * len(perm)
                    for i, p in enumerate(perm):
                        inv_perm[p] = i
                grads[inp] = grads.get(inp, 0.0) + np.transpose(dY, axes=inv_perm)
            elif node.op == "reshape":
                inp = node.inputs[0]
                grads[inp] = grads.get(inp, 0.0) + np.reshape(dY, inp.shape)
            elif node.op == "embedding":
                table = node.inputs[0]
                idx = values[node.inputs[1]].astype(np.int64)
                dTable = np.zeros(table.shape, dtype=self.dtype)
                np.add.at(dTable, idx, dY)
                grads[table] = grads.get(table, 0.0) + dTable

        return grads
