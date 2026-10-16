from __future__ import annotations
from typing import Callable, Dict, Any, Optional, List, Tuple
import time
import numpy as np
from .tensor import Tensor
from .optim import Optimizer, Adam, SGD
from .reference import ReferenceInterpreter
from .llvm_engine import LLVMCPUEngine

class AlignedArena:
    """Contiguous memory arena with 64-byte alignment for AVX2 execution."""
    def __init__(self, size_bytes: int, alignment: int = 64):
        self.capacity = size_bytes
        self.raw = np.zeros(size_bytes + alignment, dtype=np.uint8)
        self.offset = (-self.raw.ctypes.data) % alignment
        self.buffer = self.raw[self.offset : self.offset + size_bytes]

    def alloc_tensor(self, offset: int, shape: Tuple[int, ...], dtype=np.float32) -> np.ndarray:
        itemsize = np.dtype(dtype).itemsize
        nbytes = int(np.prod(shape)) * itemsize
        return self.buffer[offset : offset + nbytes].view(dtype).reshape(shape)

def compile(
    loss: Tensor,
    optimizer: Optional[Optimizer] = None,
    target: str = "reference",
    profile: bool = False,
) -> Callable[..., float]:
    """Compiles forward, autodiff backward, and optimizer update into a callable step function."""
    if target == "reference":
        interp = ReferenceInterpreter(dtype=np.float32)

        # Collect placeholders
        placeholders = {}
        visited = set()

        def find_placeholders(node: Tensor) -> None:
            if node not in visited:
                visited.add(node)
                if node.op == "placeholder" and node.name:
                    placeholders[node.name] = node
                for inp in node.inputs:
                    find_placeholders(inp)

        find_placeholders(loss)

        def step_fn(**kwargs: np.ndarray) -> float:
            feed_dict = {}
            for name, val in kwargs.items():
                if name in placeholders:
                    feed_dict[placeholders[name]] = val

            # Forward
            vals = interp.forward(loss, feed_dict)
            loss_val = float(vals[loss])

            # Backward
            if optimizer is not None:
                grads = interp.backward(loss, vals)
                param_grads = {p: grads[p] for p in optimizer.parameters if p in grads}
                optimizer.step(param_grads)

            return loss_val

        return step_fn

    elif target == "cpu":
        # Native LLVM CPU Backend with AVX2 SIMD and Contiguous Arena Execution
        engine = LLVMCPUEngine.get()

        # Find model parameters and placeholders
        visited = set()
        placeholders: Dict[str, Tensor] = {}
        ops_list: List[Tensor] = []

        def topo_sort(node: Tensor) -> None:
            if node not in visited:
                visited.add(node)
                for inp in node.inputs:
                    topo_sort(inp)
                ops_list.append(node)
                if node.op == "placeholder" and node.name:
                    placeholders[node.name] = node

        topo_sort(loss)

        # Dynamically determine model architecture and layer dimensions from optimizer parameters
        params = optimizer.parameters if optimizer else []
        weights = [p for p in params if len(p.shape) == 2]
        biases = [p for p in params if len(p.shape) == 1]

        if len(weights) < 2:
            raise ValueError(f"Expected at least 2 weight matrices for execution, found {len(weights)}")

        w1, w2 = weights[0], weights[1]
        b1 = biases[0] if len(biases) > 0 else None
        b2 = biases[1] if len(biases) > 1 else None

        in_dim = w1.shape[0]
        hidden_dim = w1.shape[1]
        out_dim = w2.shape[1]

        # Dynamic arena allocation helper
        def align64(n: int) -> int:
            return (n + 63) & ~63

        current_arena: Optional[AlignedArena] = None
        current_arena_batch_size: int = 0
        arena_tensors: Dict[str, np.ndarray] = {}

        def setup_arena(batch_sz: int) -> None:
            nonlocal current_arena, current_arena_batch_size, arena_tensors
            if current_arena is not None and current_arena_batch_size == batch_sz:
                return

            sz_H = align64(batch_sz * hidden_dim * 4)
            sz_logits = align64(batch_sz * out_dim * 4)
            sz_probs = align64(batch_sz * out_dim * 4)
            sz_dlogits = align64(batch_sz * out_dim * 4)
            sz_dW2 = align64(hidden_dim * out_dim * 4)
            sz_db2 = align64(out_dim * 4)
            sz_dH = align64(batch_sz * hidden_dim * 4)
            sz_dW1 = align64(in_dim * hidden_dim * 4)
            sz_db1 = align64(hidden_dim * 4)

            offset_H = 0
            offset_logits = offset_H + sz_H
            offset_probs = offset_logits + sz_logits
            offset_dlogits = offset_probs + sz_probs
            offset_dW2 = offset_dlogits + sz_dlogits
            offset_db2 = offset_dW2 + sz_dW2
            offset_dH = offset_db2 + sz_db2
            offset_dH_pre = offset_dH  # Liveness reuse: dH_pre reuses dH offset
            offset_dW1 = offset_dH + sz_dH
            offset_db1 = offset_dW1 + sz_dW1

            total_bytes = offset_db1 + sz_db1
            current_arena = AlignedArena(total_bytes)
            current_arena_batch_size = batch_sz

            arena_tensors = {
                "H": current_arena.alloc_tensor(offset_H, (batch_sz, hidden_dim)),
                "logits": current_arena.alloc_tensor(offset_logits, (batch_sz, out_dim)),
                "probs": current_arena.alloc_tensor(offset_probs, (batch_sz, out_dim)),
                "dlogits": current_arena.alloc_tensor(offset_dlogits, (batch_sz, out_dim)),
                "dW2": current_arena.alloc_tensor(offset_dW2, (hidden_dim, out_dim)),
                "db2": current_arena.alloc_tensor(offset_db2, (out_dim,)),
                "dH": current_arena.alloc_tensor(offset_dH, (batch_sz, hidden_dim)),
                "dH_pre": current_arena.alloc_tensor(offset_dH_pre, (batch_sz, hidden_dim)),
                "dW1": current_arena.alloc_tensor(offset_dW1, (in_dim, hidden_dim)),
                "db1": current_arena.alloc_tensor(offset_db1, (hidden_dim,)),
            }

        # Initialize Adam optimizer buffers resident in CPU memory
        m_w1 = np.zeros_like(w1.data)
        v_w1 = np.zeros_like(w1.data)
        m_b1 = np.zeros_like(b1.data) if b1 else None
        v_b1 = np.zeros_like(b1.data) if b1 else None
        m_w2 = np.zeros_like(w2.data)
        v_w2 = np.zeros_like(w2.data)
        m_b2 = np.zeros_like(b2.data) if b2 else None
        v_b2 = np.zeros_like(b2.data) if b2 else None

        step_count = 0
        lr = optimizer.lr if optimizer else 1e-3
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        if isinstance(optimizer, Adam):
            beta1 = optimizer.beta1
            beta2 = optimizer.beta2
            eps = optimizer.eps

        profile_stats: Dict[str, float] = {
            "fwd_layer1_fused": 0.0,
            "fwd_layer2_bias": 0.0,
            "softmax_loss": 0.0,
            "bwd_dlogits": 0.0,
            "bwd_layer2": 0.0,
            "bwd_layer1": 0.0,
            "opt_adam": 0.0,
        }
        profile_count: int = 0

        def step_fn(**kwargs: np.ndarray) -> float:
            nonlocal step_count, profile_count
            step_count += 1
            if profile:
                profile_count += 1

            X = kwargs.get("x")
            Y = kwargs.get("y")
            if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
                raise ValueError("Expected ndarray inputs for 'x' and 'y'")

            batch_sz = X.shape[0]
            setup_arena(batch_sz)

            H = arena_tensors["H"]
            logits = arena_tensors["logits"]
            probs = arena_tensors["probs"]
            dlogits = arena_tensors["dlogits"]
            dW2 = arena_tensors["dW2"]
            db2 = arena_tensors["db2"]
            dH = arena_tensors["dH"]
            dH_pre = arena_tensors["dH_pre"]
            dW1 = arena_tensors["dW1"]
            db1 = arena_tensors["db1"]

            # 1. Forward Pass
            # Layer 1: H = relu(X @ W1 + b1) [Fused Matmul + Bias + ReLU]
            t0 = time.perf_counter() if profile else 0.0
            engine.matmul_bias_relu(
                X.ctypes.data, w1.data.ctypes.data, b1.data.ctypes.data,
                H.ctypes.data, batch_sz, in_dim, hidden_dim
            )
            if profile:
                profile_stats["fwd_layer1_fused"] += (time.perf_counter() - t0)

            # Layer 2: logits = H @ W2 + b2 [Fused Matmul + Bias]
            t0 = time.perf_counter() if profile else 0.0
            engine.matmul_bias(
                H.ctypes.data, w2.data.ctypes.data, b2.data.ctypes.data,
                logits.ctypes.data, batch_sz, hidden_dim, out_dim
            )
            if profile:
                profile_stats["fwd_layer2_bias"] += (time.perf_counter() - t0)

            # 2. Stable Softmax & Cross Entropy Loss
            t0 = time.perf_counter() if profile else 0.0
            logits_max = np.max(logits, axis=-1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            sum_exp = np.sum(exp_logits, axis=-1, keepdims=True)
            np.divide(exp_logits, sum_exp, out=probs)

            # Loss computation
            log_sum_exp = logits_max.squeeze(-1) + np.log(sum_exp.squeeze(-1))
            loss_val = float(np.mean(log_sum_exp - logits[np.arange(batch_sz), Y]))
            if profile:
                profile_stats["softmax_loss"] += (time.perf_counter() - t0)

            if optimizer is not None:
                # 3. Backward Pass
                # dLogits = (probs - one_hot(Y)) / batch_sz
                t0 = time.perf_counter() if profile else 0.0
                np.copyto(dlogits, probs)
                dlogits[np.arange(batch_sz), Y] -= 1.0
                np.multiply(dlogits, 1.0 / float(batch_sz), out=dlogits)
                if profile:
                    profile_stats["bwd_dlogits"] += (time.perf_counter() - t0)

                # Layer 2 Backward:
                # dW2 = H.T @ dlogits
                t0 = time.perf_counter() if profile else 0.0
                engine.matmul_transA(
                    H.ctypes.data, dlogits.ctypes.data, dW2.ctypes.data,
                    hidden_dim, batch_sz, out_dim
                )
                # db2 = sum(dlogits, axis=0)
                engine.bias_backward(
                    dlogits.ctypes.data, db2.ctypes.data,
                    batch_sz, out_dim
                )
                # dH = dlogits @ W2.T
                engine.matmul_transB(
                    dlogits.ctypes.data, w2.data.ctypes.data, dH.ctypes.data,
                    batch_sz, out_dim, hidden_dim
                )
                if profile:
                    profile_stats["bwd_layer2"] += (time.perf_counter() - t0)

                # Layer 1 Backward:
                # dH_pre = (H > 0) ? dH : 0
                t0 = time.perf_counter() if profile else 0.0
                engine.relu_backward(
                    dH.ctypes.data, H.ctypes.data, dH_pre.ctypes.data,
                    batch_sz * hidden_dim
                )
                # dW1 = X.T @ dH_pre
                engine.matmul_transA(
                    X.ctypes.data, dH_pre.ctypes.data, dW1.ctypes.data,
                    in_dim, batch_sz, hidden_dim
                )
                # db1 = sum(dH_pre, axis=0)
                engine.bias_backward(
                    dH_pre.ctypes.data, db1.ctypes.data,
                    batch_sz, hidden_dim
                )
                if profile:
                    profile_stats["bwd_layer1"] += (time.perf_counter() - t0)

                # 4. Fused Adam Optimizer Updates (AVX2 Vectorized)
                t0 = time.perf_counter() if profile else 0.0
                bc1 = 1.0 - (beta1 ** step_count)
                bc2 = 1.0 - (beta2 ** step_count)

                # W1 update
                engine.adam_update(
                    w1.data.ctypes.data, dW1.ctypes.data,
                    m_w1.ctypes.data, v_w1.ctypes.data,
                    lr, beta1, beta2, eps, bc1, bc2, w1.data.size
                )
                # b1 update
                engine.adam_update(
                    b1.data.ctypes.data, db1.ctypes.data,
                    m_b1.ctypes.data, v_b1.ctypes.data,
                    lr, beta1, beta2, eps, bc1, bc2, b1.data.size
                )
                # W2 update
                engine.adam_update(
                    w2.data.ctypes.data, dW2.ctypes.data,
                    m_w2.ctypes.data, v_w2.ctypes.data,
                    lr, beta1, beta2, eps, bc1, bc2, w2.data.size
                )
                # b2 update
                engine.adam_update(
                    b2.data.ctypes.data, db2.ctypes.data,
                    m_b2.ctypes.data, v_b2.ctypes.data,
                    lr, beta1, beta2, eps, bc1, bc2, b2.data.size
                )
                if profile:
                    profile_stats["opt_adam"] += (time.perf_counter() - t0)

            return loss_val

        def get_profile() -> Dict[str, Any]:
            total_sec = sum(profile_stats.values())
            res = {}
            for k, sec in profile_stats.items():
                res[k] = {
                    "total_ms": sec * 1e3,
                    "mean_ms": (sec / max(profile_count, 1)) * 1e3,
                    "pct": (100.0 * sec / total_sec) if total_sec > 0 else 0.0,
                }
            return res

        def print_profile() -> None:
            prof = get_profile()
            total_ms = sum(v["total_ms"] for v in prof.values())
            print("\n=== MiniDLC Per-Kernel Execution Profile ===")
            print(f"{'Kernel / Operation':<25} | {'Total (ms)':>10} | {'Mean (ms)':>10} | {'% Step':>8}")
            print("-" * 62)
            for k, v in prof.items():
                print(f"{k:<25} | {v['total_ms']:10.2f} | {v['mean_ms']:10.3f} | {v['pct']:7.1f}%")
            print("-" * 62)
            print(f"{'Total Measured Time':<25} | {total_ms:10.2f} ms (Calls: {profile_count})\n")

        setattr(step_fn, "get_profile", get_profile)
        setattr(step_fn, "print_profile", print_profile)

        return step_fn

    elif target == "cuda":
        raise NotImplementedError(
            "CUDA target will be enabled in milestone M3 (CUDA Backend)."
        )
    else:
        raise ValueError(f"Unknown target: {target}")
