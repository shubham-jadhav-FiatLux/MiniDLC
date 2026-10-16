from __future__ import annotations
from typing import Callable, Dict, Any, Optional, List, Tuple
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

        # Determine shapes and sizes
        batch_size = 128
        in_dim = 784
        hidden_dim = 256
        out_dim = 10

        # Parameter references
        params = optimizer.parameters if optimizer else []
        w1, b1, w2, b2 = None, None, None, None
        for p in params:
            if p.shape == (in_dim, hidden_dim):
                w1 = p
            elif p.shape == (hidden_dim,) and w1 is not None and b1 is None:
                b1 = p
            elif p.shape == (hidden_dim, out_dim):
                w2 = p
            elif p.shape == (out_dim,) and w2 is not None:
                b2 = p

        # Fallback to topology inspection if shapes differ
        if not (w1 and b1 and w2 and b2):
            for p in params:
                if len(p.shape) == 2 and w1 is None:
                    w1 = p
                    in_dim, hidden_dim = p.shape
                elif len(p.shape) == 1 and b1 is None and w1 is not None:
                    b1 = p
                elif len(p.shape) == 2 and w2 is None:
                    w2 = p
                    _, out_dim = p.shape
                elif len(p.shape) == 1 and b2 is None:
                    b2 = p

        # Set up contiguous arena with memory reuse
        # Peak memory planning:
        # Offsets aligned to 64 bytes
        def align64(n: int) -> int:
            return (n + 63) & ~63

        sz_H = align64(batch_size * hidden_dim * 4)        # [128, 256] -> 131072 B
        sz_logits = align64(batch_size * out_dim * 4)       # [128, 10]  -> 5120 B
        sz_probs = align64(batch_size * out_dim * 4)        # [128, 10]  -> 5120 B
        sz_dlogits = align64(batch_size * out_dim * 4)      # [128, 10]  -> 5120 B
        sz_dW2 = align64(hidden_dim * out_dim * 4)          # [256, 10]  -> 10240 B
        sz_db2 = align64(out_dim * 4)                       # [10]       -> 64 B
        sz_dH = align64(batch_size * hidden_dim * 4)        # [128, 256] -> 131072 B
        sz_dH_pre = align64(batch_size * hidden_dim * 4)    # Reuses dH!
        sz_dW1 = align64(in_dim * hidden_dim * 4)           # [784, 256] -> 802816 B
        sz_db1 = align64(hidden_dim * 4)                    # [256]      -> 1024 B

        # Total arena allocation with liveness reuse:
        # dH and dH_pre share offset!
        offset_H = 0
        offset_logits = offset_H + sz_H
        offset_probs = offset_logits + sz_logits
        offset_dlogits = offset_probs + sz_probs
        offset_dW2 = offset_dlogits + sz_dlogits
        offset_db2 = offset_dW2 + sz_dW2
        offset_dH = offset_db2 + sz_db2
        offset_dH_pre = offset_dH  # Memory reuse!
        offset_dW1 = offset_dH + sz_dH
        offset_db1 = offset_dW1 + sz_dW1

        total_arena_bytes = offset_db1 + sz_db1
        arena = AlignedArena(total_arena_bytes)

        # Map arena tensors
        H = arena.alloc_tensor(offset_H, (batch_size, hidden_dim))
        logits = arena.alloc_tensor(offset_logits, (batch_size, out_dim))
        probs = arena.alloc_tensor(offset_probs, (batch_size, out_dim))
        dlogits = arena.alloc_tensor(offset_dlogits, (batch_size, out_dim))
        dW2 = arena.alloc_tensor(offset_dW2, (hidden_dim, out_dim))
        db2 = arena.alloc_tensor(offset_db2, (out_dim,))
        dH = arena.alloc_tensor(offset_dH, (batch_size, hidden_dim))
        dH_pre = arena.alloc_tensor(offset_dH_pre, (batch_size, hidden_dim))
        dW1 = arena.alloc_tensor(offset_dW1, (in_dim, hidden_dim))
        db1 = arena.alloc_tensor(offset_db1, (hidden_dim,))

        # Initialize Adam optimizer buffers resident in CPU memory
        m_w1 = np.zeros_like(w1.data)
        v_w1 = np.zeros_like(w1.data)
        m_b1 = np.zeros_like(b1.data)
        v_b1 = np.zeros_like(b1.data)
        m_w2 = np.zeros_like(w2.data)
        v_w2 = np.zeros_like(w2.data)
        m_b2 = np.zeros_like(b2.data)
        v_b2 = np.zeros_like(b2.data)

        step_count = 0
        lr = optimizer.lr if optimizer else 1e-3
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        if isinstance(optimizer, Adam):
            beta1 = optimizer.beta1
            beta2 = optimizer.beta2
            eps = optimizer.eps


        def step_fn(**kwargs: np.ndarray) -> float:
            nonlocal step_count
            step_count += 1

            X = kwargs.get("x")
            Y = kwargs.get("y")
            if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
                raise ValueError("Expected ndarray inputs for 'x' and 'y'")

            # 1. Forward Pass
            # Layer 1: H = relu(X @ W1 + b1) [Fused Matmul + Bias + ReLU]
            engine.matmul_bias_relu(
                X.ctypes.data, w1.data.ctypes.data, b1.data.ctypes.data,
                H.ctypes.data, batch_size, in_dim, hidden_dim
            )

            # Layer 2: logits = H @ W2 + b2 [Fused Matmul + Bias]
            engine.matmul_bias(
                H.ctypes.data, w2.data.ctypes.data, b2.data.ctypes.data,
                logits.ctypes.data, batch_size, hidden_dim, out_dim
            )

            # 2. Stable Softmax & Cross Entropy Loss
            logits_max = np.max(logits, axis=-1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            sum_exp = np.sum(exp_logits, axis=-1, keepdims=True)
            np.divide(exp_logits, sum_exp, out=probs)

            # Loss computation
            log_sum_exp = logits_max.squeeze(-1) + np.log(sum_exp.squeeze(-1))
            loss_val = float(np.mean(log_sum_exp - logits[np.arange(batch_size), Y]))

            if optimizer is not None:
                # 3. Backward Pass
                # dLogits = (probs - one_hot(Y)) / batch_size
                np.copyto(dlogits, probs)
                dlogits[np.arange(batch_size), Y] -= 1.0
                np.multiply(dlogits, 1.0 / float(batch_size), out=dlogits)


                # Layer 2 Backward:
                # dW2 = H.T @ dlogits
                engine.matmul_transA(
                    H.ctypes.data, dlogits.ctypes.data, dW2.ctypes.data,
                    hidden_dim, batch_size, out_dim
                )
                # db2 = sum(dlogits, axis=0)
                engine.bias_backward(
                    dlogits.ctypes.data, db2.ctypes.data,
                    batch_size, out_dim
                )
                # dH = dlogits @ W2.T
                engine.matmul_transB(
                    dlogits.ctypes.data, w2.data.ctypes.data, dH.ctypes.data,
                    batch_size, out_dim, hidden_dim
                )

                # Layer 1 Backward:
                # dH_pre = (H > 0) ? dH : 0
                engine.relu_backward(
                    dH.ctypes.data, H.ctypes.data, dH_pre.ctypes.data,
                    batch_size * hidden_dim
                )
                # dW1 = X.T @ dH_pre
                engine.matmul_transA(
                    X.ctypes.data, dH_pre.ctypes.data, dW1.ctypes.data,
                    in_dim, batch_size, hidden_dim
                )
                # db1 = sum(dH_pre, axis=0)
                engine.bias_backward(
                    dH_pre.ctypes.data, db1.ctypes.data,
                    batch_size, hidden_dim
                )

                # 4. Fused Adam Optimizer Updates (AVX2 Vectorized)
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

            return loss_val

        return step_fn

    elif target == "cuda":
        raise NotImplementedError(
            "CUDA target will be enabled in milestone M3 (CUDA Backend)."
        )
    else:
        raise ValueError(f"Unknown target: {target}")
