from __future__ import annotations
from typing import Callable, Dict, Any, Optional, List, Tuple
import json
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
    threads: int = 0,
    options: Optional[Dict[str, Any]] = None,
) -> Callable[..., Any]:
    """Compiles the graph that computes `loss` (or any outputs) for `target`.

    With an optimizer the result is a training step: `step(x=..., y=...)` runs forward, backward and the update and
    returns the loss. Without one it is a function that returns the outputs.

    - "cpu": the MiniDLC compiler (autodiff and optimizer lowering in C++, fusion, buffer assignment, loop IR with CPU
      schedules, LLVM, ORC JIT). Needs the compiled `_minidlc` module (built by CMake).
    - "interp": the same compiled plan, run by the loop IR interpreter (slow; for testing).
    - "gpu-sim": the GPU schedules (bound loops, shared memory, barriers, warp shuffles, atomics) run by the lockstep
      GPU interpreter on the host — the semantics the "cuda" target executes, checkable without a GPU (slow).
    - "cuda": the GPU schedules compiled to PTX through LLVM's NVPTX backend, run on an NVIDIA GPU by the CUDA
      driver (minidlc.gpu). To run on a machine without the compiler, build an artifact with `compile_gpu`, save it,
      and load it there with `minidlc.gpu.load` (docs/gpu.md).
    - "reference": the NumPy interpreter, with its own reverse-mode autodiff and the eager optimizer — the oracle.
    - "handwritten": the hand-written AVX2 LLVM IR kernels for a fixed two-layer ReLU MLP (a benchmark baseline;
      other models get wrong results).

    `options` passes compiler switches for ablations: {"fuse": False} (a kernel per op), {"check_all": True} (run-time
    bounds checks on every access instead of proofs), {"matmul_schedule": "naive"} (untiled CPU matmuls).
    """
    if target in ("cpu", "interp", "gpu-sim", "cuda"):
        return CompiledFunction(loss, optimizer, target, profile=profile, threads=threads, options=options)
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

    elif target == "handwritten":
        # Hand-written AVX2 LLVM IR kernels for a fixed MLP: relu(x @ W1 + b1) @ W2 + b2, cross-entropy, Adam, kept as
        # a baseline for the generated code. The traced graph is not compiled.
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

    else:
        raise ValueError(f"Unknown target: {target}")


def _optimizer_options(optimizer: Optional[Optimizer]) -> Dict[str, Any]:
    if optimizer is None:
        return {}
    if isinstance(optimizer, Adam):
        return {"optimizer": "adam", "lr": optimizer.lr, "beta1": optimizer.beta1, "beta2": optimizer.beta2,
                "eps": optimizer.eps}
    if isinstance(optimizer, SGD):
        return {"optimizer": "sgd", "lr": optimizer.lr, "momentum": optimizer.momentum}
    raise ValueError(f"the compiler lowers Adam and SGD, not {type(optimizer).__name__}")


class CompiledFunction:
    """A graph compiled by the MiniDLC compiler.

    The parameters live in the executable between calls (updated in place by the training step); `sync_params()`
    copies them back into the `nn.Parameter`s, e.g. before evaluating or sampling with another function.
    """

    def __init__(self, outputs, optimizer: Optional[Optimizer], target: str, profile: bool = False, threads: int = 0,
                 options: Optional[Dict[str, Any]] = None):
        _minidlc = _compiler_module(target)
        self.function, self.training, self._params, self._inputs = _export(outputs, optimizer)
        options = {**_optimizer_options(optimizer), "threads": threads, **(options or {})}
        if target == "cuda":
            from . import gpu
            artifact, self.dumps = _gpu_artifact(_minidlc, self.function, options, self._inputs, self.training)
            self.executable = gpu.GpuExecutable(artifact)
        else:
            self.executable, self.dumps = _minidlc.compile(self.function.to_text(), target, options)
            for name, value in _values(self.function).items():
                self.executable.set(name, value)
        self.executable.set_profiling(profile)

    def __call__(self, **feeds: np.ndarray):
        missing = set(self._inputs) - set(feeds)
        if missing:
            raise ValueError(f"missing inputs: {sorted(missing)}")
        arrays = {arg: np.asarray(feeds[name], dtype=dtype, order="C") for name, (arg, dtype) in self._inputs.items()}
        outputs = self.executable.run(arrays)
        if self.training:
            return float(outputs[0])
        return outputs[0] if len(outputs) == 1 else outputs

    def sync_params(self) -> None:
        """Copies the executable's parameters back into the nn.Parameters."""
        for name, tensor in self._params.items():
            tensor.data[...] = self.executable.get(name)

    def get_profile(self) -> List[Dict[str, Any]]:
        return self.executable.profile()

    def print_profile(self) -> None:
        rows = sorted(self.get_profile(), key=lambda p: -p["seconds"])
        total = sum(p["seconds"] for p in rows) or 1.0
        print(f"{'kernel':42s} {'calls':>6s} {'ms/call':>9s} {'share':>6s} {'GFLOP/s':>8s} {'GB/s':>7s}")
        for p in rows:
            if p["calls"] == 0:
                continue
            per = p["seconds"] / p["calls"]
            print(f"{p['name']:42s} {p['calls']:6d} {1e3 * per:9.3f} {100 * p['seconds'] / total:5.1f}% "
                  f"{p['flops'] / per / 1e9 if per else 0:8.2f} {p['bytes'] / per / 1e9 if per else 0:7.2f}")


def _compiler_module(target: str):
    try:
        from . import _minidlc
    except ImportError as e:  # pragma: no cover - depends on the build
        raise RuntimeError(f"target='{target}' needs the compiled _minidlc module: build the project with CMake "
                           "(see README.md)") from e
    return _minidlc


def _export(outputs, optimizer: Optional[Optimizer]):
    """The graph as an IR function, plus the parameters and placeholders by argument name."""
    from . import ir
    roots = [outputs] if isinstance(outputs, Tensor) else list(outputs)
    training = optimizer is not None
    if training and (len(roots) != 1 or roots[0].shape != ()):
        raise ValueError("a training step needs a single scalar loss")
    params = optimizer.parameters if optimizer is not None else None
    function = ir.export(roots, name="step" if training else "forward", params=params)
    tensors = {}
    _collect_leaves(roots, tensors)
    param_tensors = {function.arg_of[i]: t for i, t in tensors.items()
                     if i in function.arg_of and getattr(t, "op", None) == "parameter"}
    inputs = {}
    for i, t in tensors.items():
        if t.op == "placeholder" and i in function.arg_of:
            if t.name in inputs:
                raise ValueError(f"two placeholders are named '{t.name}'")
            inputs[t.name] = (function.arg_of[i], np.int32 if t.dtype == "i32" else np.float32)
    return function, training, param_tensors, inputs


def _values(function) -> Dict[str, np.ndarray]:
    """The initial values of the params and consts, by argument name."""
    return {name: np.asarray(value, dtype=np.int32 if value.dtype.kind in "iu" else np.float32, order="C")
            for name, value in function.bindings.items()}


def _gpu_artifact(_minidlc, function, options, inputs, training, ptx=True, arch="sm_75"):
    from . import gpu
    compiled, dumps = _minidlc.compile_gpu(function.to_text(), options, ptx, arch)
    artifact = gpu.GpuArtifact(
        plan=json.loads(compiled.plan_json), ptx=compiled.ptx, arch=arch,
        inputs={name: [arg, "i32" if dtype == np.int32 else "f32"] for name, (arg, dtype) in inputs.items()},
        values=_values(function), training=training, source=function.to_text(), options=dict(options),
        compiled=compiled)
    return artifact, dumps


def compile_gpu(loss: Tensor, optimizer: Optional[Optimizer] = None, arch: str = "sm_75", ptx: bool = True,
                parameters: Optional[List[Tensor]] = None, include_params: bool = True):
    """Compiles the graph that computes `loss` (a training step with an optimizer) for NVIDIA GPUs and returns a
    `minidlc.gpu.GpuArtifact`: the plan, the PTX, the initial parameter values and the input names. `save()` it and run
    it anywhere with a CUDA driver through `minidlc.gpu.load` — no compiler needed there.

    `parameters` (default: the optimizer's) numbers the parameters, so that artifacts of different graphs over the same
    model — a training step and an evaluation function — can exchange weights (`GpuFunction.get_params/set_params`).
    With include_params=False the numbered parameters' values are left out (they start as zeros): for a function that
    always receives its weights from another artifact.
    """
    _minidlc = _compiler_module("cuda")
    function, training, param_tensors, inputs = _export(loss, optimizer)
    artifact, _ = _gpu_artifact(_minidlc, function, _optimizer_options(optimizer), inputs, training, ptx, arch)
    numbering = parameters if parameters is not None else (optimizer.parameters if optimizer is not None else [])
    index = {id(t): i for i, t in enumerate(numbering)}
    artifact.params = {arg: index[id(t)] for arg, t in param_tensors.items() if id(t) in index}
    if not include_params:
        artifact.values = {arg: v for arg, v in artifact.values.items() if arg not in artifact.params}
    return artifact


def _collect_leaves(roots, out) -> None:
    stack = list(roots)
    seen = set()
    while stack:
        t = stack.pop()
        if id(t) in seen:
            continue
        seen.add(id(t))
        if not t.inputs:
            out[id(t)] = t
        stack.extend(t.inputs)
