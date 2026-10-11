from __future__ import annotations
from typing import Callable, Dict, Any, Optional
import numpy as np
from .tensor import Tensor
from .optim import Optimizer
from .reference import ReferenceInterpreter

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
                # Filter grads for optimizer parameters
                param_grads = {p: grads[p] for p in optimizer.parameters if p in grads}
                optimizer.step(param_grads)

            return loss_val

        return step_fn

    elif target in ("cpu", "cuda"):
        # CPU (LLVM) / GPU (CUDA) backend driver
        raise NotImplementedError(
            f"Native target '{target}' will be enabled in milestones M2 (CPU/LLVM) and M3 (GPU/CUDA)."
        )
    else:
        raise ValueError(f"Unknown target: {target}")
