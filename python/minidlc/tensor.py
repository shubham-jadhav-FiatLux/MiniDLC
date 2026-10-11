from __future__ import annotations
from typing import Tuple, List, Optional, Any
import numpy as np

class Tensor:
    """Symbolic Tensor node for MiniDLC tracing and computation."""
    _id_counter: int = 0

    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype: str = "f32",
        op: Optional[str] = None,
        inputs: Optional[List[Tensor]] = None,
        data: Optional[np.ndarray] = None,
        name: Optional[str] = None,
    ):
        self.shape: Tuple[int, ...] = tuple(shape)
        self.dtype: str = dtype
        self.op: Optional[str] = op
        self.inputs: List[Tensor] = inputs or []
        self.data: Optional[np.ndarray] = data
        self.id: int = Tensor._id_counter
        Tensor._id_counter += 1
        self.name: str = name or f"v{self.id}"
        self.grad: Optional[Tensor] = None

    def __add__(self, other: Any) -> Tensor:
        from . import ops
        return ops.add(self, other)

    def __radd__(self, other: Any) -> Tensor:
        from . import ops
        return ops.add(other, self)

    def __sub__(self, other: Any) -> Tensor:
        from . import ops
        return ops.sub(self, other)

    def __rsub__(self, other: Any) -> Tensor:
        from . import ops
        return ops.sub(other, self)

    def __mul__(self, other: Any) -> Tensor:
        from . import ops
        return ops.mul(self, other)

    def __rmul__(self, other: Any) -> Tensor:
        from . import ops
        return ops.mul(other, self)

    def __truediv__(self, other: Any) -> Tensor:
        from . import ops
        return ops.div(self, other)

    def __matmul__(self, other: Tensor) -> Tensor:
        from . import ops
        return ops.matmul(self, other)

    def __repr__(self) -> str:
        return f"Tensor({self.name}: {self.dtype}{list(self.shape)}, op={self.op})"

def placeholder(shape: Tuple[int, ...], dtype: str = "f32", name: Optional[str] = None) -> Tensor:
    """Create a symbolic input placeholder."""
    return Tensor(shape=shape, dtype=dtype, op="placeholder", name=name)
