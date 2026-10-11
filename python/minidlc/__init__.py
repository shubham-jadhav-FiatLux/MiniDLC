"""MiniDLC: A Mini Deep-Learning Compiler"""

from .tensor import Tensor, placeholder
from .compile import compile
from . import ops
from . import nn
from . import optim
from . import data
from .reference import ReferenceInterpreter

# Top-level functional op aliases
matmul = ops.matmul
add = ops.add
sub = ops.sub
mul = ops.mul
div = ops.div
relu = ops.relu
gelu = ops.gelu
exp = ops.exp
log = ops.log
softmax = ops.softmax
log_softmax = ops.log_softmax
cross_entropy = ops.cross_entropy
layernorm = ops.layernorm
sum = ops.sum
mean = ops.mean
transpose = ops.transpose
reshape = ops.reshape
embedding = ops.embedding

__version__ = "0.1.0"

__all__ = [
    "Tensor",
    "placeholder",
    "compile",
    "ops",
    "nn",
    "optim",
    "data",
    "ReferenceInterpreter",
    "matmul",
    "add",
    "sub",
    "mul",
    "div",
    "relu",
    "gelu",
    "exp",
    "log",
    "softmax",
    "log_softmax",
    "cross_entropy",
    "layernorm",
    "sum",
    "mean",
    "transpose",
    "reshape",
    "embedding",
]
