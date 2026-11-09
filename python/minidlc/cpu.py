"""CPU artifacts: a compiled step for this CPU, saved and run later — or on another machine with the same kind of CPU —
without recompiling.

An artifact is the plan (memory sizes, argument and result slots, the launches) as JSON, an object file of the
kernels, the initial values of the parameters and constants, and what the code was compiled for (target triple, CPU
features). Loading links the object file into the process with the compiled module's JIT; a host whose CPU lacks a
feature the code uses (e.g. AVX2) is refused. The counterpart of minidlc.gpu's GPU artifacts.

    artifact = dl.compile_cpu(loss, optimizer)      # compile here
    artifact.save("mnist_step")
    step = minidlc.cpu.load("mnist_step")           # later, elsewhere: step(x=..., y=...)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from . import checkpoint

_DTYPES = {"f32": np.float32, "i32": np.int32}


@dataclass
class CpuArtifact:
    plan: Dict[str, Any]
    object: bytes
    kernel_names: List[str]
    triple: str
    cpu: str
    features: str
    inputs: Dict[str, List[str]] = field(default_factory=dict)  # placeholder name -> [argument name, dtype]
    values: Dict[str, np.ndarray] = field(default_factory=dict)  # initial params and consts
    training: bool = False
    params: Dict[str, int] = field(default_factory=dict)  # parameter argument -> index in the model's parameters

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "plan.json"), "w") as f:
            json.dump(self.plan, f, indent=1)
        with open(os.path.join(directory, "kernels.o"), "wb") as f:
            f.write(self.object)
        with open(os.path.join(directory, "meta.json"), "w") as f:
            json.dump({"kernel_names": self.kernel_names, "triple": self.triple, "cpu": self.cpu,
                       "features": self.features, "inputs": self.inputs, "training": self.training,
                       "params": self.params}, f, indent=1)
        np.savez_compressed(os.path.join(directory, "values.npz"), **self.values)

    @staticmethod
    def load(directory: str) -> "CpuArtifact":
        with open(os.path.join(directory, "plan.json")) as f:
            plan = json.load(f)
        with open(os.path.join(directory, "kernels.o"), "rb") as f:
            obj = f.read()
        with open(os.path.join(directory, "meta.json")) as f:
            meta = json.load(f)
        with np.load(os.path.join(directory, "values.npz")) as npz:
            values = {k: npz[k] for k in npz.files}
        return CpuArtifact(plan, obj, meta["kernel_names"], meta["triple"], meta["cpu"], meta["features"],
                           meta["inputs"], values, meta["training"], meta.get("params", {}))


class CpuFunction:
    """A CPU artifact behind the interface of `dl.compile`'s result: `step(x=..., y=...)` returns the loss (training)
    or the outputs; checkpoints, parameters by index, and the executable's per-kernel profile."""

    def __init__(self, artifact: CpuArtifact, threads: int = 0):
        from .compile import _compiler_module
        _minidlc = _compiler_module("cpu")
        self.artifact = artifact
        self.executable = _minidlc.load_cpu(json.dumps(artifact.plan), artifact.object, artifact.kernel_names,
                                            artifact.triple, artifact.features, threads)
        for name, value in artifact.values.items():
            self.executable.set(name, np.ascontiguousarray(value))

    def __call__(self, **feeds: np.ndarray):
        missing = set(self.artifact.inputs) - set(feeds)
        if missing:
            raise ValueError(f"missing inputs: {sorted(missing)}")
        arrays = {arg: np.asarray(feeds[name], dtype=_DTYPES[dtype], order="C")
                  for name, (arg, dtype) in self.artifact.inputs.items()}
        outputs = self.executable.run(arrays)
        if self.artifact.training:
            return float(outputs[0])
        return outputs[0] if len(outputs) == 1 else outputs

    def get_params(self) -> Dict[int, np.ndarray]:
        return {i: self.executable.get(arg) for arg, i in self.artifact.params.items()}

    def set_params(self, params: Dict[int, np.ndarray]) -> None:
        for arg, i in self.artifact.params.items():
            if i in params:
                self.executable.set(arg, np.ascontiguousarray(params[i], dtype=np.float32))

    def _stored_args(self) -> Dict[str, Any]:
        return {a["name"]: (tuple(a["shape"]), _DTYPES[a["dtype"]]) for a in self.artifact.plan["args"]
                if a["role"] in ("param", "state")}

    def save_checkpoint(self, path: str) -> None:
        checkpoint.save(path, {name: self.executable.get(name) for name in self._stored_args()}, "cpu",
                        self.artifact.plan.get("function", ""))

    def load_checkpoint(self, path: str, strict: bool = True) -> None:
        arrays, _ = checkpoint.load(path)
        expected = self._stored_args()
        checkpoint.check(arrays, expected, path, strict)
        for name, (_, dtype) in expected.items():
            self.executable.set(name, np.ascontiguousarray(arrays[name], dtype=dtype))


def load(directory: str, threads: int = 0) -> CpuFunction:
    """Loads a saved CPU artifact and links its code (no compilation)."""
    return CpuFunction(CpuArtifact.load(directory), threads)
