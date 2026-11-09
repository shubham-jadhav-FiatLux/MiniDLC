"""Checkpoints: the parameters and optimizer state of a training step, in a .npz file.

Arrays are keyed by the compiled step's argument names, which every target shares: parameters by their names
(e.g. `fc1.weight`), optimizer state after them (`fc1.weight.m`, `fc1.weight.v`, `adam.beta1_pow`,
`adam.beta2_pow` for Adam; `fc1.weight.velocity` for SGD with momentum). A checkpoint written by one target
(`cpu`, `gpu-sim`, `cuda`, a GPU artifact on Colab, or the `reference` interpreter) can be loaded by any other that
trains the same model with the same optimizer.

    step.save_checkpoint("run.npz")
    step.load_checkpoint("run.npz")   # same names, shapes and dtypes; raises ValueError otherwise
"""
from __future__ import annotations

import json
from typing import Dict, Mapping, Tuple

import numpy as np

FORMAT = 1
_META = "__minidlc_checkpoint__"


def save(path: str, arrays: Mapping[str, np.ndarray], target: str, function: str = "") -> None:
    """Writes `arrays` (argument name → array) and a description of them to `path` (exactly that path)."""
    meta = {"format": FORMAT, "target": target, "function": function,
            "arrays": {k: {"dtype": str(np.asarray(v).dtype), "shape": list(np.shape(v))} for k, v in arrays.items()}}
    with open(path, "wb") as f:
        np.savez(f, **{k: np.asarray(v) for k, v in arrays.items()}, **{_META: np.array(json.dumps(meta))})


def load(path: str) -> Tuple[Dict[str, np.ndarray], dict]:
    """The arrays of a checkpoint and its description."""
    with np.load(path, allow_pickle=False) as npz:
        if _META not in npz.files:
            raise ValueError(f"{path} is not a MiniDLC checkpoint")
        meta = json.loads(str(npz[_META]))
        if meta.get("format") != FORMAT:
            raise ValueError(f"{path}: checkpoint format {meta.get('format')}, expected {FORMAT}")
        arrays = {k: npz[k] for k in npz.files if k != _META}
    return arrays, meta


def check(arrays: Mapping[str, np.ndarray], expected: Mapping[str, Tuple[tuple, np.dtype]], path: str,
          strict: bool = True) -> None:
    """Raises ValueError unless `arrays` holds every expected name with its shape and dtype (and, if strict, nothing
    else)."""
    problems = []
    missing = sorted(set(expected) - set(arrays))
    if missing:
        problems.append(f"missing {missing}")
    if strict:
        extra = sorted(set(arrays) - set(expected))
        if extra:
            problems.append(f"not in this step: {extra}")
    for name, (shape, dtype) in expected.items():
        if name in arrays:
            a = arrays[name]
            if tuple(a.shape) != tuple(shape) or a.dtype != np.dtype(dtype):
                problems.append(f"'{name}' is {a.dtype}{list(a.shape)}, the step has {np.dtype(dtype)}{list(shape)}")
    if problems:
        raise ValueError(f"checkpoint {path} does not fit this training step: " + "; ".join(problems))
