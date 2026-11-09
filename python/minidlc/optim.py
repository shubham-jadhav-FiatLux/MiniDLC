from __future__ import annotations
from typing import List, Dict
import numpy as np
from .nn import Parameter

class Optimizer:
    def __init__(self, parameters: List[Parameter], lr: float = 1e-3) -> None:
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self) -> None:
        for p in self.parameters:
            p.grad = None

    def step(self, grads: Dict[Parameter, np.ndarray]) -> None:
        raise NotImplementedError

    # Checkpoints (minidlc/checkpoint.py): the state under the compiled step's argument names. `names` maps
    # id(parameter) to the parameter's argument name.
    def state_arrays(self, names: Dict[int, str]) -> Dict[str, np.ndarray]:
        return {}

    def load_state_arrays(self, arrays: Dict[str, np.ndarray], names: Dict[int, str]) -> None:
        pass

class SGD(Optimizer):
    def __init__(self, parameters: List[Parameter], lr: float = 1e-3, momentum: float = 0.0) -> None:
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.velocities: Dict[Parameter, np.ndarray] = {
            p: np.zeros_like(p.data) for p in self.parameters if p.data is not None
        }

    def step(self, grads: Dict[Parameter, np.ndarray]) -> None:
        for p in self.parameters:
            if p in grads and p.data is not None:
                g = grads[p]
                if self.momentum > 0.0:
                    v = self.momentum * self.velocities[p] + g
                    self.velocities[p] = v
                    p.data -= self.lr * v
                else:
                    p.data -= self.lr * g

    def state_arrays(self, names: Dict[int, str]) -> Dict[str, np.ndarray]:
        if self.momentum <= 0.0:  # plain SGD has no state (nor does the compiled step)
            return {}
        return {names[id(p)] + ".velocity": self.velocities[p].astype(np.float32) for p in self.parameters
                if id(p) in names and p in self.velocities}

    def load_state_arrays(self, arrays: Dict[str, np.ndarray], names: Dict[int, str]) -> None:
        for p in self.parameters:
            key = names.get(id(p), "") + ".velocity"
            if id(p) in names and key in arrays:
                self.velocities[p] = np.array(arrays[key], dtype=p.data.dtype)

class Adam(Optimizer):
    def __init__(
        self,
        parameters: List[Parameter],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(parameters, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        # beta^t as running products, as the compiled step keeps them, so that checkpoints carry them exactly
        self.beta1_pow = 1.0
        self.beta2_pow = 1.0
        self.m: Dict[Parameter, np.ndarray] = {
            p: np.zeros_like(p.data) for p in self.parameters if p.data is not None
        }
        self.v: Dict[Parameter, np.ndarray] = {
            p: np.zeros_like(p.data) for p in self.parameters if p.data is not None
        }

    def step(self, grads: Dict[Parameter, np.ndarray]) -> None:
        self.t += 1
        self.beta1_pow *= self.beta1
        self.beta2_pow *= self.beta2
        for p in self.parameters:
            if p in grads and p.data is not None:
                g = grads[p]
                # Update biased 1st and 2nd moment estimates
                self.m[p] = self.beta1 * self.m[p] + (1.0 - self.beta1) * g
                self.v[p] = self.beta2 * self.v[p] + (1.0 - self.beta2) * (g ** 2)
                # Compute bias-corrected estimates
                m_hat = self.m[p] / (1.0 - self.beta1_pow)
                v_hat = self.v[p] / (1.0 - self.beta2_pow)
                # Update parameters
                p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def state_arrays(self, names: Dict[int, str]) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for p in self.parameters:
            if id(p) in names and p in self.m:
                out[names[id(p)] + ".m"] = self.m[p].astype(np.float32)
                out[names[id(p)] + ".v"] = self.v[p].astype(np.float32)
        out["adam.beta1_pow"] = np.float32(self.beta1_pow)
        out["adam.beta2_pow"] = np.float32(self.beta2_pow)
        return out

    def load_state_arrays(self, arrays: Dict[str, np.ndarray], names: Dict[int, str]) -> None:
        for p in self.parameters:
            if id(p) in names and names[id(p)] + ".m" in arrays:
                self.m[p] = np.array(arrays[names[id(p)] + ".m"], dtype=p.data.dtype)
                self.v[p] = np.array(arrays[names[id(p)] + ".v"], dtype=p.data.dtype)
        self.beta1_pow = float(arrays["adam.beta1_pow"])
        self.beta2_pow = float(arrays["adam.beta2_pow"])
        # The step count, for information: the update uses the powers.
        if 0.0 < self.beta2_pow < 1.0:
            self.t = int(round(np.log(self.beta2_pow) / np.log(self.beta2)))
