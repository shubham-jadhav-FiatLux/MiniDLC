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
        self.m: Dict[Parameter, np.ndarray] = {
            p: np.zeros_like(p.data) for p in self.parameters if p.data is not None
        }
        self.v: Dict[Parameter, np.ndarray] = {
            p: np.zeros_like(p.data) for p in self.parameters if p.data is not None
        }

    def step(self, grads: Dict[Parameter, np.ndarray]) -> None:
        self.t += 1
        for p in self.parameters:
            if p in grads and p.data is not None:
                g = grads[p]
                # Update biased 1st and 2nd moment estimates
                self.m[p] = self.beta1 * self.m[p] + (1.0 - self.beta1) * g
                self.v[p] = self.beta2 * self.v[p] + (1.0 - self.beta2) * (g ** 2)
                # Compute bias-corrected estimates
                m_hat = self.m[p] / (1.0 - (self.beta1 ** self.t))
                v_hat = self.v[p] / (1.0 - (self.beta2 ** self.t))
                # Update parameters
                p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
