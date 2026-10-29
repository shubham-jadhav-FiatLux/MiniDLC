"""Training through generated code: dl.compile(loss, optimizer, target="cpu") builds forward, backward and the
update as one graph, compiles it, and keeps the parameters in the executable. Its loss trajectory must follow eager
training in the reference interpreter (float32), and the trained parameters must come back with sync_params(). The
same holds for the GPU schedules ("gpu-sim": run by the lockstep GPU interpreter), atomics and shuffles included.
Needs the compiled _minidlc module; skipped otherwise.
"""
from __future__ import annotations

import numpy as np
import pytest

import minidlc as dl
from minidlc import nn, optim
from minidlc.reference import ReferenceInterpreter

pytest.importorskip("minidlc._minidlc", reason="the _minidlc module is not built")


class MLP(nn.Module):
    def __init__(self, act):
        super().__init__()
        self.fc1 = nn.Linear(20, 32)
        self.fc2 = nn.Linear(32, 5)
        self.act = act

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def mlp(act):
    def build():
        model = MLP(act)
        x = dl.placeholder((16, 20), name="x")
        y = dl.placeholder((16,), dtype="i32", name="y")
        return model, dl.cross_entropy(model(x), y), lambda r: {"x": r.standard_normal((16, 20)).astype(np.float32),
                                                                 "y": r.integers(0, 5, 16).astype(np.int32)}
    return build


def chargpt():
    model = nn.CharGPT(vocab_size=11, block_size=8, d_model=16, num_heads=2, num_layers=2)
    x = dl.placeholder((2, 8), dtype="i32", name="x")
    y = dl.placeholder((2, 8), dtype="i32", name="y")
    return model, dl.cross_entropy(model(x), y), lambda r: {"x": r.integers(0, 11, (2, 8)).astype(np.int32),
                                                             "y": r.integers(0, 11, (2, 8)).astype(np.int32)}


MODELS = {"mlp_relu": mlp(dl.relu), "mlp_gelu": mlp(dl.gelu), "chargpt": chargpt}
OPTIMIZERS = {"adam": lambda p: optim.Adam(p, lr=0.01), "sgd_momentum": lambda p: optim.SGD(p, lr=0.05, momentum=0.9)}


def trajectory(model_name, opt_name, target, steps=6):
    np.random.seed(0)
    model, loss, batch = MODELS[model_name]()
    step = dl.compile(loss, optimizer=OPTIMIZERS[opt_name](model.parameters()), target=target)
    rng = np.random.default_rng(1)
    return [step(**batch(rng)) for _ in range(steps)], model, step


@pytest.mark.parametrize("model_name", list(MODELS))
@pytest.mark.parametrize("opt_name", list(OPTIMIZERS))
@pytest.mark.parametrize("target", ["cpu", "gpu-sim"])
def test_compiled_training_follows_eager_training(model_name, opt_name, target):
    eager, _, _ = trajectory(model_name, opt_name, "reference")
    compiled, _, _ = trajectory(model_name, opt_name, target)
    np.testing.assert_allclose(compiled, eager, rtol=2e-4, atol=2e-4)


def test_sync_params_returns_the_trained_weights():
    np.random.seed(0)
    model, loss, batch = MODELS["mlp_relu"]()
    before = model.fc1.weight.data.copy()
    step = dl.compile(loss, optimizer=optim.Adam(model.parameters(), lr=0.01), target="cpu")
    rng = np.random.default_rng(1)
    for _ in range(3):
        step(**batch(rng))
    np.testing.assert_array_equal(model.fc1.weight.data, before)  # the executable owns the weights while training
    step.sync_params()
    assert not np.allclose(model.fc1.weight.data, before)


def test_inference_function_matches_the_reference():
    np.random.seed(0)
    model = MLP(dl.gelu)
    x = dl.placeholder((7, 20), name="x")
    logits = model(x)
    infer = dl.compile(logits, target="cpu")
    xb = np.random.default_rng(2).standard_normal((7, 20)).astype(np.float32)
    want = ReferenceInterpreter().forward(logits, {x: xb})[logits]
    np.testing.assert_allclose(infer(x=xb), want, rtol=1e-5, atol=1e-5)


def test_a_compiled_step_learns():
    # A linearly separable synthetic task: the loss must fall well below chance.
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((5, 20)).astype(np.float32) * 3
    np.random.seed(0)
    model, loss, _ = MODELS["mlp_relu"]()
    step = dl.compile(loss, optimizer=optim.Adam(model.parameters(), lr=0.01), target="cpu")
    losses = []
    for _ in range(200):
        y = rng.integers(0, 5, 16).astype(np.int32)
        x = centers[y] + rng.standard_normal((16, 20)).astype(np.float32)
        losses.append(step(x=x, y=y))
    assert losses[0] > 1.0 and np.mean(losses[-20:]) < 0.1
