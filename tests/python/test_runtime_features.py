"""Runtime features: the kernel cache and checkpoints.

- Kernel cache: compiling the same kernels again loads their machine code (CPU) or PTX (GPU) from disk instead of
  running LLVM, with identical results; MINIDLC_CACHE=off disables it; clearing it removes only its own files.
- Checkpoints: parameters and optimizer state saved by one step and loaded by a fresh one continue the same
  trajectory, on every target and across targets (a checkpoint names arrays after the compiled step's arguments).
Needs the compiled _minidlc module; skipped otherwise.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

import minidlc as dl
from minidlc import nn, optim

_minidlc = pytest.importorskip("minidlc._minidlc", reason="the _minidlc module is not built")


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(12, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        return self.fc2(dl.relu(self.fc1(x)))


def build(target, opt="adam", seed=0):
    np.random.seed(seed)
    model = MLP()
    x = dl.placeholder((8, 12), name="x")
    y = dl.placeholder((8,), dtype="i32", name="y")
    optimizer = optim.Adam(model.parameters(), lr=1e-2) if opt == "adam" else optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    return dl.compile(dl.cross_entropy(model(x), y), optimizer, target=target), model


def batches(n, seed=1):
    rng = np.random.default_rng(seed)
    return [{"x": rng.standard_normal((8, 12)).astype(np.float32), "y": rng.integers(0, 4, 8).astype(np.int32)}
            for _ in range(n)]


# --- kernel cache ---

def test_second_compile_comes_from_the_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIDLC_CACHE_DIR", str(tmp_path))
    _minidlc.configure_cache()
    try:
        first, _ = build("cpu")
        second, _ = build("cpu")
        assert not first.dumps["from_cache"] and second.dumps["from_cache"]
        info = _minidlc.cache_info()
        assert info["directory"] == str(tmp_path) and info["entries"] == 1 and info["bytes"] > 0
        for b in batches(3):
            assert first(**b) == second(**b)
        # The PTX of a GPU artifact is cached too, and identical.
        np.random.seed(0)
        model = MLP()
        x = dl.placeholder((8, 12), name="x")
        art1 = dl.compile_gpu(dl.relu(model.fc1(x)))
        art2 = dl.compile_gpu(dl.relu(model.fc1(x)))
        assert art1.ptx == art2.ptx and ".entry" in art1.ptx
        assert _minidlc.cache_info()["entries"] == 2
        # Clearing removes the entries and nothing else.
        (tmp_path / "notes.txt").write_text("not a cache entry")
        _minidlc.clear_cache()
        assert _minidlc.cache_info()["entries"] == 0
        assert (tmp_path / "notes.txt").exists()
    finally:
        monkeypatch.undo()
        _minidlc.configure_cache()


def test_cache_can_be_disabled_and_llvm_dumps_bypass_it(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIDLC_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("MINIDLC_CACHE", "off")
    _minidlc.configure_cache()
    try:
        build("cpu")
        build("cpu")
        assert _minidlc.cache_info()["entries"] == 0
        monkeypatch.delenv("MINIDLC_CACHE")
        _minidlc.configure_cache()
        build("cpu")
        np.random.seed(0)
        model = MLP()
        x = dl.placeholder((8, 12), name="x")
        f = dl.compile(dl.relu(model.fc1(x)), target="cpu", options={"llvm_dumps": True})
        assert "define" in f.dumps["llvm"] and "define" in f.dumps["llvm_opt"] and not f.dumps["from_cache"]
    finally:
        monkeypatch.undo()
        _minidlc.configure_cache()


# --- checkpoints ---

@pytest.mark.parametrize("target", ["cpu", "interp", "gpu-sim"])
@pytest.mark.parametrize("opt", ["adam", "sgd"])
def test_checkpoint_resumes_the_same_trajectory(tmp_path, target, opt):
    data = batches(6)
    step, _ = build(target, opt)
    for b in data[:3]:
        step(**b)
    path = str(tmp_path / "run.npz")
    step.save_checkpoint(path)
    want = [step(**b) for b in data[3:]]

    resumed, model = build(target, opt, seed=123)  # other initial weights: the checkpoint must replace them
    resumed.load_checkpoint(path)
    # The nn.Parameters get the checkpoint's values too (e.g. for evaluating with another compiled function).
    name = next(n for n, t in resumed._params.items() if t is model.fc1.weight)
    saved = np.load(path)
    np.testing.assert_array_equal(model.fc1.weight.data, saved[name])
    got = [resumed(**b) for b in data[3:]]
    if target == "gpu-sim":  # atomics (the loss) may add in another order
        np.testing.assert_allclose(got, want, rtol=1e-5)
    else:
        assert got == want


def test_checkpoints_move_between_targets(tmp_path):
    data = batches(6)
    cpu, _ = build("cpu")
    for b in data[:3]:
        cpu(**b)
    path = str(tmp_path / "cpu.npz")
    cpu.save_checkpoint(path)
    want = [cpu(**b) for b in data[3:]]
    for target in ["reference", "gpu-sim"]:
        other, _ = build(target, seed=7)
        other.load_checkpoint(path)
        np.testing.assert_allclose([other(**b) for b in data[3:]], want, rtol=2e-5, err_msg=target)
    # And back: the reference interpreter's checkpoint resumes on the CPU.
    ref, _ = build("reference")
    for b in data[:3]:
        ref(**b)
    ref.save_checkpoint(str(tmp_path / "ref.npz"))
    expect = [ref(**b) for b in data[3:]]
    cpu2, _ = build("cpu", seed=9)
    cpu2.load_checkpoint(str(tmp_path / "ref.npz"))
    np.testing.assert_allclose([cpu2(**b) for b in data[3:]], expect, rtol=2e-5)


def test_checkpoint_of_another_model_is_rejected(tmp_path):
    step, _ = build("cpu")
    path = str(tmp_path / "mlp.npz")
    step.save_checkpoint(path)
    sgd, _ = build("cpu", opt="sgd")
    with pytest.raises(ValueError, match="does not fit"):
        sgd.load_checkpoint(path)  # Adam state where the step expects SGD's velocities
    np.random.seed(0)
    other = nn.Linear(12, 3)
    x = dl.placeholder((8, 12), name="x")
    y = dl.placeholder((8,), dtype="i32", name="y")
    small = dl.compile(dl.cross_entropy(other(x), y), optim.Adam(other.parameters()), target="cpu")
    with pytest.raises(ValueError, match="does not fit"):
        small.load_checkpoint(path)
    np.save(str(tmp_path / "plain.npy"), np.zeros(3))
    with pytest.raises(Exception):
        step.load_checkpoint(str(tmp_path / "plain.npy"))


def test_gpu_runtime_checkpoints(tmp_path):
    """The GPU runtime (minidlc.gpu, as on Colab), on the simulated device: its checkpoints load on the CPU."""
    from minidlc import gpu
    np.random.seed(0)
    model = MLP()
    x = dl.placeholder((8, 12), name="x")
    y = dl.placeholder((8,), dtype="i32", name="y")
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    artifact = dl.compile_gpu(dl.cross_entropy(model(x), y), optimizer, ptx=False)
    step = gpu.GpuFunction(artifact, gpu.SimDevice())
    data = batches(6)
    for b in data[:3]:
        step(**b)
    path = str(tmp_path / "gpu.npz")
    step.save_checkpoint(path)
    want = [step(**b) for b in data[3:]]
    cpu, _ = build("cpu", seed=5)
    cpu.load_checkpoint(path)
    np.testing.assert_allclose([cpu(**b) for b in data[3:]], want, rtol=2e-5)
    step.load_checkpoint(path)  # and back into the GPU step
    np.testing.assert_allclose([step(**b) for b in data[3:]], want, rtol=1e-5)


# --- CPU artifacts ---

def test_cpu_artifact_saves_and_runs_without_recompiling(tmp_path, monkeypatch):
    from minidlc import cpu
    data = batches(4)
    step, _ = build("cpu")
    want = [step(**b) for b in data]
    np.random.seed(0)
    model = MLP()
    x = dl.placeholder((8, 12), name="x")
    y = dl.placeholder((8,), dtype="i32", name="y")
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    artifact = dl.compile_cpu(dl.cross_entropy(model(x), y), optimizer)
    assert artifact.object[:4] == b"\x7fELF" and artifact.triple and "+" in artifact.features
    artifact.save(str(tmp_path / "step"))
    # Loading needs no compilation: with the kernel cache off and empty, nothing is compiled or cached.
    monkeypatch.setenv("MINIDLC_CACHE", "off")
    _minidlc.configure_cache()
    try:
        before = _minidlc.cache_info()["misses"]
        loaded = cpu.load(str(tmp_path / "step"))
        assert [loaded(**b) for b in data] == want
        assert _minidlc.cache_info()["misses"] == before
        # Checkpoints work as for any target.
        loaded.save_checkpoint(str(tmp_path / "ckpt.npz"))
        other, _ = build("cpu", seed=3)
        other.load_checkpoint(str(tmp_path / "ckpt.npz"))
        np.testing.assert_array_equal(other.executable.get(next(iter(other._params))),
                                      loaded.executable.get(next(iter(other._params))))
    finally:
        monkeypatch.undo()
        _minidlc.configure_cache()


def test_cpu_artifact_for_another_cpu_is_refused(tmp_path):
    from minidlc import cpu
    np.random.seed(0)
    model = MLP()
    x = dl.placeholder((8, 12), name="x")
    artifact = dl.compile_cpu(dl.relu(model.fc1(x)))
    artifact.features += ",+a-feature-no-cpu-has"
    with pytest.raises(ValueError, match="lacks a-feature-no-cpu-has"):
        cpu.CpuFunction(artifact)
    artifact.features = artifact.features.replace(",+a-feature-no-cpu-has", "")
    artifact.triple = "aarch64-unknown-linux-gnu"
    with pytest.raises(ValueError, match="compiled for aarch64"):
        cpu.CpuFunction(artifact)
