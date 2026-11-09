"""The GPU path without a GPU: the CUDA runtime (minidlc.gpu) replaying compiled plans on the simulated device, saved
artifacts, and the PTX — its instructions, and ptxas accepting it when ptxas is installed. A run on a real GPU happens
when libcuda is present (test_cuda_matches_reference), e.g. on Colab (docs/gpu.md).

The GPU schedules themselves are checked op by op in test_cpu_backend.py (target "gpu-sim") and in training in
test_cpu_training.py.
"""
from __future__ import annotations

import ctypes.util
import glob
import json
import os
import time
import shutil
import subprocess
import sys

import numpy as np
import pytest

import minidlc as dl
from minidlc import gpu, nn, optim

pytest.importorskip("minidlc._minidlc", reason="the _minidlc module is not built")


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 70)
        self.fc2 = nn.Linear(70, 5)

    def forward(self, x):
        return self.fc2(dl.relu(self.fc1(x)))


def mlp_step(seed=0):
    np.random.seed(seed)
    model = MLP()
    x = dl.placeholder((19, 20), name="x")
    y = dl.placeholder((19,), dtype="i32", name="y")
    loss = dl.cross_entropy(model(x), y)
    return model, loss


def batches(n, seed=1):
    rng = np.random.default_rng(seed)
    return [{"x": rng.standard_normal((19, 20)).astype(np.float32), "y": rng.integers(0, 5, 19).astype(np.int32)}
            for _ in range(n)]


def losses(step, data):
    return [step(**b) for b in data]


def test_runtime_on_the_simulated_device_follows_eager_training():
    data = batches(5)
    model, loss = mlp_step()
    eager = losses(dl.compile(loss, optimizer=optim.Adam(model.parameters(), lr=0.01), target="reference"), data)
    model, loss = mlp_step()
    artifact = dl.compile_gpu(loss, optimizer=optim.Adam(model.parameters(), lr=0.01), ptx=False)
    step = gpu.GpuFunction(artifact, device=gpu.SimDevice())
    np.testing.assert_allclose(losses(step, data), eager, rtol=2e-4, atol=2e-4)


def test_runtime_state_get_set_and_reset():
    model, loss = mlp_step()
    artifact = dl.compile_gpu(loss, optimizer=optim.SGD(model.parameters(), lr=0.1, momentum=0.9), ptx=False)
    step = gpu.GpuFunction(artifact, device=gpu.SimDevice())
    exe = step.executable
    weight = next(name for name, a in exe.args.items() if a["role"] == "param" and len(a["shape"]) == 2)
    before = exe.get(weight)
    np.testing.assert_array_equal(before, artifact.values[weight])
    data = batches(3)
    first = losses(step, data)
    assert not np.array_equal(exe.get(weight), before)
    # Restoring the parameters and the optimizer state replays the same losses.
    exe.set(weight, before)
    for name, value in artifact.values.items():
        exe.set(name, value)
    exe.reset_state()
    np.testing.assert_allclose(losses(step, data), first, rtol=1e-6)


def test_executables_share_a_device():
    """A training step and an evaluation function on one device keep their own kernels."""
    np.random.seed(0)
    model = MLP()
    x = dl.placeholder((19, 20), name="x")
    y = dl.placeholder((19,), dtype="i32", name="y")
    train = dl.compile_gpu(dl.cross_entropy(model(x), y), optim.SGD(model.parameters(), lr=0.1), ptx=False)
    evaluate = dl.compile_gpu(model(x), ptx=False, parameters=model.parameters())
    device = gpu.SimDevice()
    step, forward = gpu.GpuFunction(train, device), gpu.GpuFunction(evaluate, device)
    data = batches(3)
    losses(step, data)
    forward.set_params(step.get_params())
    logits = forward(x=data[0]["x"])
    assert logits.shape == (19, 5)
    # The evaluation function sees the trained weights; the step still runs its own kernels.
    trained = {i: p for i, p in step.get_params().items()}
    for p, value in zip(model.parameters(), [trained[i] for i in range(len(trained))]):
        p.data[...] = value
    root = model(x)
    want = dl.ReferenceInterpreter().forward(root, {x: data[0]["x"]})[root]
    np.testing.assert_allclose(logits, want, rtol=1e-4, atol=1e-4)
    assert np.isfinite(step(**data[1]))


def test_saved_artifacts_round_trip(tmp_path):
    model, loss = mlp_step()
    artifact = dl.compile_gpu(loss, optimizer=optim.Adam(model.parameters(), lr=0.01))
    artifact.save(str(tmp_path))
    assert sorted(os.listdir(tmp_path)) == ["kernels.ptx", "meta.json", "plan.json", "source.dlc", "values.npz"]
    loaded = gpu.GpuArtifact.load(str(tmp_path))
    assert loaded.plan == artifact.plan and loaded.ptx == artifact.ptx and loaded.inputs == artifact.inputs
    assert loaded.training and loaded.kernel_names == artifact.kernel_names
    for name, value in artifact.values.items():
        np.testing.assert_array_equal(loaded.values[name], value)
    # The loaded artifact behaves like the original (the simulated device recompiles its source).
    data = batches(2)
    a = losses(gpu.GpuFunction(artifact, device=gpu.SimDevice()), data)
    b = losses(gpu.GpuFunction(loaded, device=gpu.SimDevice()), data)
    assert a == b


class EventClockDevice(gpu.SimDevice):
    """The simulated device with CUDA-event semantics on the host clock: exercises the runtime's event path."""
    has_events = True

    def __init__(self):
        super().__init__()
        self.created = 0

    def event(self):
        self.created += 1
        return [0.0]

    def record(self, event):
        event[0] = time.perf_counter()

    def elapsed_ms(self, start, end):
        return 1e3 * (end[0] - start[0])

    def destroy_event(self, event):
        pass


def test_profiling_with_events_times_every_launch():
    model, loss = mlp_step()
    artifact = dl.compile_gpu(loss, optimizer=optim.Adam(model.parameters(), lr=0.01), ptx=False)
    device = EventClockDevice()
    step = gpu.GpuFunction(artifact, device=device)
    step.executable.set_profiling(True)
    for b in batches(3):
        step(**b)
    prof = {p["name"]: p for p in step.executable.profile()}
    assert set(prof) == {l["name"] for l in artifact.plan["launches"]}
    assert all(p["calls"] == 3 and p["seconds"] > 0 for p in prof.values())
    assert device.created == 2 * len(artifact.plan["launches"])  # one pair per launch, reused every step
    step.executable.clear_profile()
    assert step.executable.profile() == []


COMMITTED = os.path.join(os.path.dirname(__file__), "..", "..", "artifacts", "gpu")


@pytest.mark.parametrize("name", sorted(os.listdir(COMMITTED)) if os.path.isdir(COMMITTED) else [])
def test_committed_artifacts_are_current(name):
    """Each committed artifact's plan and loop IR are what the compiler produces from its source today (else rebuild
    them: examples/gpu_artifacts.py)."""
    from minidlc import _minidlc
    from minidlc.compile import loop_ir_hash
    artifact = gpu.GpuArtifact.load(os.path.join(COMMITTED, name))
    compiled, dumps = _minidlc.compile_gpu(artifact.source, artifact.options, False, artifact.arch)
    assert json.loads(compiled.plan_json) == artifact.plan, "the plan changed: rebuild the artifacts"
    assert loop_ir_hash(dumps["loopir"]) == artifact.loop_ir_sha256, "the kernels changed: rebuild the artifacts"
    for kernel in artifact.kernel_names:
        assert f".entry {kernel}(" in artifact.ptx


def test_every_plan_launch_has_a_ptx_entry():
    model, loss = mlp_step()
    artifact = dl.compile_gpu(loss, optimizer=optim.Adam(model.parameters(), lr=0.01))
    for name in artifact.kernel_names:
        assert f".entry {name}(" in artifact.ptx
    for launch in artifact.plan["launches"]:
        assert all(g >= 1 for g in launch["grid"]) and np.prod(launch["block"]) <= 1024


def test_ptx_uses_shared_memory_barriers_shuffles_and_atomics():
    np.random.seed(0)
    model = nn.CharGPT(vocab_size=11, block_size=8, d_model=16, num_heads=2, num_layers=1)
    x = dl.placeholder((2, 8), dtype="i32", name="x")
    y = dl.placeholder((2, 8), dtype="i32", name="y")
    artifact = dl.compile_gpu(dl.cross_entropy(model(x), y), optimizer=optim.Adam(model.parameters(), lr=0.01))
    ptx = artifact.ptx
    for needle in (".shared .align 16 .b8", "bar.sync", "shfl.sync.bfly.b32", "red.global.add.f32", "ex2.approx",
                   "lg2.approx", "fma.rn.f32", "%tid.x", "%tid.y", "%ctaid.x"):
        assert needle in ptx, needle
    assert ".local" not in ptx, "a register tile spilled to local memory"


def _ptxas():
    candidates = [os.environ.get("PTXAS"), shutil.which("ptxas")]
    candidates += glob.glob(os.path.join(sys.prefix, "lib", "python3*", "site-packages", "nvidia", "cuda_nvcc", "bin",
                                         "ptxas"))
    return next((c for c in candidates if c and os.access(c, os.X_OK)), None)


@pytest.mark.skipif(_ptxas() is None, reason="ptxas not found (set PTXAS, or pip install nvidia-cuda-nvcc-cu12)")
def test_ptxas_accepts_the_ptx(tmp_path):
    np.random.seed(0)
    model = nn.CharGPT(vocab_size=11, block_size=8, d_model=16, num_heads=2, num_layers=1)
    x = dl.placeholder((2, 8), dtype="i32", name="x")
    y = dl.placeholder((2, 8), dtype="i32", name="y")
    artifact = dl.compile_gpu(dl.cross_entropy(model(x), y), optimizer=optim.Adam(model.parameters(), lr=0.01))
    path = tmp_path / "kernels.ptx"
    path.write_text(artifact.ptx)
    out = subprocess.run([_ptxas(), "-arch=sm_75", "-v", str(path), "-o", str(tmp_path / "kernels.cubin")],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "spill" not in out.stderr or "0 bytes spill stores" in out.stderr


def _has_cuda():
    if not ctypes.util.find_library("cuda"):
        return False
    try:
        gpu.CudaDevice()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _has_cuda(), reason="no CUDA driver or GPU")
def test_cuda_matches_reference():
    data = batches(5)
    model, loss = mlp_step()
    eager = losses(dl.compile(loss, optimizer=optim.Adam(model.parameters(), lr=0.01), target="reference"), data)
    model, loss = mlp_step()
    step = dl.compile(loss, optimizer=optim.Adam(model.parameters(), lr=0.01), target="cuda")
    # ex2/lg2 approximations: slightly looser than the CPU's tolerance.
    np.testing.assert_allclose(losses(step, data), eager, rtol=1e-3, atol=1e-3)
