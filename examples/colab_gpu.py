#!/usr/bin/env python3
"""Runs the committed GPU artifacts on an NVIDIA GPU (e.g. a Colab T4) and reports what happened.

Needs NumPy and the CUDA driver; not LLVM, not the compiled module. For each model it
  1. checks that the artifact's initial weights are the ones the model definition (examples/gpu_models.py) produces,
  2. runs a few training steps on the GPU and in the NumPy reference interpreter from the same weights and batches,
     and compares the losses,
  3. trains (MNIST: whole epochs; Char-GPT: N steps) and times it,
  4. evaluates (MNIST: test accuracy; Char-GPT held-out loss) with the evaluation artifact and the trained weights,
  5. profiles the kernels of a few steps with CUDA events.
Then the baselines, if PyTorch with CUDA is installed (it is on Colab): the same training steps in PyTorch on the same
GPU, and MiniDLC's GEMM kernels against cuBLAS (torch.matmul). Training steps are timed by the wall clock with the
loss fetched every step (as a training loop does), kernels with CUDA events.
The results go to stdout and to a JSON file (--json).

    git clone <repo> && cd MiniDLC
    PYTHONPATH=python python3 examples/colab_gpu.py [mnist|chargpt|baselines|all] [--epochs=2] [--steps=2000]
        [--json=gpu_run.json] [--artifacts=artifacts/gpu] [--checkpoint=DIR]

--checkpoint=DIR saves each model's parameters and Adam state after training (DIR/<model>.npz), loadable by any
target (minidlc/checkpoint.py).

--sim runs a few steps of everything on the simulated device (the lockstep GPU interpreter; needs the compiled module):
a smoke test of this script without a GPU.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import minidlc as dl  # noqa: E402
from minidlc import gpu  # noqa: E402
from minidlc.data import load_mnist, load_tinyshakespeare, mnist_batches  # noqa: E402

import gpu_models  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarks"))

ARTIFACTS = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--artifacts=")),
                 os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "artifacts", "gpu"))
SIM = "--sim" in sys.argv  # a few steps of everything, on the simulated device
CHECKPOINTS = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--checkpoint=")), None)
T4 = {"fp32_tflops": 8.1, "dram_gbps": 320.0}  # datasheet ceilings, for scale
TORCH_DEVICE = "cuda"  # where the PyTorch baselines run


def save_checkpoint(step, name):
    if CHECKPOINTS:
        os.makedirs(CHECKPOINTS, exist_ok=True)
        path = os.path.join(CHECKPOINTS, name + ".npz")
        step.save_checkpoint(path)
        print(f"  checkpoint: {path}")


def check_initial_weights(artifact, model):
    params = model.parameters()
    for arg, i in artifact.params.items():
        if not np.array_equal(artifact.values[arg], params[i].data.astype(np.float32)):
            raise SystemExit(f"artifact weight {arg} differs from the model definition: rebuild the artifacts")


def compare_with_reference(name, device, loss, optimizer, batches):
    """Losses of the first steps on the GPU and in the reference interpreter, from the same weights."""
    reference = dl.compile(loss, optimizer=optimizer, target="reference")
    want = [reference(**b) for b in batches]
    step = gpu.GpuFunction(gpu.GpuArtifact.load(os.path.join(ARTIFACTS, name)), device)
    got = [step(**b) for b in batches]
    worst = max(abs(g - w) / max(abs(w), 1e-6) for g, w in zip(got, want))
    print(f"  first {len(batches)} steps, GPU vs reference: {' '.join(f'{g:.5f}/{w:.5f}' for g, w in zip(got, want))}")
    print(f"  largest relative difference {worst:.2e}")
    return {"gpu": got, "reference": want, "max_rel_diff": worst}


def profile(step, run_steps, top=8):
    """Per-kernel times of a few steps (CUDA events on a GPU), with the sum: the GPU's share of a step."""
    step.executable.set_profiling(True)
    step.executable.clear_profile()
    run_steps()
    step.executable.set_profiling(False)
    rows = sorted(step.executable.profile(), key=lambda p: -p["seconds"])
    total = sum(p["seconds"] for p in rows) or 1.0
    steps = max(p["calls"] for p in rows) if rows else 1
    print(f"  kernel time per step (sum over {len(rows)} kernels): {1e3 * total / steps:.3f} ms")
    print(f"  {'kernel':40s} {'ms/step':>8s} {'share':>6s} {'GFLOP/s':>8s} {'GB/s':>7s}")
    out = {"kernel_ms_per_step": 1e3 * total / steps, "kernels": []}
    for p in rows[:top]:
        per = p["seconds"] / p["calls"]
        print(f"  {p['name'][:40]:40s} {1e3 * per:8.3f} {100 * p['seconds'] / total:5.1f}% "
              f"{p['flops'] / per / 1e9:8.1f} {p['bytes'] / per / 1e9:7.1f}")
        out["kernels"].append({"name": p["name"], "ms": 1e3 * per, "gflops": p["flops"] / per / 1e9,
                               "gbps": p["bytes"] / per / 1e9})
    return out


def run_mnist(device, epochs):
    print("\n== MNIST MLP (784-256-10, batch 128, Adam) ==")
    tr_x, tr_y, ts_x, ts_y = load_mnist()
    model, loss, optimizer, _ = gpu_models.mnist_graphs()
    train_artifact = gpu.GpuArtifact.load(os.path.join(ARTIFACTS, "mnist_train"))
    check_initial_weights(train_artifact, model)
    rng = np.random.default_rng(0)
    first = [{"x": tr_x[i:i + 128], "y": tr_y[i:i + 128].astype(np.int32)} for i in rng.integers(0, 50000, 2 if SIM else 5)]
    result = {"check": compare_with_reference("mnist_train", device, loss, optimizer, first)}

    step = gpu.GpuFunction(train_artifact, device)
    np.random.seed(0)
    epoch_times = []
    for ep in range(epochs):
        t0 = time.perf_counter()
        for b, (xb, yb) in enumerate(mnist_batches(tr_x, tr_y, batch_size=128, shuffle=True)):
            last = step(x=xb, y=yb)
            if SIM and b == 2:
                break
        epoch_times.append(time.perf_counter() - t0)
        print(f"  epoch {ep + 1}: loss {last:.4f}, {epoch_times[-1]:.2f} s ({1e3 * epoch_times[-1] / (len(tr_x) // 128):.3f} ms/step)")

    evaluate = gpu.GpuFunction(gpu.GpuArtifact.load(os.path.join(ARTIFACTS, "mnist_eval")), device)
    evaluate.set_params(step.get_params())
    correct = sum(int(np.sum(np.argmax(evaluate(x=ts_x[i:i + 1000]), axis=-1) == ts_y[i:i + 1000]))
                  for i in range(0, 1000 if SIM else len(ts_x), 1000))
    accuracy = 100.0 * correct / (1000 if SIM else len(ts_x))
    print(f"  test accuracy {accuracy:.2f}%")
    save_checkpoint(step, "mnist")
    batches = list(mnist_batches(tr_x, tr_y, batch_size=128, shuffle=False))[:2 if SIM else 50]
    xb0, yb0 = batches[0]
    ms = wall_ms(lambda: step(x=xb0, y=yb0), 2 if SIM else 200, warmup=1 if SIM else 10)
    print(f"  {ms:.3f} ms/step (wall clock, loss fetched every step)")
    result.update(epoch_seconds=epoch_times, test_accuracy=accuracy, ms_per_step=ms,
                  profile=profile(step, lambda: [step(x=xb, y=yb) for xb, yb in batches]))
    return result


def run_chargpt(device, steps):
    c = gpu_models.CHARGPT
    print(f"\n== Char-GPT (d={c['d_model']}, {c['heads']} heads, {c['layers']} layers, context {c['block']}, "
          f"batch {c['batch']}, Adam) ==")
    text, tokenizer = load_tinyshakespeare()
    data = tokenizer.encode(text)
    split = int(0.9 * len(data))
    train, val = data[:split], data[split:]
    counts = np.ones((tokenizer.vocab_size, tokenizer.vocab_size))
    np.add.at(counts, (train[:-1], train[1:]), 1)
    bigram = float(-np.log((counts / counts.sum(axis=1, keepdims=True))[val[:-1], val[1:]]).mean())
    print(f"  held-out bigram baseline {bigram:.4f} nats")
    model, loss, optimizer, _ = gpu_models.chargpt_graphs(tokenizer.vocab_size)
    train_artifact = gpu.GpuArtifact.load(os.path.join(ARTIFACTS, "chargpt_train"))
    check_initial_weights(train_artifact, model)

    def batch(source, rng):
        ix = rng.integers(0, len(source) - c["block"] - 1, c["batch"])
        return {"x": np.stack([source[i:i + c["block"]] for i in ix]).astype(np.int32),
                "y": np.stack([source[i + 1:i + 1 + c["block"]] for i in ix]).astype(np.int32)}

    rng = np.random.default_rng(0)
    result = {"check": compare_with_reference("chargpt_train", device, loss, optimizer, [batch(train, rng) for _ in range(3)])}
    step = gpu.GpuFunction(train_artifact, device)
    evaluate = gpu.GpuFunction(gpu.GpuArtifact.load(os.path.join(ARTIFACTS, "chargpt_eval")), device)

    def held_out(n=2 if SIM else 40):
        evaluate.set_params(step.get_params())
        r = np.random.default_rng(123)
        return float(np.mean([evaluate(**batch(val, r)) for _ in range(n)]))

    curve = [(0, held_out())]
    print(f"  step     0  held-out loss {curve[-1][1]:.4f}")
    t0 = time.perf_counter()
    train_seconds = 0.0
    for s in range(1, steps + 1):
        b = batch(train, rng)
        t = time.perf_counter()
        last = step(**b)
        train_seconds += time.perf_counter() - t
        if s % max(1, steps // 10) == 0 or s == steps:
            curve.append((s, held_out()))
            print(f"  step {s:5d}  train loss {last:.4f}  held-out loss {curve[-1][1]:.4f}  ({time.perf_counter() - t0:.0f} s)")
    print(f"  {1e3 * train_seconds / steps:.2f} ms/step")
    fixed = [batch(train, rng) for _ in range(2 if SIM else 20)]
    print(f"  final held-out loss {curve[-1][1]:.4f} ({curve[-1][1] - bigram:+.4f} vs bigram)")
    save_checkpoint(step, "chargpt")
    result.update(held_out=curve, bigram=bigram, ms_per_step=1e3 * train_seconds / steps,
                  profile=profile(step, lambda: [step(**b) for b in fixed]))
    return result


# --- baselines -------------------------------------------------------------------------------------------------------

def wall_ms(fn, steps, warmup=3):
    """Wall-clock milliseconds per call (each call ends by fetching a result, as a training loop does)."""
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(steps):
        fn()
    return 1e3 * (time.perf_counter() - t0) / steps


def torch_kernel_ms(torch, fn, steps, warmup=3):
    """GPU milliseconds per call of `fn` (no host synchronization inside), with CUDA events around the run."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(steps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / steps


def our_kernel_ms(function, feeds, steps):
    """Event-timed milliseconds per call of each of the artifact's kernels, summed."""
    function.executable.set_profiling(True)
    function.executable.clear_profile()
    for _ in range(steps):
        function(**feeds)
    function.executable.set_profiling(False)
    prof = function.executable.profile()
    return sum(p["seconds"] / p["calls"] for p in prof) * 1e3


def run_baselines(device, results):
    print("\n== Baselines: PyTorch on the same GPU, cuBLAS ==")
    try:
        import torch
    except ImportError:
        print("  PyTorch is not installed: no baselines")
        return {"skipped": "no PyTorch"}
    if SIM or not torch.cuda.is_available():
        print("  PyTorch has no CUDA device here: no baselines")
        return {"skipped": "no CUDA device for PyTorch"}
    cuda = torch.device(TORCH_DEVICE)
    out = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "gemm": {}, "steps": {}}
    reps = 50

    # GEMMs: MiniDLC's kernel (one launch, epilogue fused) against cuBLAS, and PyTorch's relu(a @ b + bias).
    for name, g in gpu_models.GEMMS.items():
        flops = 2.0 * g["m"] * g["k"] * g["n"]
        rng = np.random.default_rng(0)
        feeds = {"a": rng.standard_normal((g["m"], g["k"])).astype(np.float32),
                 "b": rng.standard_normal((g["k"], g["n"])).astype(np.float32)}
        if g["epilogue"]:
            feeds["bias"] = rng.standard_normal(g["n"]).astype(np.float32)
        ours = gpu.GpuFunction(gpu.GpuArtifact.load(os.path.join(ARTIFACTS, name)), device)
        got = ours(**feeds)
        want = feeds["a"] @ feeds["b"] + (feeds["bias"] if g["epilogue"] else 0)
        want = np.maximum(want, 0) if g["epilogue"] else want
        err = float(np.max(np.abs(got - want)) / (np.max(np.abs(want)) or 1.0))
        a, b = (torch.tensor(feeds[k], device=cuda) for k in ("a", "b"))
        rows = {"MiniDLC kernel": our_kernel_ms(ours, feeds, reps), "cuBLAS a @ b": torch_kernel_ms(torch, lambda: a @ b, reps)}
        if g["epilogue"]:
            bias = torch.tensor(feeds["bias"], device=cuda)
            rows["PyTorch relu(a @ b + bias)"] = torch_kernel_ms(torch, lambda: torch.relu(a @ b + bias), reps)
        print(f"  {name}: {g['m']}x{g['k']} @ {g['k']}x{g['n']}{' + bias + ReLU' if g['epilogue'] else ''} "
              f"({flops / 1e6:.1f} MFLOP; MiniDLC max relative error {err:.1e})")
        for label, ms in rows.items():
            print(f"    {label:30s} {ms:8.4f} ms  {flops / (ms * 1e-3) / 1e9:8.1f} GFLOP/s "
                  f"({100 * flops / (ms * 1e-3) / (T4['fp32_tflops'] * 1e12):.1f}% of the T4's FP32 peak)")
        out["gemm"][name] = {"mflop": flops / 1e6, "max_rel_error": err, "ms": rows}

    # Training steps in PyTorch from the same initial weights, batch shapes and optimizer settings.
    model, _, _, _ = gpu_models.mnist_graphs()
    w1, b1, w2, b2 = (torch.tensor(p.data, device=cuda, requires_grad=True) for p in model.parameters())
    adam = torch.optim.Adam([w1, b1, w2, b2], lr=gpu_models.MNIST["lr"])
    rng = np.random.default_rng(0)
    x = torch.tensor(rng.standard_normal((128, 784)).astype(np.float32), device=cuda)
    y = torch.tensor(rng.integers(0, 10, 128), device=cuda)

    def mnist_step(sync=True):
        adam.zero_grad()
        loss = torch.nn.functional.cross_entropy(torch.relu(x @ w1 + b1) @ w2 + b2, y)
        loss.backward()
        adam.step()
        return loss.item() if sync else loss

    mn = {"PyTorch wall ms": wall_ms(mnist_step, reps),
          "PyTorch GPU ms": torch_kernel_ms(torch, lambda: mnist_step(sync=False), reps)}

    from torch_chargpt import TorchCharGPT
    from minidlc.data import load_tinyshakespeare
    _, tokenizer = load_tinyshakespeare()
    cmodel, _, _, _ = gpu_models.chargpt_graphs(tokenizer.vocab_size)
    ref = TorchCharGPT(cmodel, device=TORCH_DEVICE)
    cadam = torch.optim.Adam(list(ref.params.values()), lr=gpu_models.CHARGPT["lr"])
    c = gpu_models.CHARGPT
    cx, cy = (rng.integers(0, tokenizer.vocab_size, (c["batch"], c["block"])).astype(np.int64) for _ in range(2))
    tx, ty = torch.tensor(cx, device=cuda), torch.tensor(cy, device=cuda)

    def chargpt_step(sync=True):
        cadam.zero_grad()
        loss = torch.nn.functional.cross_entropy(ref.logits(tx).reshape(-1, tokenizer.vocab_size), ty.reshape(-1))
        loss.backward()
        cadam.step()
        return loss.item() if sync else loss

    cg = {"PyTorch wall ms": wall_ms(chargpt_step, reps),
          "PyTorch GPU ms": torch_kernel_ms(torch, lambda: chargpt_step(sync=False), reps)}
    for key, row in (("mnist", mn), ("chargpt", cg)):
        if key in results:
            row["MiniDLC wall ms"] = results[key].get("ms_per_step")
            row["MiniDLC kernel ms"] = results[key].get("profile", {}).get("kernel_ms_per_step")
        out["steps"][key] = row
        print(f"  {key} training step: " + ", ".join(f"{k} {v:.3f}" for k, v in row.items() if v is not None))
    return out


def main():
    which = next((a for a in sys.argv[1:] if not a.startswith("--")), "all")
    option = lambda key, default: next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith(f"--{key}=")), default)
    device = gpu.SimDevice() if SIM else gpu.CudaDevice()
    print(f"GPU: {device.name} (compute capability {device.compute_capability[0]}.{device.compute_capability[1]})")
    results = {"gpu": device.name, "compute_capability": device.compute_capability}
    if which in ("mnist", "all"):
        results["mnist"] = run_mnist(device, int(option("epochs", 2)))
    if which in ("chargpt", "all"):
        results["chargpt"] = run_chargpt(device, 2 if SIM else int(option("steps", 2000)))
    if which in ("baselines", "all"):
        results["baselines"] = run_baselines(device, results)
    path = option("json", "gpu_run.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nresults written to {path}")


if __name__ == "__main__":
    main()
