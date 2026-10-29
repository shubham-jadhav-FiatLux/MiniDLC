# MiniDLC: a mini deep-learning compiler

[![CI](https://github.com/shubham-jadhav-FiatLux/MiniDLC/actions/workflows/ci.yml/badge.svg)](https://github.com/shubham-jadhav-FiatLux/MiniDLC/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

MiniDLC is a learning project: a deep-learning compiler built from scratch in C++20 and Python. A model written in its
own Python frontend is traced into an SSA graph IR, differentiated, optimized, lowered to a loop IR, and compiled through
LLVM — for x86-64 CPUs and, via LLVM's NVPTX backend, for NVIDIA GPUs.

> **Status (2026-10-29): v2 rebuild, phases P0–P5 built.** A model traced in Python is exported as IR text and compiled
> by the C++ compiler — autodiff, the Adam/SGD update, fusion, buffer assignment, loop IR with CPU schedules, LLVM,
> ORC JIT — into one native training step. MNIST and a character-level Transformer train through the generated code
> and match the NumPy reference. The GPU path (GPU schedules, NVPTX → PTX, a CUDA driver runtime) is built and checked
> on a lockstep GPU interpreter and by ptxas, but **has not yet run on a GPU** ([`docs/gpu.md`](docs/gpu.md)). The v2
> design and the status of each component are in [`docs/architecture.md`](docs/architecture.md); for a tour of one
> training step through every stage, read [`docs/walkthrough.md`](docs/walkthrough.md).

## What works today

- **Compiled training on the CPU.** `dl.compile(loss, optimizer, target="cpu")` builds forward, backward and the update
  as one graph and compiles it in-process. The MNIST MLP (784 → 256 → 10) trains 2 epochs in about 1–2 s on a 4-core
  laptop and reaches 96.8–96.9% test accuracy. Char-GPT (multi-head attention) reaches 1.93 nats held-out after
  2,000 steps, against 1.93 for PyTorch from the same weights and batches and 2.48 for a bigram model
  ([`docs/perf-report.md`](docs/perf-report.md)).
- **The C++ compiler.**
  - An SSA graph IR with an op registry, a round-tripping parser and printer, and a verifier that runs after every
    pass.
  - Graph passes: shape inference, canonicalization, CSE, DCE, activity analysis, reverse-mode autodiff (every VJP
    checked against finite differences), Adam/SGD lowering with in-place updates, fusion with legality checks, a cost
    model, and buffer assignment with a checker.
  - A loop IR with per-kernel-class CPU schedules (4×16 register-tiled matmul, vectorized elementwise and row
    kernels, parallel outer loops), and loop passes: range analysis, bounds checking with guards for data-dependent
    indices, bounds-check elimination, and a dependence check of every parallel and vectorized loop.
  - An LLVM emitter (`IRBuilder`, O3, ORC LLJIT) whose output LLVM vectorizes into `<8 x float>` FMAs.
  - GPU schedules (shared-memory tiled matmul with register tiles, warp-per-row kernels with shuffles, atomics) and
    NVPTX emission to PTX; a lockstep GPU interpreter that rejects divergent barriers and checks every op and both
    training models against the reference (`target="gpu-sim"`).
- **GPU runtime.** `minidlc/gpu.py` runs compiled artifacts (plan JSON + PTX) through the CUDA driver API with
  ctypes; `examples/colab_gpu.py` runs the committed MNIST artifacts on Colab and checks them against the reference.
- **Python frontend and reference interpreter.** Tensors with operator overloading, `nn` modules (Linear, LayerNorm,
  Embedding, causal self-attention, a character-level Transformer), SGD and Adam, MNIST/TinyShakespeare loaders, and a
  NumPy interpreter with its own autodiff — the oracle every compiled result is compared with.
- **Testing.** Translation validation of every graph pass; differential tests of every op through the generated code
  (odd shapes, broadcasting, zero-size dimensions, NaN/Inf); a mutation fuzzer for the parser and passes; ASan/UBSan.

## Known limitations

- **Nothing has run on a GPU yet.** The GPU path is tested on the simulated device and with ptxas; the hardware run
  (`examples/colab_gpu.py` on Colab) is pending, so there are no GPU timings.
- **Static shapes:** a compiled step has fixed shapes, so a new batch size or context length means a new compilation
  (text generation, whose length grows, runs in the reference interpreter).
- `--target=handwritten` keeps the hand-written AVX2 kernels for one fixed MLP as a benchmark baseline only.

## Repository layout

```
MiniDLC/
├── CMakeLists.txt        # C++ build: libraries, dlc-opt, tests (GoogleTest + lit)
├── pyproject.toml        # Python package; pytest configuration
├── .github/workflows/    # CI: sanitizer and release builds, unit and lit tests, pytest
├── docs/                 # walkthrough, architecture (v2 + status), IR spec, GPU path, perf report, course map, devlog
├── include/dlc/, lib/    # C++: ir/, passes/, loopir/, backends/llvm/, runtime/, compiler/ (some v1 files remain, unbuilt)
├── tools/dlc-opt/        # opt-style driver: --passes, --dot, --target=cpu|gpu, --emit-{loopir,llvm,llvm-opt,ptx}, reports
├── python/bindings/      # the nanobind module _minidlc: compile(ir_text, target, options) -> Executable
├── python/minidlc/       # frontend, reference interpreter, IR export/parse/execute (ir.py), GPU runtime (gpu.py)
├── artifacts/gpu/        # compiled GPU artifacts (plan, PTX, initial weights) for examples/colab_gpu.py
├── tests/
│   ├── unit/             # GoogleTest; test_known_bugs.cpp records known defects as disabled tests
│   ├── lit/              # golden IR tests run by lit with LLVM's FileCheck
│   └── python/           # gradient checks, contracts, fuzzer, Transformer tests, translation validation of the passes
├── examples/             # mnist_mlp.py, char_gpt.py, export_ir.py, gpu_artifacts.py, colab_gpu.py
└── benchmarks/           # baselines and ablations, rooflines, Char-GPT held-out evaluation vs PyTorch, memory plans
```

## Quickstart

### Prerequisites

- Linux on x86-64 with AVX2 and FMA
- CMake ≥ 3.20 and a C++20 compiler
- LLVM 23 development files (headers, `LLVMConfig.cmake`, `libLLVM`) and `FileCheck`
- Python ≥ 3.11 with NumPy and nanobind (`pip install numpy nanobind`); `pytest` and `lit` for the tests

### Build and test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=$(which python3)
cmake --build build -j4          # also builds the _minidlc module into python/minidlc/
ctest --test-dir build --output-on-failure   # GoogleTest, lit/FileCheck, and the Python compiler and backend tests
pytest                                       # all Python tests (the compiler ones need build/dlc-opt and _minidlc)
pip install -e .                             # optional: the minidlc package, with the compiled module
```

For a sanitizer build, configure with `-DCMAKE_BUILD_TYPE=Debug -DDLC_SANITIZE=ON -DDLC_BUILD_PYTHON=OFF`
(AddressSanitizer and UndefinedBehaviorSanitizer); CI runs both configurations.

### Train

```bash
PYTHONPATH=python python3 examples/mnist_mlp.py 2                     # compiled training step (target cpu)
PYTHONPATH=python python3 examples/mnist_mlp.py 2 --target=reference  # NumPy interpreter
PYTHONPATH=python python3 examples/char_gpt.py                        # Char-GPT, compiled
PYTHONPATH=python python3 benchmarks/chargpt_heldout.py 2000 --targets=cpu,torch  # held-out loss vs PyTorch
PYTHONPATH=python python3 benchmarks/ablations.py                     # baselines and ablations, repeated timings
```

### Look inside with `dlc-opt`

```bash
PYTHONPATH=python python3 examples/export_ir.py mlp > mlp.dlc          # the traced model as IR text
./build/dlc-opt mlp.dlc --passes="autodiff,adam{lr=0.001},canonicalize,cse,dce" > step.dlc
./build/dlc-opt step.dlc --print-cost          # kernels, FLOPs, bytes, intensity
./build/dlc-opt step.dlc --print-mem-report    # persistent region, transient arena, in-place updates
./build/dlc-opt step.dlc --emit-loopir         # loop IR with the CPU schedules
./build/dlc-opt step.dlc --print-loop-report   # accesses proven in bounds / guarded
./build/dlc-opt step.dlc --emit-llvm-opt       # LLVM IR after O3
./build/dlc-opt step.dlc --target=gpu --emit-loopir   # the GPU schedules
./build/dlc-opt step.dlc --target=gpu --emit-ptx      # PTX for sm_75
```

### On a GPU

See [`docs/gpu.md`](docs/gpu.md): on Colab, `PYTHONPATH=python python3 examples/colab_gpu.py mnist` runs the committed
artifacts with only NumPy and the CUDA driver.

## Roadmap

v2 keeps the Python frontend, makes the textual graph IR the contract with the C++ compiler, adds a GPU-aware loop IR
with per-target schedules, and generates code through one LLVM emitter for x86-64 (ORC JIT) and NVIDIA GPUs (NVPTX →
PTX). Phases: **P0** truthful docs and real tests (done) → **P1** IR and Python↔C++ bridge (done) → **P2** training
graph in C++ (done) → **P3** fusion and buffer assignment (done) → **P4** CPU backend (done) → **P5** GPU backend (built; GPU run
pending) → **P6** Transformer on both
targets → **P7** measurements and write-up. Details: [`docs/architecture.md`](docs/architecture.md).

## Course mapping

MiniDLC applies graduate coursework from IIT Madras: program analysis (CS6843), GPU programming (CS6023), computer
architecture (CS6600/CS6400) and parallel scientific computing (ID5130). [`docs/course-map.md`](docs/course-map.md)
maps each concept to the code and marks what is implemented and what is planned.

## License

MIT — see [LICENSE](LICENSE).
