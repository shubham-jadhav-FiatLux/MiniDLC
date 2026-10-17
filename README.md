# MiniDLC: a mini deep-learning compiler

[![CI](https://github.com/shubham-jadhav-FiatLux/MiniDLC/actions/workflows/ci.yml/badge.svg)](https://github.com/shubham-jadhav-FiatLux/MiniDLC/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

MiniDLC is a learning project: a deep-learning compiler built from scratch in C++20 and Python. A model written in its
own Python frontend is traced into an SSA graph IR, differentiated, optimized, lowered to a loop IR, and compiled through
LLVM — for x86-64 CPUs and, via LLVM's NVPTX backend, for NVIDIA GPUs.

> **Status (2026-10-17): v1 prototype, being rebuilt as v2.** An audit found that the v1 pieces are not yet connected end
> to end: the Python frontend and the C++ compiler do not talk to each other, and the C++ code generators are text
> templates. This README describes what exists today; the v2 design and the status of each component are in
> [`docs/architecture.md`](docs/architecture.md).

## What works today

- **Python frontend and reference interpreter.** Tensors with operator overloading, `nn` modules (Linear, LayerNorm,
  Embedding, causal self-attention, a character-level Transformer), SGD and Adam, and MNIST/TinyShakespeare loaders. A
  NumPy interpreter executes traced graphs and implements reverse-mode autodiff for 18 ops. Every op's gradient is
  checked against float64 central finite differences, alongside a seeded random-DAG fuzzer and NaN/Inf/overflow
  contract tests.
- **MNIST training.** An MLP (784 → 256 → 10) reaches 96.93% test accuracy on the reference interpreter and 96.88% with
  `--target=cpu`, which runs **hand-written** AVX2 LLVM IR kernels (fused matmul+bias+ReLU, backward GEMMs, Adam)
  compiled at run time with `clang -O3` and called through ctypes.
- **Character-level Transformer.** Trains on TinyShakespeare with the reference interpreter; a seeded 2,000-step run
  reaches 2.10 nats held-out cross-entropy, below the 2.45-nat bigram baseline
  ([`docs/perf-report.md`](docs/perf-report.md)).
- **C++ compiler infrastructure.** An SSA graph IR (builder, printer, parser, verifier, Graphviz export), graph passes
  (shape inference, canonicalization, CSE, DCE, autodiff, fusion grouping, memory planning), a loop IR with interval
  range analysis and remainder-loop peeling, and `dlc-opt`, which prints the IR after each pass.

## Known limitations (being fixed in v2)

- **The frontend and the C++ compiler are not connected.** `dl.compile(..., target="cpu")` runs the hand-written kernels
  for a fixed two-layer ReLU MLP. It does not compile the traced graph, and other models get wrong results on that
  target — use `target="reference"` for anything else.
- **The C++ passes have known bugs** (several autodiff VJPs, constant folding, CSE keys, memory planning with fusion),
  and the LLVM/CUDA emitters in `lib/backends/` pick text templates by kernel name instead of translating the loop IR.
  They run only inside `dlc-opt` and the tests.
- **No code has run on a GPU yet**, so the CUDA emitter's output is unverified.
- Each known bug is recorded as a disabled unit test in [`tests/unit/test_known_bugs.cpp`](tests/unit/test_known_bugs.cpp)
  or an expected-failure lit test, to be enabled by the phase that fixes it.

## Repository layout

```
MiniDLC/
├── CMakeLists.txt        # C++ build: libraries, dlc-opt, tests (GoogleTest + lit)
├── pyproject.toml        # Python package; pytest configuration
├── .github/workflows/    # CI: sanitizer and release builds, unit and lit tests, pytest
├── docs/                 # architecture (v2 + status), IR spec, pass pages, perf report, course map, devlog
├── include/dlc/, lib/    # C++: ir/, passes/, loopir/, backends/ (v1 templates), runtime/
├── tools/dlc-opt/        # opt-style driver: --passes, --print-after-all, --dot, --emit-{loopir,llvm,cuda}
├── python/minidlc/       # frontend, reference interpreter, hand-written LLVM kernels (llvm_engine.py)
├── tests/
│   ├── unit/             # GoogleTest; test_known_bugs.cpp records known defects as disabled tests
│   ├── lit/              # golden IR tests run by lit with LLVM's FileCheck
│   └── python/           # gradient checks, contracts, fuzzer, reference and Transformer tests
├── examples/             # mnist_mlp.py, char_gpt.py
└── benchmarks/           # FMA peak, STREAM, rooflines, fusion ablation, Char-GPT held-out evaluation
```

## Quickstart

### Prerequisites

- Linux on x86-64 with AVX2 and FMA
- CMake ≥ 3.20 and a C++20 compiler
- `clang` on `PATH` (the CPU kernels are compiled at run time)
- Python ≥ 3.11 with NumPy
- For the tests: GoogleTest (system package, or fetched by CMake), LLVM's `FileCheck`, `lit` (`pip install lit`) and
  `pytest`

### Build and test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
ctest --test-dir build --output-on-failure   # 21 GoogleTest unit tests + 11 lit/FileCheck golden tests
pytest                                       # 34 Python tests (pythonpath is set in pyproject.toml)
```

For a sanitizer build, configure with `-DCMAKE_BUILD_TYPE=Debug -DDLC_SANITIZE=ON` (AddressSanitizer and
UndefinedBehaviorSanitizer); CI runs both configurations.

### Inspect passes with `dlc-opt`

```bash
./build/dlc-opt tests/lit/basic_relu_matmul.dlc --passes=shape-infer,canonicalize,cse,dce --print-after-all
./build/dlc-opt tests/lit/fusion.dlc --passes=shape-infer,fusion --emit-loopir
./build/dlc-opt tests/lit/fusion.dlc --passes=shape-infer,fusion --emit-llvm   # v1 text template
./build/dlc-opt tests/lit/cuda_tiled_matmul.dlc --emit-cuda                    # v1 text template, never run
```

### Train

```bash
PYTHONPATH=python python3 examples/mnist_mlp.py 2                     # hand-written LLVM kernels, ~35 s
PYTHONPATH=python python3 examples/mnist_mlp.py 2 --target=reference  # NumPy interpreter
PYTHONPATH=python python3 examples/char_gpt.py                        # 40-step demo on the interpreter
PYTHONPATH=python python3 benchmarks/chargpt_heldout.py               # 2,000 steps + held-out loss, ~2 min
```

MNIST timings depend heavily on the BLAS library your NumPy links; see [`docs/perf-report.md`](docs/perf-report.md).

## Roadmap

v2 keeps the Python frontend, makes the textual graph IR the contract with the C++ compiler, adds a GPU-aware loop IR
with per-target schedules, and generates code through one LLVM emitter for x86-64 (ORC JIT) and NVIDIA GPUs (NVPTX →
PTX). Phases: **P0** truthful docs and real tests (this revision) → **P1** IR and Python↔C++ bridge → **P2** training
graph in C++ → **P3** fusion and buffer assignment → **P4** CPU backend → **P5** GPU backend → **P6** Transformer on both
targets → **P7** measurements and write-up. Details: [`docs/architecture.md`](docs/architecture.md).

## Course mapping

MiniDLC applies graduate coursework from IIT Madras: program analysis (CS6843), GPU programming (CS6023), computer
architecture (CS6600/CS6400) and parallel scientific computing (ID5130). [`docs/course-map.md`](docs/course-map.md)
maps each concept to the code and marks what is implemented and what is planned.

## License

MIT — see [LICENSE](LICENSE).
