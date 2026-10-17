# MiniDLC: A Mini Deep-Learning Compiler

[![Build & Tests](https://img.shields.io/badge/tests-18%20unit%20%7C%2011%20lit%20%7C%207%20transformer-brightgreen)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Linux%20(x86__64)-lightgrey)]()

MiniDLC is an end-to-end, learning-focused deep-learning compiler built strictly from scratch in **C++20** and **Python**. It lowers neural network architectures written in a custom Python frontend through a two-level intermediate representation (**Graph IR** and **Loop IR**) to native machine code via **LLVM (CPU)** and **CUDA (GPU)**.

```
 ┌──────────────┐   trace   ┌──────────────┐  graph passes  ┌──────────────┐  lower  ┌──────────────┐ loop passes ┌────────────────────┐
 │ Python front │ ────────► │ Graph IR     │ ─────────────► │ Graph IR     │ ──────► │ Loop IR      │ ──────────► │ Backends           │
 │ end (tensors,│           │ (SSA, C++)   │ shape, CSE,DCE │ + backward   │         │ (loop nests, │ tile,unroll │ reference (NumPy)  │
 │ nn, optim)   │           │              │ autodiff,fuse, │ + optimizer  │         │ buffers)     │ vectorize,  │ LLVM → CPU (JIT)   │
 └──────────────┘           └──────────────┘ mem-plan       └──────────────┘         └──────────────┘ bounds-elim │ CUDA → GPU (NVRTC) │
                                                                                                                 └────────┬───────────┘
                                                                                     runtime: arena · thread pool · kernel plan · timers
```

---

## Key Highlights & Measured Results

All figures below are empirically measured and documented in [`docs/perf-report.md`](docs/perf-report.md):

- **End-to-End MNIST Training**: Trains a $784 \to 256 \to 10$ MLP to **96.88% test accuracy** in **34.0 seconds** on CPU via LLVM JIT code generation (**$2.27\times$ faster** than the reference interpreter).
- **Epilogue Operator Fusion**: Fuses bias addition and activation directly into the matrix multiplication accumulation loop, achieving a **$1.39\times$ DRAM memory traffic reduction** ($1,860.6\text{ KB} \to 1,336.3\text{ KB}$) and a **$2.77\times$ kernel speedup**.
- **Liveness Memory Planning**: Allocates tensors into a contiguous arena using liveness interval coloring, saving **26.2% peak memory** ($1.64\text{ MB} \to 1.21\text{ MB}$).
- **Static Range Analysis & Loop Versioning (`CS6843 A2/A4`)**: Statically proves affine memory indices remain within bounds, eliminating runtime guards on tile multiples and versioning non-multiples into vectorized main loops and scalar tail loops.
- **Autoregressive Character Transformer**: Trains a 30,656-parameter Transformer on TinyShakespeare ($1.115\text{M}$ characters) via `dl.compile`, dropping cross-entropy loss from **4.33 to 3.15** in **0.78 seconds**.
- **CUDA GPU Acceleration (`CS6023`)**: Shared-memory 2D tiled GEMM ($16 \times 16$), on-chip transposition, and 5-step warp-shuffle reduction trees (`__shfl_down_sync`) compiled via NVRTC.

---

## Repository Architecture

```
mini-dlc/
├── CMakeLists.txt          # Root C++ build configuration
├── pyproject.toml          # Python package specification
├── README.md               # Quickstart and overview
├── LICENSE                 # MIT License
├── docs/                   # Specifications, pass documentation, course map
│   ├── architecture.md     # Compiler architecture & pipeline specification
│   ├── ir-spec.md          # Formal Graph IR and Loop IR syntax & semantics
│   ├── course-map.md       # Detailed academic course mapping (CS6843, CS6023, CS6600, ID5130)
│   ├── perf-report.md      # Empirical benchmarks, FMA peak, STREAM bandwidth & rooflines
│   ├── blog-post.md        # Technical retrospective on building MiniDLC
│   ├── resume-entry.md     # Recruiter-legible resume entries with measured numbers
│   ├── devlog.md           # Milestone development log
│   └── passes/             # Per-pass documentation (algorithms, complexity, course links)
├── include/dlc/            # C++ public headers
│   ├── ir/                 # Graph IR: types, ops, builder, verifier, printer, parser
│   ├── passes/             # Graph passes: shape-infer, cse, dce, autodiff, fusion, mem-plan
│   ├── loopir/             # Loop IR, lowering, tiling, vectorization, bounds-check elimination
│   ├── backends/           # LLVM CPU JIT and CUDA NVRTC codegen
│   └── runtime/            # Arena allocator, thread pool, execution plan
├── lib/                    # C++ implementations
├── tools/dlc-opt/          # Command-line compiler driver tool
├── python/minidlc/         # Python frontend package, ops, nn, optim, data loaders
├── tests/                  # Test suites
│   ├── lit/                # 11 Lit golden IR tests with FileCheck
│   ├── unit/               # 18 C++ unit tests
│   └── python/             # Gradient checks, reference tests, transformer tests, contract suite, DAG fuzzer
├── examples/               # End-to-end models (MNIST MLP, Char-GPT)
└── benchmarks/             # Peak FMA compute, STREAM memory bandwidth, rooflines
```

---

## 10-Minute Quickstart

### 1. Build the C++ Core and Tools

```bash
# Configure and build in Release mode
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
```

This builds the `dlc-opt` compiler tool at `./build/dlc-opt` and the C++ unit test suite at `./build/dlc_unit_tests`.

### 2. Inspect Passes with `dlc-opt`

```bash
# Print IR after shape inference, canonicalization, CSE, and DCE
./build/dlc-opt tests/lit/basic_relu_matmul.dlc --passes=shape-infer,canonicalize,cse,dce --print-after-all

# Emit lowered and vectorized Loop IR with bounds-check elimination
./build/dlc-opt tests/lit/fusion.dlc --passes=shape-infer,fusion --emit-loopir

# Emit generated LLVM IR (CPU)
./build/dlc-opt tests/lit/basic_relu_matmul.dlc --passes=shape-infer,fusion --emit-llvm

# Emit generated CUDA C++ (GPU)
./build/dlc-opt tests/lit/cuda_tiled_matmul.dlc --emit-cuda
```

### 3. Train MNIST MLP on CPU (LLVM JIT)

```bash
# Train an MLP (784 -> 256 -> 10) for 2 epochs on the MNIST dataset (~34 seconds)
PYTHONPATH=python python3 examples/mnist_mlp.py 2
```

### 4. Train Char-GPT on TinyShakespeare

```bash
# Train an autoregressive Transformer and sample Shakespeare text (~0.8 seconds)
PYTHONPATH=python python3 examples/char_gpt.py
```

### 5. Run the Automated Test Suites

```bash
# 1. C++ Unit Tests (18/18 passing)
./build/dlc_unit_tests

# 2. Lit Golden IR Tests (11/11 passing)
python3 tests/lit/run_lit_tests.py

# 3. Analytic Gradient VJP Checks vs Float64 Finite Differences
PYTHONPATH=python python3 tests/python/test_gradcheck.py

# 4. Transformer & Char-GPT Test Suite (7/7 passing)
PYTHONPATH=python python3 tests/python/test_transformer.py

# 5. Rigorous Contract & Corner-Case Stress Tests (9/9 passing)
PYTHONPATH=python python3 tests/python/test_contracts.py

# 6. Randomized Computational DAG Fuzzer vs Float64 Finite Differences (25/25 passing)
PYTHONPATH=python python3 tests/python/test_fuzzer.py
```

---

## Academic Course Mapping

MiniDLC directly applies principles from graduate computer science coursework at IIT Madras:

- **CS6843 Program Analysis**: SSA dominance, dataflow frameworks (Available Expressions CSE, Liveness DCE), interval range analysis and bounds-check elimination with loop versioning (Assignments A2/A4), Chaitin-style interval interference memory planning.
- **CS6023 GPU Programming**: 2D shared-memory tiling ($16 \times 16$), on-chip transposition, 100% coalesced global memory access, 5-step intra-warp reduction trees (`__shfl_down_sync`), hardware atomic scatter (`atomicAdd`).
- **CS6600 / CS6400 Computer Architecture**: Cache blocking (L1/L2), 256-bit AVX2 SIMD vectorization, instruction-level parallelism (loop unrolling factor 4), Williams et al. Roofline model.
- **ID5130 Parallel Scientific Computing**: Multi-threaded thread-pool task scheduling, numerical stability (log-sum-exp, fp32 accumulation), IEEE-754 / PyTorch NaN propagation.

For a full breakdown of course assignments and repo locations, see [`docs/course-map.md`](docs/course-map.md).

---

## License

MiniDLC is open-source software licensed under the [MIT License](LICENSE).
