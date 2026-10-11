# MiniDLC: A Mini Deep-Learning Compiler

MiniDLC is an end-to-end deep-learning compiler built from scratch, lowering models written in a custom Python frontend through a two-level intermediate representation (Graph IR and Loop IR) to native machine code via **LLVM (CPU)** and **CUDA (GPU)**.

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

## Key Features

- **Custom Python Frontend:** Trace neural network operations without external deep learning frameworks. Standard PyTorch-like API with `Tensor`, `nn.Module`, `nn.Linear`, `nn.LayerNorm`, and optimizers (`SGD`, `Adam`).
- **Two-Level Intermediate Representation:**
  - **Graph IR:** Pure SSA dataflow graph with static shape typing and verification.
  - **Loop IR:** Expresses iteration spaces, loop bounds, memory buffers, affine index access patterns, and vector/parallel annotations.
- **Graph Transformation Passes:**
  - Forward shape inference and broadcasting.
  - Canonicalization and constant folding.
  - Common subexpression elimination (CSE) via value numbering.
  - Dead code elimination (DCE) via reverse reachability.
  - Reverse-mode automatic differentiation (autodiff) appending VJP adjoints.
  - Operator fusion (elementwise chains and matmul epilogues).
  - Liveness-based memory planning for arena buffer reuse.
- **Loop Optimizations:**
  - Cache and shared-memory tiling.
  - Loop unrolling.
  - SIMD vectorization (AVX2 / 8 floats).
  - Multi-threaded parallelization.
  - Range analysis and bounds-check elimination.
- **Execution Backends:**
  - Reference Python/NumPy interpreter for validation and float64 gradient checks.
  - LLVM ORC JIT for multi-threaded CPU code generation.
  - CUDA backend with NVRTC runtime compilation.

---

## Repository Structure

```
minidlc/
├── CMakeLists.txt          # Root build configuration
├── pyproject.toml          # Python package specification
├── README.md               # Project documentation
├── LICENSE                 # License
├── docs/                   # Specifications, pass documentation, course map
│   ├── architecture.md
│   ├── ir-spec.md
│   ├── course-map.md
│   ├── perf-report.md
│   ├── devlog.md
│   └── passes/
├── include/dlc/            # C++ public headers
│   ├── ir/                 # Graph IR: types, ops, builder, verifier, printer, parser
│   ├── passes/             # Graph passes: shape-infer, cse, dce, autodiff, fusion, mem-plan
│   ├── loopir/             # Loop IR, lowering, tiling, vectorization
│   ├── backends/           # LLVM CPU JIT and CUDA NVRTC backends
│   └── runtime/            # Arena allocator, thread pool, execution plan
├── lib/                    # C++ implementations
├── tools/dlc-opt/          # Command-line compiler driver tool
├── python/                 # Python frontend package & nanobind bindings
│   ├── bindings/
│   └── minidlc/
├── tests/                  # Test suites
│   ├── lit/                # Golden IR tests with FileCheck
│   ├── unit/               # C++ unit tests
│   └── python/             # Frontend, reference, and gradient checks
├── examples/               # End-to-end models (MNIST MLP, Char-GPT)
├── benchmarks/             # Microbenchmarks (stream bandwidth, roofline)
└── docker/                 # Build environments
```

---

## Quickstart & Usage

### 1. Building the C++ Core and Tools

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

This builds the `dlc-opt` compiler tool in `build/tools/dlc-opt/dlc-opt`.

### 2. Inspecting IR Passes with `dlc-opt`

```bash
./build/tools/dlc-opt/dlc-opt tests/lit/basic_relu_matmul.dlc --passes=shape-infer,canonicalize,cse,dce --print-after-all
```

### 3. Using the Python Frontend

```python
import minidlc as dl
from minidlc import nn, optim

class MLP(nn.Module):
    def __init__(self):
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        return self.fc2(dl.relu(self.fc1(x)))

model = MLP()
x = dl.placeholder((128, 784), dtype="f32")
y = dl.placeholder((128,), dtype="i32")
loss = dl.cross_entropy(model(x), y)

# Compile training step
train_step = dl.compile(
    loss,
    optimizer=optim.Adam(model.parameters(), lr=1e-3),
    target="reference"  # or "cpu", "cuda"
)

# Run a step
for xb, yb in dl.data.synthetic_mnist_batches(batch_size=128, steps=5):
    l = train_step(x=xb, y=yb)
    print(f"Step loss: {l:.4f}")
```

### 4. Running Tests

```bash
# Python frontend and reference interpreter tests
PYTHONPATH=python pytest tests/python

# C++ unit tests
ctest --test-dir build --output-on-failure
```
