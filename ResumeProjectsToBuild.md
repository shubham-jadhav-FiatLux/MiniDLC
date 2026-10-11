# Resume Projects To Build

> **Purpose.** Each entry in this file is a complete specification for a project I have *decided* to build: what it
> is, why, how it will be built, and what the finished project will look like. It should be detailed enough to start
> building from it later without re-deriving anything.
>
> **Ground rules (apply to every entry):**
> - Built for real, used for learning, using my coursework.
> - No novelty or "beats X" claims unless measured.
>
> **Related files:** `Phase3_Project_Ideas.md` (brainstorm, idea catalog, prior-art sweeps, decision log) and
> `Courses/<Course>/<Course>_Learning.md` (the course notes each project draws on).

## Index

| # | Project | Courses used | Status |
|---|---|---|---|
| 1 | **MiniDLC** — a mini deep-learning compiler: Python → graph IR → autodiff/optimizations → LLVM (CPU) and CUDA (GPU) | CS6843, CS6023, CS6600/CS6400, ID5130, LLVM, math | **Decided 2026-10-09** · not started |


Status legend: **Decided** (spec written, not started) → **In progress (Mx)** → **Done** (definition of done met) →
**Polished** (docs, blog, resume line updated with measured numbers).

---

# Project 1 — MiniDLC: a mini deep-learning compiler, from Python to LLVM and CUDA

*"MiniDLC" is a working name; rename at kickoff if desired.*

## 1.1 At a glance

| | |
|---|---|
| **One-liner** | Write a small neural network in my own Python frontend; my compiler differentiates it, optimizes it, generates machine code via **LLVM (CPU)** and **CUDA (GPU)**, and trains it. |
| **Proof of completeness** | An MLP trains on **MNIST** and a tiny **character-level GPT** trains on **TinyShakespeare**, end to end on my own stack, on both CPU and GPU. |
| **Type** | Learning + portfolio project. Real, useful, complete. **No novelty claims.** |
| **Courses** | CS6843 Program Analysis (IR, dataflow, liveness, range analysis, dependence, LLVM), CS6023 GPU Programming (CUDA tiling, shared memory, warp reductions), CS6600/CS6400 Architecture (caches, SIMD, roofline), ID5130 Parallel Scientific Computing (threads, numerics, performance analysis), math (autodiff, linear algebra) |
| **Key decisions made** | **My own small Python frontend** (no PyTorch import) · C++ core with Python bindings · two-level IR (graph IR → loop IR) · backends: NumPy reference interpreter, LLVM CPU, CUDA GPU · fp32, static shapes, single device |
| **Effort** | ~11–12 weeks part-time (milestones M0–M5, §1.8). After M2 there is already a complete CPU compiler that trains MNIST, a valid stopping point. |
| **Hardware** | Any Linux machine. M0–M2 need only a CPU. M3–M4 need an NVIDIA GPU: a machine that has one (preferred if available), the IITM cluster, or Colab/Kaggle. See §1.7. |
| **Status** | Decided 2026-10-09. Not started. |

## 1.2 Why this project

**Goals**
1. **Learn by building the whole stack once:** frontend → IR → analyses/optimizations → code generation → runtime →
   training. Every layer is something I studied but never assembled end to end.
2. **Use my coursework for real.** Each component maps to a course concept (§1.6), so the project doubles as revision.
3. **Be complete and usable.** It trains real models, has tests and docs, and someone else can clone it and train
   MNIST in minutes.
4. **Be recruiter-legible.** ML-compiler, GPU-kernel, and systems roles all recognize "IR, passes, autodiff, LLVM codegen,
   CUDA kernels".

**Non-goals (explicitly out)**
- Beating PyTorch, torch.compile, or any production compiler.
- Covering the PyTorch op set.
- Dynamic shapes, multi-GPU or distributed execution, autotuning.
- MLIR, unless added as a stretch goal (§1.15).

## 1.3 What the finished project will look like

### 1.3.1 Finished-state checklist

- [ ] `pip install -e .` plus a CMake build produces the `minidlc` Python package and the `dlc-opt` command-line tool.
- [ ] `examples/mnist_mlp.py` trains an MLP to ~97–98% test accuracy on CPU (LLVM backend) and GPU (CUDA backend).
- [ ] `examples/char_gpt.py` trains a small GPT on TinyShakespeare and samples Shakespeare-like text. Its loss curve
      matches a reference implementation of the same model within tolerance.
- [ ] `dlc-opt --print-after-all` shows the IR after every pass; `--dot` exports graphs to Graphviz.
- [ ] Tests pass:
  - golden IR tests (lit/FileCheck);
  - differential tests against the NumPy reference (and against PyTorch where available);
  - finite-difference gradient checks;
  - a random-graph fuzzer.
- [ ] Docs: architecture, IR specification, one page per pass, a **course-concept map**, a performance report (roofline).
- [ ] A blog post or write-up explaining the design.
- [ ] Resume line updated **with measured numbers only** (§1.12).

### 1.3.2 What using it will look like

Illustrative API (final names may differ):

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
x = dl.placeholder((128, 784), dtype="f32")      # symbolic inputs with static shapes
y = dl.placeholder((128,), dtype="i32")
loss = dl.cross_entropy(model(x), y)

# One call compiles forward + backward (autodiff pass) + Adam update into a single training step.
train_step = dl.compile(loss, optimizer=optim.Adam(model.parameters(), lr=1e-3),
                        target="cpu")             # or target="cuda", or target="reference"

for xb, yb in dl.data.mnist_batches(batch_size=128):
    l = train_step(x=xb, y=yb)                  # NumPy arrays in, scalar loss out
```

### 1.3.3 What happens inside (the MLP example, step by step)

1. **Trace.** Python operator overloading records each op into a C++ graph (via bindings) instead of computing it.
2. **Graph IR.** The traced graph in SSA form:
   ```
   func @mlp_fwd(%x: f32[128,784], %y: i32[128], %w1: f32[784,256], %b1: f32[256],
                 %w2: f32[256,10], %b2: f32[10]) -> f32[] {
     %0 = matmul(%x, %w1)          : f32[128,256]
     %1 = add(%0, %b1)             : f32[128,256]   // bias broadcast over rows
     %2 = relu(%1)                 : f32[128,256]
     %3 = matmul(%2, %w2)          : f32[128,10]
     %4 = add(%3, %b2)             : f32[128,10]
     %5 = cross_entropy(%4, %y)    : f32[]
     return %5
   }
   ```
3. **Autodiff pass.** Appends the backward graph (e.g., `%g4 = softmax(%4) - onehot(%y)` scaled by 1/N;
   `%gw2 = matmul(transpose(%2), %g4)` …). The Adam update is then appended, so **one graph = one training step**.
4. **Graph passes.** Shape inference → CSE → DCE → constant folding → fusion (e.g., `add+relu` folded into the matmul
   *epilogue*) → memory planning (intermediate tensors share one arena using liveness intervals).
5. **Lowering to loop IR.** Each fused group becomes a loop nest:
   ```
   kernel @matmul_bias_relu(C: f32[128,256], A: f32[128,784], B: f32[784,256], bias: f32[256]) {
     parallel for i0 in 0..128 step 32:            // tiled for cache / shared memory
       for j0 in 0..256 step 64:
         for k in 0..784:
           for i in i0..i0+32: for j in j0..j0+64 vectorize(8):
             C[i,j] += A[i,k] * B[k,j]
       epilogue: C[i,j] = relu(C[i,j] + bias[j])
   }
   ```
6. **Loop passes.** Tiling, unrolling, vectorization, parallelization, and **bounds-check elimination**: static shapes
   plus range analysis prove indices are in range, so checks are dropped (the idea of my CS6843 A4).
7. **Code generation.**
   - *CPU:* the loop IR becomes LLVM IR with SIMD vector types, then is JIT-compiled.
     ```llvm
     %va = load <8 x float>, ptr %pa, align 4
     %vb = load <8 x float>, ptr %pb, align 4
     %vs = fadd <8 x float> %va, %vb
     %vr = call <8 x float> @llvm.maximum.v8f32(<8 x float> %vs, <8 x float> zeroinitializer) ; NaN-propagating relu
     store <8 x float> %vr, ptr %po, align 4
     ```
   - *GPU:* CUDA C++ is generated, compiled at runtime with NVRTC, and launched.
     ```cuda
     template <int T> __global__ void matmul_bias_relu(const float* A, const float* B,
                                                        const float* bias, float* C, int M, int N, int K) {
       __shared__ float As[T][T], Bs[T][T];
       int row = blockIdx.y * T + threadIdx.y, col = blockIdx.x * T + threadIdx.x;
       float acc = 0.f;
       for (int k0 = 0; k0 < K; k0 += T) {
         As[threadIdx.y][threadIdx.x] = A[row * K + k0 + threadIdx.x];   // bounds checks removed when
         Bs[threadIdx.y][threadIdx.x] = B[(k0 + threadIdx.y) * N + col]; // shapes are tile multiples
         __syncthreads();
         for (int k = 0; k < T; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
         __syncthreads();
       }
       C[row * N + col] = fmaxf(acc + bias[col], 0.f);  // fused epilogue (see NaN note in §1.5)
     }
     ```
8. **Runtime.** Allocates the arena once, keeps parameters and Adam state resident, runs the kernel sequence per step,
   and returns the loss.

### 1.3.4 Demos in the final repo

1. **MNIST MLP**: CPU and GPU training curves; test accuracy.
2. **Char-GPT on TinyShakespeare**: training curve against a reference; generated text samples.
3. **"Follow one op through the compiler"**: a walkthrough doc with IR dumps at every stage.
4. **Performance report**: each kernel on a roofline plot (CPU and GPU); fused vs unfused traffic (learning, not claims).

### 1.3.5 Repository layout (planned)

```
minidlc/
├── CMakeLists.txt, pyproject.toml, README.md, LICENSE
├── docs/
│   ├── architecture.md      ir-spec.md      course-map.md      perf-report.md      devlog.md
│   └── passes/              (one page per pass: what / why / algorithm / course link)
├── include/dlc/  &  lib/
│   ├── ir/                  graph IR: types, ops, builder, verifier, printer, parser
│   ├── passes/              shape-infer, canonicalize, const-fold, cse, dce, autodiff, fusion, mem-plan
│   ├── loopir/              loop IR, lowering, tiling, unroll, vectorize, parallelize, bounds-elim
│   ├── backends/llvm/       loop IR → LLVM IR, ORC JIT
│   ├── backends/cuda/       loop IR → CUDA C++, NVRTC, launcher
│   └── runtime/             arena allocator, thread pool, executable plan, timers
├── tools/dlc-opt/           opt-like driver: --passes=..., --print-after-all, --dot
├── python/
│   ├── bindings/            nanobind module (_minidlc)
│   └── minidlc/             frontend: tensor.py ops.py nn.py optim.py compile.py data.py reference.py
├── tests/
│   ├── lit/                 golden IR tests (*.dlc with RUN/CHECK lines)
│   ├── unit/                C++ unit tests (GoogleTest)
│   └── python/              differential, gradcheck, fuzz, end-to-end
├── examples/                mnist_mlp.py, char_gpt.py
├── benchmarks/              stream_bw.py, roofline.py, bench_ops.py
└── docker/                  CUDA build image (for building/checking CUDA code on non-GPU machines)
```

## 1.4 Architecture in detail

```
 ┌──────────────┐   trace   ┌──────────────┐  graph passes  ┌──────────────┐  lower  ┌──────────────┐ loop passes ┌────────────────────┐
 │ Python front │ ────────► │ Graph IR     │ ─────────────► │ Graph IR     │ ──────► │ Loop IR      │ ──────────► │ Backends           │
 │ end (tensors,│           │ (SSA, C++)   │ shape, CSE,DCE │ + backward   │         │ (loop nests, │ tile,unroll │ reference (NumPy)  │
 │ nn, optim)   │           │              │ autodiff,fuse, │ + optimizer  │         │ buffers)     │ vectorize,  │ LLVM → CPU (JIT)   │
 └──────────────┘           └──────────────┘ mem-plan       └──────────────┘         └──────────────┘ bounds-elim │ CUDA → GPU (NVRTC) │
                                                                                                                 └────────┬───────────┘
                                                                                     runtime: arena · thread pool · kernel plan · timers
```

### 1.4.1 Python frontend (my own)

- `Tensor` objects carry a symbolic value handle, shape and dtype. Operators (`+ * @`) and functions (`dl.relu`,
  `dl.softmax`, …) **record graph nodes** through the C++ bindings rather than computing.
- `nn` has `Module`, `Parameter`, `Linear`, `Embedding`, `LayerNorm`, `Sequential`, plus a small `MultiHeadAttention`
  built from primitive ops.
- `optim` has `SGD` and `Adam`, which emit update ops into the training graph.
- `dl.compile(loss, optimizer, target)` runs the pass pipeline and returns a callable step function.
- `dl.data` has MNIST and TinyShakespeare loaders (NumPy).
- `reference.py` is the NumPy reference interpreter: the correctness oracle. It also runs in float64 for gradient
  checks.

### 1.4.2 Graph IR (C++)

- **Values and ops in SSA form.** Every op produces new values; there is no mutation inside a graph. Parameter
  updates are returned as outputs and use **buffer donation**: each new parameter aliases its old buffer.
- **Types:** `f32` (plus `i32` for labels and indices), static shapes `f32[128,784]`, scalars `f32[]`.
- **Op set (~16 ops):**
  - `matmul` (with batch dims, i.e. bmm), `add`, `sub`, `mul`, `div` (with broadcasting);
  - `relu`, `gelu`, `exp`, `log`;
  - `softmax`, `log_softmax`, `cross_entropy`;
  - `layernorm`, `sum`, `mean`;
  - `transpose`/`permute`, `reshape`, `embedding` (gather);
  - `constant`/`full`, `onehot`.
- **Infrastructure:**
  - a builder;
  - a **verifier** (types and shapes consistent, SSA dominance);
  - a textual **printer and parser** (round-trips, so golden tests can be written as text);
  - a Graphviz exporter.

### 1.4.3 Graph passes

| Pass | What it does | Algorithm | Course link |
|---|---|---|---|
| Shape inference | Computes and validates every value's shape/dtype; broadcasting rules | Forward abstract interpretation over a shape lattice | CS6843 (abstract interpretation, §3.2) |
| Canonicalize / const-fold | `x*1→x`, `x+0→x`, fold constant subgraphs, normalize operand order | Pattern rewrites to a fixpoint | Compilers |
| CSE | Merge identical computations | Value numbering (hash of op, operands, attributes) | CS6843 (available expressions) |
| DCE | Drop values not reaching outputs | Mark live from outputs backwards (reverse reachability) | CS6843 (liveness) |
| **Autodiff (reverse mode)** | Builds the backward graph from the loss | Reverse topological order; per-op VJP rules (§1.5); **sum adjoints at fan-out** | Math: chain rule |
| Optimizer lowering | Appends SGD/Adam update ops; parameters/state as donated buffers | Graph construction | Math |
| **Fusion** | Groups ops into kernels: elementwise chains, matmul **epilogues** (bias, activation), row-wise reductions | Greedy grouping on the dataflow graph; legal only if it doesn't create cycles and iteration spaces are compatible | CS6843 PAR (dependences) |
| **Memory planning** | Assigns each intermediate tensor an arena offset; reuses memory when lifetimes don't overlap | Liveness intervals over the schedule → interference → greedy offset assignment (largest-first). Optimal for equal sizes (interval colouring); a heuristic in general | CS6843 DFA ("memory optimization" via live variables); register allocation analogy |

### 1.4.4 Loop IR and lowering

- A small **loop-nest IR** with buffers (from memory planning), affine index expressions, `for` loops with
  attributes (`parallel`, `vectorize(w)`, `unroll(u)`), statements (load, store, arithmetic, reductions), and kernel
  boundaries.
- **Lowering** turns each fused group into a loop nest using per-op templates. Examples:
  - matmul: a triple loop;
  - reductions: a loop over the reduced axis with an accumulator;
  - softmax: max pass → exp-sum pass → normalize pass;
  - layernorm: a mean/variance pass plus a normalize pass.

### 1.4.5 Loop passes

| Pass | What | Course link |
|---|---|---|
| Tiling | Split loops into tiles sized for L1/L2 (CPU) or shared memory (GPU) | CS6600 caches; CS6023 shared memory |
| Unrolling | Unroll inner loops to expose ILP and cut loop overhead | CS6600 ILP |
| Vectorization | Mark the innermost contiguous loop `vectorize(8)` for AVX2 (256-bit = 8 floats) | CS6600/CS6400 SIMD |
| Parallelization | Mark outer loops `parallel` (CPU threads / GPU blocks) after a dependence check | CS6843 PAR; ID5130 OpenMP |
| **Bounds-check elimination** | With static shapes, interval (range) analysis proves `i < M` and similar, so guards and masks disappear; versions the loop otherwise | CS6843 A2 (range analysis) + A4 (bounds-check elimination) |
| Simplification | Fold index arithmetic, hoist loop invariants | CS6843 DFA |

### 1.4.6 Backends

1. **Reference interpreter (Python/NumPy).** Executes the graph IR directly and is the oracle for every test. It also
   supports float64.
2. **LLVM CPU backend (C++).**
   - Walks the loop IR and emits LLVM IR with `IRBuilder`: loops as basic blocks with phi nodes, `<8 x float>`
     vector ops, `llvm.fmuladd`/FMA.
   - Runs LLVM's `-O3` pipeline via `PassBuilder` and JIT-compiles with **ORC LLJIT**. AOT via `clang -shared` is a
     fallback.
   - **Threading:** each kernel takes a `[begin, end)` range of its outer parallel loop; a C++ thread-pool runtime
     splits the range across cores. This avoids OpenMP codegen complexity.
3. **CUDA GPU backend (C++).**
   - Walks the loop IR and emits CUDA C++.
     - Matmul: shared-memory tiling, register blocking, then fused epilogues.
     - Reductions: warp shuffles (`__shfl_down_sync`, 5 steps for 32 lanes) plus block-level combine.
     - Softmax and layernorm: one block per row.
     - Embedding backward: `atomicAdd` scatter.
   - Compiles **at runtime with NVRTC** on the GPU machine, loads with the driver API (`cuModuleLoadData`,
     `cuLaunchKernel`), and allocates the arena with `cudaMalloc`.
   - Launch configuration comes from tile sizes.

### 1.4.7 Runtime

- **Executable plan:** an ordered list of kernel launches with buffer offsets.
- **Arena allocator:** one allocation sized by memory planning.
- **Persistent buffers:** parameters plus optimizer state (Adam `m`, `v`).
- **Device abstraction:** CPU / CUDA, with host↔device copies only at step boundaries.
- **Instrumentation:** per-kernel timers and bytes moved, for the roofline report.
- **Checkpoints:** save/load to `.npz`.

## 1.5 The math: forward and backward rules

Notation: `dY` is the upstream gradient (adjoint). N is the batch size. Row-wise ops act on the last dimension.

| Op | Forward | Backward (VJP) |
|---|---|---|
| matmul | `Y = X W` | `dX = dY Wᵀ`, `dW = Xᵀ dY` (batched: same per batch, sum over batch for shared W) |
| add (broadcast) | `Y = A + B` | `dA = dY`, `dB = sum of dY over broadcast dims` |
| mul | `Y = A ⊙ B` | `dA = dY ⊙ B`, `dB = dY ⊙ A` (reduce over broadcast dims) |
| relu | `Y = max(X, 0)` | `dX = dY ⊙ 1[X > 0]` |
| gelu (tanh approx.) | `Y = ½ X (1 + t)`, `t = tanh(√(2/π)(X + 0.044715 X³))` | `dX = dY ⊙ [½(1+t) + ½ X (1−t²) √(2/π)(1 + 3·0.044715 X²)]` |
| softmax (stable) | `S = exp(X − max X) / Σ exp(X − max X)` | `dX = S ⊙ (dY − Σ(dY ⊙ S))` (row-wise sum) |
| log_softmax | `Y = X − LSE(X)`, `LSE(X) = m + log Σ exp(X − m)` | `dX = dY − softmax(X) · Σ dY` |
| cross_entropy (int labels) | `L = (1/N) Σᵢ [LSE(zᵢ) − zᵢ,yᵢ]` | `dz = (softmax(z) − onehot(y)) / N` |
| layernorm | `x̂ = (x − μ)·rstd`, `rstd = 1/√(σ² + ε)`, `y = γ x̂ + β` | `dβ = Σ dy`, `dγ = Σ dy ⊙ x̂`; with `g = dy ⊙ γ`: `dx = (rstd/D)(D·g − Σg − x̂·Σ(g ⊙ x̂))` |
| sum / mean | `Y = Σ X` / `Y = (1/n) Σ X` | broadcast `dY` back (÷n for mean) |
| transpose / reshape | permute / view | inverse permute / reshape |
| embedding | `Y[i] = E[idx[i]]` | `dE[idx[i]] += dY[i]` (scatter-add; atomics on GPU) |

- **Adam update:**
  ```
  m ← β₁m + (1−β₁)g
  v ← β₂v + (1−β₂)g²
  m̂ = m/(1−β₁ᵗ),  v̂ = v/(1−β₂ᵗ)
  θ ← θ − lr·m̂/(√v̂ + ε)
  ```
- **Numerical-stability rules (ID5130):**
  - always subtract the row max before `exp`;
  - compute cross-entropy from log-softmax, never `log(softmax)`;
  - accumulate in fp32;
  - optionally use **online softmax** (Milakov & Gimelshein 2018), a single pass with a running max and sum.
- **NaN semantics trap:** `fmaxf(NaN, 0) = 0` and `llvm.maxnum` behave the same way, but `relu(NaN)` should be NaN to
  match PyTorch. Use `llvm.maximum` (NaN-propagating) on the CPU and an explicit `x != x` check on the GPU. A contract
  test covers this (§1.9).

## 1.6 Course-concept map (where each concept lives in the project)

| Concept | Where in MiniDLC | Notes to revisit before starting |
|---|---|---|
| SSA, IR design, verifier, textual IR | Graph IR (§1.4.2) | `CS6843_PA_Learning.md` §1.4 (LLVM primer) |
| Dataflow analysis: available expressions, liveness | CSE, DCE, memory planning | `CS6843_PA_Learning.md` §2 (DFA) |
| Abstract interpretation | Shape inference | `CS6843_PA_Learning.md` §3.2 |
| Range analysis, bounds-check elimination | Loop pass (§1.4.5) | `CS6843_PA_Learning.md` §2.6 and §10.2/§10.4 (my A2/A4 and their bugs, which are the lessons) |
| Dependence analysis, legality of reordering | Fusion, parallelization | `CS6843_PA_Learning.md` §6 (PAR) |
| lit/FileCheck testing | `tests/lit/` | `CS6843_PA_Learning.md` §1.4, §10.6 (local LLVM recipe) |
| Caches, blocking, SIMD, ILP | CPU tiling, vectorization, unrolling | `CS6600_CompArch_Learning.md` §2, §5 |
| Roofline, DSAs, Amdahl | Performance report | `CS6400_ACA_Learning.md` L12, L16; `ID5130_PSC_Learning.md` §6 |
| Threads, scheduling, speedup | CPU thread pool | `ID5130_PSC_Learning.md` §3 (OpenMP), §6 |
| CUDA memory hierarchy, coalescing, banks, shared memory | GPU matmul and reductions | `CS6023_GPU_Learning.md` §3 (L3 Memory) |
| Synchronization, atomics, reductions, warp shuffles | GPU reductions, softmax, embedding backward | `CS6023_GPU_Learning.md` §4 (L4), §8 (L8 warp primitives) |
| Numerics: stability, accumulation precision | Softmax/LSE/layernorm kernels; test tolerances | `ID5130_PSC_Learning.md` §2 |
| Chain rule, VJPs, gradient checking | Autodiff pass | §1.5 above |

## 1.7 Tech stack and environment

- **Languages and build:**
  - C++20 (core) and Python ≥ 3.11 (frontend, tests);
  - CMake plus scikit-build-core (or setuptools) for the Python package;
  - **nanobind** for bindings (pybind11 also works).
- **LLVM:** pin **one** version for the whole project.
  - Simplest: the distro's LLVM development package (e.g., `llvm` on Arch, `llvm-dev` plus `libmlir` on Ubuntu, or
    apt.llvm.org packages).
  - It must provide headers, `FileCheck` and `lit`.
  - Alternatives: the course's LLVM 20 Docker image, or the no-sudo extraction recipe in `CS6843_PA_Learning.md`
    §10.6.
  - Keep all LLVM API use inside `backends/llvm/` to contain version churn.
- **CUDA:** the CUDA Toolkit (nvcc headers, NVRTC, driver API) on the GPU machine. On non-GPU machines, a CUDA Docker
  image builds and syntax-checks the CUDA backend; kernels compile at runtime with NVRTC where they run.
- **Python packages:** NumPy, pytest, matplotlib (plots). PyTorch is optional, used only as an *external* reference in
  tests and to cross-check the char-GPT.
- **C++ tests:** GoogleTest. **Docs:** Markdown in `docs/`; Graphviz for IR graphs.

**Machines (not tied to one):**

| Option | Use it for | Notes |
|---|---|---|
| Any Linux machine with an **NVIDIA GPU** (preferred if available) | Everything, M0–M5, including Nsight Compute profiling | Install the CUDA Toolkit plus LLVM dev; do everything in one place |
| CPU-only Linux machine (e.g., the current laptop: i7-8550U, 4C/8T, AVX2+FMA, L1d 32 KB/core, L2 256 KB/core, L3 8 MB, 8 GB RAM) | M0–M2 fully; build and check M3 code | Run GPU parts elsewhere |
| IITM GPU cluster | M3–M4 runs and Nsight profiling | Profilers need GPU performance-counter access, which clusters usually allow |
| Colab / Kaggle (T4: CC 7.5, 16 GB, ~8.1 TFLOPS fp32, ~320 GB/s) | M3–M4 runs | Free tier; profiling counters are often restricted |

## 1.8 Build plan: milestones

Each milestone ends with a tagged commit, passing tests, and a short devlog entry.

### M0 — Foundations (≈1 week)

- **Tasks:**
  - repo skeleton, CMake + Python package, nanobind module stub;
  - graph IR core: types, values, ops, builder, verifier, printer/parser;
  - Python frontend tracing for `matmul`, `add`, `relu`;
  - NumPy reference interpreter;
  - test infrastructure: pytest, GoogleTest, lit/FileCheck config;
  - CI (GitHub Actions, CPU only).
- **Done when:** a small `relu(x @ W + b)` graph is traced from Python, printed, round-trips through the parser,
  verifies, runs in the reference interpreter, and matches NumPy. The first lit tests pass.
- **Concepts:** SSA, IR design, testing discipline.

### M1 — Graph passes + autodiff; MNIST through the interpreter (≈2 weeks)

- **Tasks:**
  - the full op set with shape inference;
  - canonicalization, constant folding, CSE, DCE;
  - the **autodiff pass** with VJP rules (§1.5);
  - SGD/Adam lowering;
  - `dl.compile(..., target="reference")`;
  - finite-difference gradient checks (float64) for every op;
  - the `dlc-opt` tool with `--print-after-all`.
- **Done when:** **the MLP trains on MNIST through the reference interpreter** to ~97% test accuracy (slow but
  correct), and gradient checks pass for all ops.
- **Concepts:** dataflow (CSE/DCE), abstract interpretation, the chain rule.
- **Risks:** fan-out adjoint accumulation bugs, and broadcasting gradients summed over the wrong axes. Gradient
  checks catch both.

### M2 — Loop IR + LLVM CPU backend + memory planning (≈3 weeks)

- **Tasks:**
  - loop IR and lowering templates;
  - loop passes: tiling, unrolling, vectorization, parallelization, simplification;
  - LLVM codegen (IRBuilder, vector types, `-O3`, ORC JIT);
  - thread-pool runtime;
  - memory planning plus arena;
  - elementwise-chain and epilogue fusion;
  - per-kernel timers;
  - STREAM-like bandwidth and peak-FLOP microbenchmarks for the roofline.
- **Done when:** **the MLP trains on CPU through LLVM-generated code**, with results matching the reference within
  tolerance; a CPU roofline plot exists; IR dumps show fusion and memory reuse.
- **Concepts:** caches and blocking, SIMD, ILP, threads, liveness-based allocation, LLVM codegen.
- **Milestone note:** **this is a complete, honest stopping point** — a full CPU deep-learning compiler. The resume
  entry's frontend, graph-pass and CPU-backend bullets can switch to past tense here (§1.12).

### M3 — CUDA backend (≈2–3 weeks)

- **Tasks:**
  - CUDA codegen: shared-memory tiled matmul with fused epilogues, elementwise kernels, warp-shuffle reductions,
    row-wise softmax/layernorm/cross-entropy, embedding gather and atomic scatter;
  - NVRTC compilation plus driver-API launch;
  - device arena;
  - host↔device copies at step boundaries.
- **Done when:** the MLP trains on GPU, matches the reference, and has a GPU roofline plot (Nsight Compute where
  available).
- **Concepts:** CUDA memory hierarchy, coalescing, bank conflicts, synchronization, atomics, warp primitives.

### M4 — Transformer ops + bounds-check elimination; char-GPT (≈2 weeks)

- **Tasks:**
  - batched matmul, causal mask (as a constant added before softmax), layernorm, gelu, embedding with positional
    table;
  - `nn.MultiHeadAttention` from primitives;
  - range-analysis bounds-check elimination with loop versioning;
  - a char-GPT example: vocab 65, context ~64–128, 2–4 layers, d_model ~128, 4 heads.
- **Done when:** **the char-GPT trains on CPU and GPU**, its loss curve tracks a reference implementation, it samples
  readable text, and bounds checks disappear from IR dumps when shapes are tile multiples.
- **Concepts:** range analysis (A2/A4), attention math, numerics.

### M5 — Polish and write-up (≈1 week)

- **Tasks:**
  - docs: architecture, IR spec, per-pass pages, **course map**, performance report;
  - README quickstart (clone → train MNIST in about 10 minutes);
  - a blog post;
  - cleanup;
  - a license;

## 1.9 Testing and verification strategy

| Level | What | Tooling |
|---|---|---|
| Golden IR | Each pass's input → expected output text (e.g., CSE merges duplicates; DCE removes dead values; fusion groups) | `dlc-opt` + lit/FileCheck, like CS6843 assignments |
| Unit | IR builder/verifier, interval analysis, memory planner (no two live tensors overlap) | GoogleTest |
| Gradients | Every op's VJP vs central finite differences in **float64** (reference interpreter) | pytest |
| Differential | Every backend (LLVM, CUDA) vs the reference interpreter on random inputs. Tolerances: elementwise `rtol=1e-5`; matmul/reductions scaled with K (e.g., `rtol≈1e-4`) | pytest |
| **Contract checks** | NaN/Inf propagation, non-tile-multiple shapes (masking or versioning path), zero-size dims, large magnitudes (softmax overflow), fp32 accumulation | pytest |
| Fuzzing | Random valid graphs (random ops and shapes) through the whole pipeline; compare to the reference | pytest + seeds |
| End to end | MNIST accuracy threshold; char-GPT loss curve vs reference within a band | examples as tests (nightly) |

## 1.10 Performance methodology (for learning, not claims)

- **Roofline per machine:**
  - measure peak bandwidth (STREAM-like copy/triad) and peak FLOPs (FMA microbenchmark) on the actual machine;
  - for each kernel, compute arithmetic intensity (FLOPs / bytes moved), measure GFLOP/s, and plot it against the
    roofline.
  - Example estimate for the current laptop: AVX2+FMA gives 32 fp32 FLOPs/cycle/core, so ~4 cores × ~3 GHz ≈ ~380
    GFLOP/s theoretical, thermally limited in practice. DRAM ~30–38 GB/s. **Measure; don't quote these.**
- **Teaching comparisons:** fused vs unfused (bytes moved and time); tiled vs naive matmul; scalar vs vectorized;
  thread scaling (Amdahl).
- **Context baselines:** NumPy (BLAS) and PyTorch eager on the same machine, reported honestly whatever the outcome.

## 1.11 Definition of done (whole project)

1. MNIST MLP and char-GPT train end to end on **both** CPU (LLVM) and GPU (CUDA), matching the reference.
2. All test levels in §1.9 pass in CI (CPU) and on the GPU machine.
3. `dlc-opt` shows every pass; every pass has a doc page with a course link.
4. A performance report with roofline plots and honest baselines.
5. README quickstart works on a fresh machine; a blog post is published.
6. The resume entry contains only facts and measured numbers.

## 1.12 Resume entry

- **Rules:**
  - list only tools actually used;
  - every number must come from `docs/perf-report.md`;
  - don't post-date the timeline;
  - link the repo.
- **House style** (matching my existing entries): three bullets, verb first, every clause carrying a concrete
  technical noun or number. No problem-statement opener, no meta-commentary about evaluation.
- **Story order:** problem → what I'm building → implementation and the coursework it applies → how success is measured.
- **Keeping it honest:** the entry says *In Progress* until the work exists. As each part is built, switch that
  bullet to past tense; after M5, replace the last bullet with measured results (accuracy, roofline %).

```latex
    \rowcolor{SecondaryColor}
    \CellStrut{\EntryTitleFont MiniDLC: End-to-End Deep-Learning Compiler} \textit{| C++20, LLVM, CUDA, Python} \hfill {\DateRange{Oct'26}{Present}} \\
    \vspace{-1.5ex}
    \begin{Bullets}
      \item Engineering an end-to-end \textbf{deep-learning compiler} that lowers models from a custom \textbf{Python} frontend through C++ \textbf{SSA graph} and \textbf{loop IRs} into native machine code.
      \item Implementing \textbf{reverse-mode autodiff} as a graph pass, dependence-checked \textbf{operator fusion}, and liveness-based \textbf{memory planning} via interference-graph coloring.
      \item Generating vectorized multithreaded CPU code through \textbf{LLVM ORC JIT} and tiled GPU kernels through \textbf{CUDA NVRTC} (shared memory, warp-shuffle reductions) to train an \textbf{MNIST MLP} and \textbf{char-GPT}.
    \end{Bullets}
    \vspace{-1ex} \\ \hline
```

## 1.13 Interview questions this project prepares me for

1. Explain reverse-mode autodiff, and why one backward pass costs about the same as the forward pass for a scalar
   loss.
2. How does the fusion pass decide what is legal and profitable to fuse? Why do memory-bound ops benefit most?
3. How does memory planning work, how does it relate to register allocation, and why use a greedy heuristic?
4. Why does tiling speed up matmul (data reuse in cache / shared memory)? What are coalescing and bank conflicts?
5. Walk through a warp-shuffle reduction. Why 5 steps? When are atomics needed (embedding backward)?
6. Roofline: arithmetic intensity of matmul vs elementwise ops; where my kernels sit and why.
7. Numerical stability: why subtract the max in softmax? Why compute cross-entropy from log-softmax? What is the NaN
   behaviour of `max`?
8. SSA, value numbering (CSE), DCE via liveness; what my verifier checks.
9. LLVM: IRBuilder, phi nodes, vector types, the `-O3` pipeline, ORC JIT.
10. Testing a compiler: golden IR tests, differential testing, gradient checks, fuzzing.
11. Bounds-check elimination with range analysis and loop versioning (my CS6843 A4 idea on GPU tiles).

## 1.14 Risks and mitigations

| Risk | Mitigation |
|---|---|
| Scope creep | Fixed op list, static shapes, fp32; anything else goes to stretch goals (§1.15). Milestone gates. |
| LLVM API/version churn | Pin one LLVM version; isolate LLVM calls in `backends/llvm/` |
| GPU access | CPU-first design; CUDA built via Docker anywhere; run on whichever GPU machine is available (§1.7) |
| Numerical mismatches | float64 reference, explicit tolerance policy, contract tests |
| Autodiff bugs (fan-out, broadcasting) | Per-op gradient checks plus whole-model gradient checks on tiny models |
| Time pressure (placements) | **M2 is a complete stopping point** (a full CPU compiler training MNIST); M3–M5 extend it |
| Performance below expectations | Not a goal; report honestly against the roofline |

## 1.15 Stretch goals (only after the definition of done)

- **Triton backend** (emit Triton instead of CUDA C++).
- **Fused, logits-free LM-head + cross-entropy** via online log-sum-exp (see `Phase3_Project_Ideas.md` §12.3).
- **Mixed precision** (bf16/fp16 storage with fp32 accumulation) plus error analysis.
- **MLIR export** of one op family (`linalg` → `gpu` → NVVM).
- Simple **autotuner** for tile sizes; **CUDA graphs** for launch overhead; a kernel cache.
- A small **web visualizer** for IR dumps (Compiler-Explorer-like).
- **PyTorch FX import** (now optional; the main frontend stays my own).

## 1.16 Open decisions to finalize at kickoff

1. Project name (working name: MiniDLC).
2. LLVM version to pin (distro LLVM vs the course LLVM 20 image).
3. nanobind (recommended) vs pybind11.
4. JIT (ORC, recommended) vs AOT shared-library build for the CPU backend.
5. Which GPU machine for M3–M4.
6. License (MIT or Apache-2.0).
7. C++ test framework (GoogleTest recommended).

## 1.17 References and learning resources

- **Closest curriculum:** CMU 10-414/714 *Deep Learning Systems* ("needle": autograd + CPU/CUDA backends) —
  https://dlsyscourse.org
- **Autodiff:** Karpathy's micrograd (https://github.com/karpathy/micrograd); Baydin et al., *Automatic
  Differentiation in Machine Learning: a Survey*, JMLR 2018 (https://www.jmlr.org/papers/v18/17-468.html).
- **Models and kernels:**
  - Karpathy's llm.c (GPT-2 in raw C/CUDA, reference kernels) — https://github.com/karpathy/llm.c
  - nanoGPT (char-level TinyShakespeare setup) — https://github.com/karpathy/nanoGPT
- **LLVM:** Kaleidoscope tutorial (https://llvm.org/docs/tutorial/); ORC JIT (https://llvm.org/docs/ORCv2.html).
- **IR design:** MLIR Toy tutorial — https://mlir.llvm.org/docs/Tutorials/Toy/
- **Compiler architectures:**
  - tinygrad (https://github.com/tinygrad/tinygrad);
  - Chen et al., *TVM*, OSDI 2018;
  - Ragan-Kelley et al., *Halide*, PLDI 2013.
- **CUDA:**
  - *CUDA C++ Programming Guide*;
  - NVRTC docs (https://docs.nvidia.com/cuda/nvrtc/);
  - Hwu, Kirk & El Hajj, *Programming Massively Parallel Processors* (4th ed.).
- **Bindings:** nanobind — https://github.com/wjakob/nanobind
- **Performance:**
  - Williams, Waterman & Patterson, *Roofline*, CACM 2009;
  - McCalpin's STREAM benchmark.
- **Algorithms:**
  - Poletto & Sarkar, *Linear Scan Register Allocation*, TOPLAS 1999 (memory planning analogy);
  - Milakov & Gimelshein, *Online normalizer calculation for softmax*, 2018 (arXiv:1805.02867).
- **Compiler textbooks:** Aho, Lam, Sethi & Ullman (Dragon book), Ch. 8–9; Cooper & Torczon, *Engineering a
  Compiler*.
- **My notes:** `Courses/*/…_Learning.md` (see §1.6 for exact sections).

## 1.18 Status log

| Date | Update |
|---|---|
| 2026-10-09 | Project decided and re-scoped to a learning-focused, complete, course-driven compiler. Own Python frontend chosen. Spec written. Not started. Brainstorm history: `Phase3_Project_Ideas.md` §12.1–§12.5. |

---


