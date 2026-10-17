# MiniDLC Architecture

This document has two parts: the **v2 design** that MiniDLC is being rebuilt to (Part 1), and the **status of each
component** as of the 2026-10-17 audit (Part 2). Known bugs are recorded as disabled unit tests in
`tests/unit/test_known_bugs.cpp` and expected-failure lit tests; each is enabled by the phase that fixes it.

## Part 1: v2 design

### Why v2: the design gaps in v1

The v1 shape — Python → SSA graph IR → graph passes → loop IR → LLVM/CUDA, with a NumPy oracle — was right. But six
design gaps made the backends template-based and several passes wrong (see [Part 2](#part-2-status-audit-2026-10-17)):

1. **The loop IR was CPU-shaped.** Loops carried only `parallel`/`vectorize`/`unroll`. Nothing expressed which block or
   thread runs a loop, global vs shared vs register memory, barriers, reductions across threads, or guards. A backend
   that can't read these from the IR has to recognise "this is a matmul" and paste a template.
2. **What a kernel computes was not separated from how it is scheduled.** Tiling for L1 and tiling for shared memory
   with thread binding are different transformations of one computation; one loop IR can't serve two targets without
   per-target schedules.
3. **The op set was forward-only.** The backward rules need `1[X>0]`, scatter-add, reductions over broadcast axes and the
   normalized x̂; none were ops. (The C++ ReLU VJP multiplied by `x` for lack of a select op.)
4. **Memory planning sat at the wrong level.** Liveness must be computed per kernel after fusion, with parameters and
   optimizer state excluded. Planning on the unfused op schedule placed a fused kernel's output on top of its inputs.
5. **Autodiff had no activity analysis.** Gradients were built for every input, including the MNIST images: a
   128×256×784 GEMM per step (about half of the backward FLOPs) thrown away.
6. **Bounds-check elimination had nothing to eliminate.** The loop IR had no guard statement, so the pass could only
   add an attribute.

Two process gaps: nothing required each pass to be checked for semantic correctness, and there was no plan for
developing GPU code without a local GPU.

### Overview

```
 ┌──────────────┐ IR text  ┌───────────────────────── libdlc (C++) ──────────────────────────────┐
 │ Python       │────────► │ Graph IR ─► graph passes ─► fused graph + buffer plan                │
 │ frontend     │          │        │ lower each fused group                                      │
 │ nn·optim·data│          │        ▼                                                             │
 └──────┬───────┘          │ Loop IR v2: compute ─► target schedule ─► loop passes                │
        │                  │  (thread binding, shared/register buffers, barriers, guards, reduce) │
        │                  │        ▼  one LLVM IR emitter                                        │
        │                  │  x86-64 ─► ORC LLJIT              NVPTX ─► PTX text                  │
        │                  └────┬──────────────────────────────────┬──────────────────────────────┘
        │                       ▼                                  ▼
        │               CPU device (thread pool)       CUDA device (driver loads PTX → GPU code)
        │                       └── runtime: executable plan · persistent + transient arenas ──┘
        ▼
 Reference interpreter (NumPy, fp64) executes the IR text before/after every pass = the oracle
```

Key decisions:

1. **The textual graph IR is the contract between Python and C++.** Tracing stays in Python; `dl.compile` serializes
   the graph to IR text; nanobind exposes only `compile` and `run`. The reference interpreter also executes IR text,
   so every C++ pass can be checked by running the graph before and after it on random inputs.
2. **One computation, one schedule per target.** Lowering produces a compute definition; rule-based schedules per kernel
   class (matmul-like, elementwise, row reduction, gather/scatter) turn it into the loop nest for each target. No
   autotuning (non-goal).
3. **One LLVM IR emitter, two targets.** x86-64 and NVPTX differ only in intrinsics, address spaces and the runtime.
   The emitters (including the optional CUDA C++ one) translate the loop IR mechanically and every optimization
   decision lives in the schedule, so codegen can never fall back to "templates by name".
4. **Compiling and running are separable.** An executable (plan + PTX or object code + parameters) can be saved on the
   laptop and run on Colab with only the runtime and the NVIDIA driver.
5. **A small device layer** (allocate, copy, launch, synchronize, timers) with CPU and CUDA implementations; a
   persistent region for parameters and Adam state, a planned transient arena, and in-place parameter updates.

### Python frontend

- `Tensor` objects carry a shape, dtype, op and inputs. Operators (`+ * @`) and functions (`dl.relu`, `dl.softmax`, …)
  **record graph nodes** instead of computing.
- `nn` has `Module`, `Parameter`, `Linear`, `Embedding`, `LayerNorm`, `Sequential`, and `MultiHeadAttention` built from
  primitive ops (heads split with `reshape` + `transpose` and computed with 4-D batched matmul).
- `optim` has `SGD` and `Adam`. Under `dl.compile` they describe the update, which the C++ optimizer-lowering pass
  appends to the training graph; the eager NumPy versions remain for the reference target.
- `dl.compile(loss, optimizer, target)` exports IR text, runs the C++ pipeline and returns a callable step function;
  `target="reference"` runs the NumPy interpreter instead.
- `dl.data` has MNIST and TinyShakespeare loaders (NumPy).
- `reference.py` is the NumPy reference interpreter — the correctness oracle. It runs in float64 for gradient checks and
  executes IR text, so C++ passes can be translation-validated.

### Graph IR (C++)

- **Values and ops in SSA form.** Every op produces new values; there is no mutation inside a graph. Parameter and
  optimizer-state updates are outputs annotated as **donated**: each aliases its old buffer.
- **Function signature** distinguishes `params` (trainable), `state` (Adam `m`, `v`, step count) and per-step `inputs`.
- **Types:** `f32` (plus `i32` for labels and indices), static shapes `f32[128,784]`, scalars `f32[]`.
- **Op registry** (one table, like a tiny MLIR ODS): each op declares its arity, attributes, shape/dtype rule, VJP rule,
  lowering class and reference implementation. Shape inference, the verifier, autodiff, lowering and the Python
  reference interpreter all read the same definitions.
- **Op set:**
  - forward: `matmul` (with batch dims and `transpose_a`/`transpose_b` flags), `add`, `sub`, `mul`, `div`
    (broadcasting), `relu`, `gelu`, `exp`, `log`, `softmax`, `log_softmax`, `cross_entropy`, `layernorm`,
    `sum`/`mean` (list of axes, keepdims), `transpose`, `reshape`, `embedding`, `constant`/`full`, `onehot`;
  - **backward primitives** (needed by the backward rules): `cmp_gt`, `select`, `broadcast_to`, `reduce_sum(axes)`,
    `reduce_max(axes)`, `scatter_add`, `cast`, `tanh` (GELU derivative), `rsqrt`.
- **Infrastructure:** a builder; a **verifier** (per-op type rules, SSA dominance, signature, a single terminator); a
  textual **printer and parser** that round-trip — attributes and multi-value returns included — and report errors on
  undefined names; a Graphviz exporter; use lists kept consistent when ops are erased.

### Graph passes

Pipeline order:

| # | Pass | Kind | Purpose | Algorithm / course link |
|---|---|---|---|---|
| 1 | Verifier | analysis | Types, shapes, dominance, signature; after every pass in debug builds | SSA (CS6843) |
| 2 | Shape/dtype inference | analysis | Infer types; reject invalid programs with an error | Forward abstract interpretation over a shape lattice (CS6843) |
| 3 | Canonicalize + const-fold | transform | Type-preserving, NaN-safe identities; constant folding; transposes folded into matmul flags; reshape/transpose cancellation | Pattern rewrites to a fixpoint |
| 4 | CSE | transform | Merge identical computations (op, operands, attributes, result type) | Value numbering (CS6843 available expressions) |
| 5 | DCE | transform | Drop values that don't reach an output | Reverse reachability (CS6843 liveness) |
| 6 | **Activity analysis** | analysis | Values that need adjoints = forward-varied (depend on a parameter) ∩ backward-useful (reach the loss) | Two dataflow passes (CS6843 DFA); standard in AD tools |
| 7 | **Autodiff (reverse mode)** | transform | Backward graph from the loss, for active values only | Reverse topological order; per-op VJP rules; adjoints summed at fan-out |
| 8 | Optimizer lowering | transform | Adam/SGD as elementwise ops; parameter/state outputs donated | Graph construction |
| 9 | Cleanup | transform | Canonicalize, CSE and DCE again on the backward graph | — |
| 10 | Cost model | analysis | FLOPs and bytes per op and per kernel | Drives fusion decisions and the roofline report |
| 11 | **Fusion** | transform | Kernels: matmul epilogues, elementwise chains (multiple outputs), row kernels (softmax, LayerNorm, cross-entropy) | Greedy on the dataflow graph; legal only if no cycle is created (reachability), iteration spaces are compatible, and every external operand is defined before the kernel runs (CS6843 PAR) |
| 12 | **Buffer assignment** | analysis + transform | Arena offsets for transient values; persistent region for parameters/state; in-place reuse; 256-byte alignment | Per-kernel liveness intervals → interference → greedy offset assignment, largest first (CS6843 DFA; register-allocation analogy) |

### Loop IR v2 and lowering

A small loop-nest IR (a "TIR-lite") that can express both targets:

- **Loops** with a kind: `serial`, `parallel` (CPU thread-pool chunks), `vectorize(w)`, `unroll(u)`, or
  `bind(blockIdx.{x,y,z} | threadIdx.{x,y,z})` (GPU). Extents are static.
- **Buffers** with a memory scope: `global`, `shared` (per block) or `register` (per thread); `alloc` statements for
  shared and register buffers.
- **Statements:** load/store, scalar assignment, `if` guards (explicit, so BCE has something to delete), `barrier`,
  `atomic_add`, warp shuffle, nested loops.
- **Expressions:** affine index expressions in canonical form (Σ cᵢ·iᵢ + c₀) for analysis; float ops (`add`, `mul`,
  `fma`, `max_nan`, `exp`, `log`, `tanh`, `rsqrt`, `select`, comparisons).
- **Kernel:** parameters (buffers and scalars), a body, and a launch configuration derived from the bound loops.

**Lowering** turns each fused group into a compute definition; **schedules** (rule-based, per kernel class and target)
turn it into loops:

| Kernel class | CPU schedule | GPU schedule |
|---|---|---|
| Matmul-like (+ epilogue) | parallel row tiles, k-blocking, `vectorize(8)` on columns | 64×64 block tile, 16-wide k tiles staged in shared memory, 4×4 register tile per thread, epilogue in registers |
| Elementwise / injective | flatten, `vectorize(8)`, parallel chunks | 1-D grid of 256-thread blocks, guard on the tail |
| Row reduction (softmax, LayerNorm, cross-entropy, bias gradient) | parallel rows, vectorized inner loop with a horizontal reduce | one warp or one block per row: warp shuffles plus a shared-memory combine across warps |
| Gather / scatter (embedding) | parallel rows | 2-D grid; `atomic_add` for the scatter |

The matmul+bias+ReLU group of the MNIST MLP's first layer with the T4 schedule:

```
kernel @mm_bias_relu  grid(4,2) block(16,16)          // 64×64 tile per block, 4×4 outputs per thread
  for by in 0..2 bind(blockIdx.y), bx in 0..4 bind(blockIdx.x)
  for ty in 0..16 bind(threadIdx.y), tx in 0..16 bind(threadIdx.x)
    alloc As f32[64,16] shared (padded vs bank conflicts) · Bs f32[16,64] shared · acc f32[4,4] register = 0
    for k0 in 0..49                                    // 784/16; guards proven redundant: 128%64 = 784%16 = 0
      cooperative load As ← A, Bs ← B ; barrier
      for kk in 0..16 unroll: acc[r,c] = fma(As[ty*4+r,kk], Bs[kk,tx*4+c], acc[r,c])
      barrier
    C[by*64+ty*4+r, bx*64+tx*4+c] = max_nan(acc[r,c] + bias[...], 0)    // epilogue stays in registers
```

### Loop passes

| Pass | Kind | Purpose | Course link |
|---|---|---|---|
| Affine access analysis | analysis | Index = Σ cᵢ·iᵢ + c₀; strides, contiguity, coalescing | CS6843 |
| Dependence analysis | analysis | Distance/direction vectors and reduction detection; legality of the transforms below | CS6843 PAR |
| Schedule primitives | transform | Split/tile, reorder, bind, stage into shared memory/registers, unroll, vectorize, parallel | CS6600 caches, ILP, SIMD; CS6023 shared memory |
| Cross-thread reduction | transform | Warp shuffles (5 steps for 32 lanes) plus a shared-memory combine; atomics for scatter | CS6023 L4, L8 |
| Guard insertion + **bounds-check elimination** | analysis + transform | Range analysis deletes the guards it proves true; partial tiles split into an unguarded interior and a guarded boundary | CS6843 A2 (range analysis) + A4 (bounds checks) |
| Simplification | transform | Fold index arithmetic, hoist loop invariants (LICM), strength reduction | CS6843 DFA |
| Bank-conflict analysis + padding | analysis + transform | Shared-memory strides mod 32 → padded tiles (e.g. `[16][17]`) | CS6023 L3 |
| Resource check | analysis | Shared memory, threads and registers per block vs the target's limits | CS6023 |

### Backends

| Backend | Path | Role |
|---|---|---|
| `reference` | NumPy; executes graph IR (float64 capable) | Oracle; translation validation of every pass |
| Loop IR interpreter (development only) | Executes loop nests sequentially | Finds lowering and schedule bugs before codegen |
| `cpu` | Loop IR → LLVM IR (`IRBuilder`) → `-O3` (`PassBuilder`) → **ORC LLJIT**; thread pool | Main CPU target (x86-64, AVX2+FMA) |
| `cuda` | Same emitter with the NVPTX target → PTX → `cuModuleLoadDataEx` → `cuLaunchKernel` | Main GPU target; `sm_75` (T4) first, configurable |
| `cuda-c` (optional) | Loop IR → CUDA C++ → NVRTC → PTX | Readable kernels; cross-check of the PTX path |

**The GPU path.** PTX is NVIDIA's virtual assembly language: plain text with unlimited virtual registers. The driver
compiles it to the real machine code of whichever GPU loads it, so PTX can be produced on any machine. LLVM's NVPTX
backend (part of LLVM 23; no CUDA toolkit needed) emits it, so PTX is never written by hand except as inline snippets
for specific instructions. Mapping, each row checked locally with clang 23:

| Loop IR v2 | LLVM IR (NVPTX) | PTX |
|---|---|---|
| Loop bound to a thread / block | `llvm.nvvm.read.ptx.sreg.tid.x` / `.ctaid.x` | `%tid.x` / `%ctaid.x` |
| Global / shared buffer | `ptr addrspace(1)` / `addrspace(3)` global | `ld.global` / `.shared`, `ld.shared` |
| Barrier | `llvm.nvvm.barrier0` | `bar.sync 0` |
| Warp shuffle | `llvm.nvvm.shfl.sync.down.f32` | `shfl.sync.down.b32` |
| Guard | `icmp` + `br` | `setp` + `@%p bra` |
| Scatter-add | `atomicrmw fadd` (LLVM 23 emits a compare-and-swap loop) or inline PTX | `red.global.add.f32` |
| NaN-propagating max | `llvm.maximum.f32` | `max.NaN.f32` on `sm_80`; ~6 extra instructions on `sm_75` |

**Developing without a local GPU:** PTX golden tests (FileCheck) run in CPU CI; `ptxas -v` in the CUDA Docker image
validates PTX and reports registers and spills; execution, differential tests and `compute-sanitizer` run on Colab.

### Runtime

- **Device layer:** `alloc`, `free`, `copy_h2d`/`copy_d2h`, `launch`, `synchronize`, timers. CPU device: aligned arena
  plus thread pool. CUDA device: driver API loaded with `dlopen` (`cuInit`, primary context, `cuMemAlloc_v2`,
  `cuModuleLoadDataEx`, `cuLaunchKernel`, `cuEvent*`), so the laptop build needs no CUDA.
- **Memory:** a persistent region (parameters, Adam `m`/`v`) allocated once and updated in place; a transient arena
  sized by buffer assignment; host↔device copies only at step boundaries (`x`, `y` in; loss out).
- **Executable plan:** an ordered list of kernel launches with buffer offsets, scalar arguments (e.g. the Adam step) and
  launch configurations.
- **Artifacts:** save/load an executable (plan + PTX or object code + parameters) to compile on one machine and run on
  another; a kernel cache keyed by a hash of the loop IR and the target.
- **Instrumentation:** per-kernel timers (CPU clock; CUDA events) and bytes/FLOPs from the cost model, feeding the
  roofline report.
- **Checkpoints:** save/load parameters and optimizer state to `.npz`.

### Testing strategy

| Level | What | Tooling |
|---|---|---|
| Golden IR | Each pass's input → expected output text (CSE merges duplicates; DCE removes dead values; fusion groups), plus loop IR, LLVM IR and PTX dumps (e.g. `bar.sync`, `shfl.sync.down`, guards removed by BCE) | `dlc-opt` + lit + LLVM FileCheck |
| Unit | IR builder/verifier, parser round-trip, interval and dependence analysis, buffer assignment (no two live buffers overlap) | GoogleTest; assertions always on |
| **Translation validation** | Every graph pass: run the graph before and after it in the reference interpreter on random inputs | pytest |
| Gradients | Every op's VJP — Python and C++ — vs central finite differences in **float64** | pytest |
| Differential | Loop IR interpreter, CPU and GPU executables vs the reference on random inputs. Tolerances: elementwise `rtol=1e-5`; matmul/reductions scaled with K (e.g. `rtol≈1e-4`) | pytest (GPU on Colab) |
| **Contract checks** | NaN/Inf propagation, non-tile-multiple shapes (guard/peeling path), zero-size dims, large magnitudes (softmax overflow), fp32 accumulation — on every backend, not only NumPy | pytest |
| Fuzzing | Random valid graphs (random ops and shapes) through the whole pipeline vs the reference; parser round-trip fuzzing | pytest + seeds |
| GPU checks | `ptxas -v` (registers, spills) in the CUDA Docker image; `compute-sanitizer` memcheck/racecheck on Colab | CUDA toolkit |
| End to end | MNIST accuracy threshold; char-GPT held-out loss vs a reference within a band | examples as tests (nightly) |

All C++ tests run with assertions enabled, and CI runs them under AddressSanitizer and UndefinedBehaviorSanitizer.

### Build phases

Each phase ends in a tested, demonstrable state.

| Phase | Deliverable | Done when | Size |
|---|---|---|---|
| **P0** | Truthful docs and resume; private notes untracked; LLVM 23 dev package; GoogleTest with always-on assertions (known bugs recorded as disabled tests); real lit + FileCheck; CPU CI with ASan/UBSan | CI green with real checks; docs claim only what exists | S |
| **P1** | Op registry; robust parser (attributes, multi-value returns, errors on undefined names) and verifier; use-list hygiene; Python → IR text export; reference interpreter executes IR text | Print → parse → print is a fixpoint for the MLP and Char-GPT graphs; every graph pass is translation-validated | M |
| **P2** | Backward op set; C++ autodiff fixed, with activity analysis; optimizer lowering with donation; canonicalize/CSE fixes; transpose folding | C++ training-graph gradients match finite differences and the Python VJPs; one graph = one training step | M |
| **P3** | Fusion with legality checks and a cost model; per-kernel buffer assignment with persistent and transient regions | A checker proves no two live buffers overlap; peak memory measured on the training graph | M |
| **P4** | Loop IR v2, CPU schedules, loop analyses and transforms, LLVM `IRBuilder` emitter + ORC LLJIT, device layer and runtime, nanobind | MNIST and Char-GPT train through generated CPU code and match the reference — the first complete stopping point | L |
| **P5** | GPU schedules, NVPTX emission, CUDA device runtime, artifact save/load, PTX golden tests | MNIST and Char-GPT train on a Colab T4 and match the reference | L |
| **P6** | Real multi-head attention; row-reduction kernels; embedding scatter; held-out evaluation | Char-GPT beats the bigram baseline on both targets and tracks a PyTorch reference | M |
| **P7** | Baselines (OpenBLAS NumPy, PyTorch CPU/GPU, cuBLAS) and ablations; walkthrough doc; resume from measured numbers | All six definition-of-done items (§1.11) met | M |

## Part 2: Status (audit, 2026-10-17)

`✓` built and working · `◐` partial or buggy · `✗` missing. Evidence: a Debug build with assertions and
AddressSanitizer/UndefinedBehaviorSanitizer, targeted repro programs, and re-measurement on the i7-8550U laptop.

| Spec item | Status | Notes |
|---|---|---|
| Frontend, `nn`, data loaders | ✓ | `MultiHeadAttention` is single-head (no head split) |
| Reference interpreter, fp64 gradient checks | ✓ | The strongest part of v1 |
| Python ↔ C++ link (nanobind; graph exchange) | ✗ | 15-line stub; no exchange at all |
| `dl.compile` runs the pass pipeline | ✗ | Runs the interpreter, or a hard-coded MLP |
| Optimizer as ops in the graph; buffer donation | ✗ | Adam runs eagerly in NumPy |
| Graph IR, builder, Graphviz export | ✓ | |
| Verifier (types, shapes, dominance) | ◐ | Only checks def-before-use |
| Printer/parser round-trip | ✗ | Attributes dropped, returns lost, hangs on a multi-value return |
| Shape inference, canonicalize, CSE, DCE | ◐ | No shape errors; constant folding is a use-after-free; CSE merges different constants; dangling uses |
| Autodiff pass | ◐ | Structure fine; ReLU, GELU, mean and LayerNorm VJPs wrong; embedding missing |
| Fusion | ◐ | Finds groups, but no legality checks, and lowering of the groups is broken |
| Memory planning | ◐ | Algorithm fine; wrong level; overwrites parameters |
| Loop IR and lowering templates | ◐ | 2-D matmul and the first elementwise op only; everything else becomes a copy loop |
| Tiling, unroll, vectorize, parallelize, simplify | ✗ | No-op / never run / attribute only / unconditional / missing |
| Range-analysis BCE and loop versioning | ◐ | Analysis and peeling exist; no guards; codegen ignores the result |
| LLVM backend: `IRBuilder`, `-O3` via `PassBuilder`, ORC JIT | ✗ | Text templates chosen by kernel name, compiled by a `clang` subprocess |
| CUDA backend: codegen, NVRTC, driver launch, device arena | ✗ | Text templates (softmax wrong beyond 32 columns); never compiled or run |
| Runtime: plan, device layer, timers, checkpoints | ◐ | Plan and arena exist but are unused; no device layer; no checkpoints |
| Tests | ◐ | Gradient checks ✓; no differential tests; the rest weak or NumPy-only; asserts compiled out; no CI |
| Performance | ◐ | CPU roofline exists; GPU unmeasured; baselines skewed |

**Measured facts behind the table:**
- `dl.compile(target="cpu")` ignores the traced graph. A GELU MLP, a scaled-logits MLP and a 3-layer MLP all get the
  hard-coded 2-layer ReLU MLP (losses 2.3137 / 2.3137 / 4.1890 vs reference 2.3090 / 2.3765 / 2.3049), with no error.
- Running the C++ pipeline end to end (parse → fusion → plan → lower → LLVM backend → execute) on
  `tests/lit/fusion.dlc` gives a max error of 4.7e7: the fused kernel's output buffer lies inside the weight buffer it
  is still reading.
- MNIST, 2 epochs: `cpu` target 31.6 s / 96.88%; reference interpreter 72.6 s with the system NumPy (netlib BLAS) and
  13.6 s with an OpenBLAS NumPy.
- Same AVX2 GEMM, fused vs unfused epilogue: 1.07×. OpenBLAS runs the layer GEMM in 0.69 ms vs 7.41 ms.
- 79% of the `cpu` step is the scalar backward `matmul_transA` (31.5 ms vs 6.6 ms for the vectorized forward GEMM).
- Char-GPT: the 40-step demo ends near the unigram baseline (3.31 nats); a seeded 2,000-step run reaches 2.10 nats
  held-out (bigram baseline 2.45).
- The generated CUDA softmax reduces only warp 0, so rows longer than 32 are wrong (a 65-wide row sums to 2.93).
- Release unit tests were built with `-DNDEBUG`, so every `assert` was compiled out.

**What runs today.** Two execution paths exist, and neither goes through the C++ compiler:

- `target="reference"` executes the traced Python graph in the NumPy interpreter, with reverse-mode autodiff.
- `target="cpu"` runs hand-written AVX2 LLVM IR kernels (`python/minidlc/llvm_engine.py`) for a fixed two-layer ReLU MLP.

The C++ compiler is exercised by `dlc-opt` (IR dumps after each pass, loop IR, LLVM and CUDA text) and by the tests.
