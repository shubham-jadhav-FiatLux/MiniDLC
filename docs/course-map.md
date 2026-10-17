# Course-Concept Mapping

This document provides a comprehensive mapping between academic coursework and the architectural implementation of **MiniDLC**. Every major compiler pass, runtime optimization, and hardware backend directly exercises concepts from graduate computer science and systems coursework.

---

## 1. Summary Map

| Concept | Location in MiniDLC | Course Reference |
|---|---|---|
| **SSA Form, IR Design & Type Verification** | `include/dlc/ir/`, `lib/ir/` | **CS6843 Program Analysis** (L1–L4, LLVM Primer) |
| **Dataflow Analysis: Available Expressions (CSE)** | `lib/passes/CSE.cpp` | **CS6843 Program Analysis** (L6 Dataflow Frameworks) |
| **Dataflow Analysis: Liveness & Reverse Reachability (DCE)** | `lib/passes/DCE.cpp` | **CS6843 Program Analysis** (L7 Liveness & Dead Code) |
| **Abstract Interpretation (Shape Lattice)** | `lib/passes/ShapeInference.cpp` | **CS6843 Program Analysis** (L14 Abstract Interpretation) |
| **Interval Range Analysis & Bounds-Check Elimination** | `lib/loopir/LoopPasses.cpp` | **CS6843 Program Analysis** (Assignment A2 & A4) |
| **Dependence Analysis & Fusion Legality** | `lib/passes/Fusion.cpp` | **CS6843 Program Analysis** (Loop Parallelism & Dependences) |
| **Lit / FileCheck Golden Testing Workflow** | `tests/lit/` | **CS6843 Program Analysis** (LLVM Developer Tooling) |
| **Register Allocation Analogy: Liveness Memory Planning** | `lib/passes/MemoryPlanning.cpp` | **CS6843 Program Analysis** (Chaitin-Briggs Interval Coloring) |
| **Shared-Memory 2D Tiling & Bank Conflicts** | `lib/backends/cuda/CUDACodeGen.cpp` | **CS6023 GPU Programming** (L3 CUDA Memory Hierarchy) |
| **Intra-Warp Tree Reductions (`__shfl_down_sync`)** | `lib/backends/cuda/CUDACodeGen.cpp` | **CS6023 GPU Programming** (L8 Warp Primitives & Shuffles) |
| **Hardware Atomic Scatter Updates (`atomicAdd`)** | `lib/backends/cuda/CUDACodeGen.cpp` | **CS6023 GPU Programming** (L4 Synchronization & Atomics) |
| **Global Memory Coalescing & Transposition** | `lib/backends/cuda/CUDACodeGen.cpp` | **CS6023 GPU Programming** (L3 DRAM Coalescing) |
| **Cache Blocking & Loop Tiling** | `lib/loopir/LoopPasses.cpp` | **CS6600 Computer Architecture** (L10 Cache Hierarchies) |
| **Instruction-Level Parallelism (ILP) & Unrolling** | `lib/loopir/LoopPasses.cpp` | **CS6600 Computer Architecture** (L5 Dynamic Scheduling & ILP) |
| **SIMD Vectorization (256-bit AVX2)** | `lib/backends/llvm/LLVMCodeGen.cpp` | **CS6600 / CS6400 Computer Architecture** (SIMD Execution) |
| **Roofline Performance Modeling** | `benchmarks/roofline.py`, `docs/perf-report.md` | **CS6400 ACA / ID5130 PSC** (Arithmetic Intensity) |
| **Thread-Pool Runtime & Static Chunking** | `include/dlc/runtime/ThreadPool.h` | **ID5130 Parallel Scientific Computing** (L5 OpenMP & Work-sharing) |
| **Numerical Stability (Log-Sum-Exp, fp32 Accumulation)** | `lib/passes/Autodiff.cpp`, `python/minidlc/reference.py` | **ID5130 Parallel Scientific Computing** (Floating-Point Numerics) |
| **Reverse-Mode Autodiff (Vector-Jacobian Products)** | `lib/passes/Autodiff.cpp` | **Mathematical Foundations** (Chain Rule & Adjoint State Method) |

---

## 2. Course-by-Course Technical Deep Dive

### A. CS6843 Program Analysis
- **SSA Construction & Dominance**: Graph IR maintains static single-assignment (SSA) discipline where every value has a unique defining operation. The verifier checks dominance properties before running optimizations.
- **Dataflow Analysis (DFA)**:
  - *Available Expressions (CSE)*: Computes forward available expressions via op hashing and operand value numbering to merge redundant operations in $O(N)$ time.
  - *Liveness Analysis (DCE)*: Traverses backward from function outputs to identify live sets. Unreferenced nodes without side-effects are purged.
  - *Memory Planning*: Models buffer lifetimes as liveness intervals $[t_{\text{start}}, t_{\text{end}}]$ over scheduled kernel executions. Constructs an interval interference graph and solves memory reuse via greedy coloring, saving **26.2% peak DRAM**.
- **Static Range Analysis & Bounds-Check Elimination (Assignments A2 & A4)**:
  - Propagates interval ranges $\text{Env}: v \mapsto [0, \text{bound}-1]$ across nested induction variables.
  - Statically proves affine index expressions $A[\text{idx}]$ remain within $[0, N-1]$, stamping `bounds_check_eliminated = true`.
  - Implements **loop versioning/peeling** (A4) when dimensions are non-tile multiples, producing an un-guarded vectorized main loop and a scalar cleanup tail loop.

### B. CS6023 GPU Programming
- **Shared Memory Staging**: Tiled matrix multiplications (`matmul`, `matmul_transA`, `matmul_transB`) stage $16 \times 16$ blocks in `__shared__ float sA[16][16]` and `sB[16][16]`, providing **$16\times$ data reuse** and cutting DRAM traffic by $16\times$.
- **Coalesced DRAM Access**: Matrix transposition in backpropagation ($X^T dY$ or $dY W^T$) is performed in on-chip shared memory at zero DRAM cost, maintaining 100% coalesced 32-byte/64-byte global transactions.
- **Warp-Shuffle Primitives (`__shfl_down_sync`)**:
  - Implements in-warp reduction trees for row-wise softmax and cross-entropy across 5 butterfly steps ($16 \to 8 \to 4 \to 2 \to 1$).
  - Eliminates shared-memory latency and `__syncthreads()` barrier stalls.
- **Hardware Atomic Scatter (`atomicAdd`)**: Handles gradient scatter-accumulation in backward embedding lookup tables across multiple thread blocks.

### C. CS6600 / CS6400 Computer Architecture
- **Cache-Aware Blocking**: Tiling passes partition large matrix multiplication iteration spaces into $32 \times 64 \times 32$ blocks sized to fit within L1 ($32\text{ KB}$) and L2 ($256\text{ KB}$) caches.
- **AVX2 Vectorization**: Lowers contiguous loops to 256-bit SIMD instructions (`<8 x float>`), leveraging 8-wide hardware vector ALUs.
- **Loop Unrolling**: Unrolls innermost computation loops by factor 4, hiding ALU latency and maximizing instruction-level parallelism (ILP).
- **Roofline Analysis**: Establishes theoretical hardware bounds (Peak FLOP/s vs Memory Bandwidth) and plots kernel operational points to diagnose compute-bound vs memory-bound regimes.

### D. ID5130 Parallel Scientific Computing
- **Lightweight Thread Pooling**: Multi-threaded CPU execution uses a persistent C++ `std::thread` pool with lock-free atomic task queues, avoiding OpenMP runtime library dependencies.
- **Numerical Stability**:
  - Softmax and cross-entropy evaluate $z - \max(z)$ before exponentiation to eliminate floating-point overflow.
  - Cross-entropy gradient directly computes $(p_i - y_i) / N$ instead of evaluating $1/p_i \cdot \nabla p_i$, preventing catastrophic cancellation and `NaN` propagation.
  - IEEE-754 / PyTorch NaN semantics are maintained across activations (`llvm.maximum` on CPU, `(x != x) ? x : ...` on GPU).
