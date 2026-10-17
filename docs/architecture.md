# MiniDLC Architecture Specification

## 1. Overview and Core Philosophy

MiniDLC is structured around clean separation of compiler phases:
1. **Frontend**: Pure Python symbolic tracing using operator overloading, producing an in-memory Graph IR representation.
2. **Graph IR**: SSA-form dataflow representation where high-level operations (`matmul`, `relu`, `layernorm`) operate on statically shaped tensors.
3. **Graph Passes**: Shape inference, canonicalization, common subexpression elimination, dead code elimination, reverse-mode automatic differentiation, operator fusion, and memory planning.
4. **Loop IR**: Lowering from fused tensor graph groups into nested loops with affine index calculations, buffer allocations, and parallel/SIMD attributes.
5. **Backends**:
   - Reference NumPy Interpreter: Ground truth verification engine and float64 gradient checker.
   - LLVM CPU Backend: Loops converted to LLVM IR with SIMD vector instructions and multi-threading runtime dispatch.
   - CUDA GPU Backend: Tiled CUDA C++ emitted and compiled via NVRTC with shared memory and warp shuffle reductions.
6. **Runtime**: Fixed-size arena memory allocator, persistent weights, and kernel launch sequence dispatcher.

## 2. Compilation Pipeline

```
[ Python Frontend ] (x, W, b, loss)
       │
       ▼ Tracing
[ Graph IR (SSA) ]
       │
       ├──► Shape Inference & Broadcasting
       ├──► Canonicalization & Constant Folding
       ├──► CSE (Value Numbering)
       ├──► DCE (Liveness Reachability)
       ├──► Autodiff Pass (Appends VJP backward graph + optimizer update)
       ├──► Fusion Pass (Groups elementwise chains & matmul epilogues)
       └──► Memory Planning (Interval interference coloring for arena offsets)
       │
       ▼ Lowering
[ Loop IR ]
       │
       ├──► Loop Tiling
       ├──► Loop Unrolling
       ├──► Vectorization
       ├──► Parallelization
       └──► Bounds-Check Elimination
       │
       ├──► [ LLVM CPU Backend ] ──► ORC JIT Machine Code
       ├──► [ CUDA GPU Backend ] ──► NVRTC PTX / Driver API
       └──► [ NumPy Reference ]  ──► Eager Validation
```

## 3. Memory Arena Architecture

Instead of dynamic allocations during execution, MiniDLC performs static memory planning:
- Lifetimes of intermediate tensors are computed via liveness intervals over the scheduled execution sequence.
- An interference graph is constructed where an edge connects two values with overlapping lifetimes.
- A greedy offset assignment assigns non-overlapping memory spans within a single contiguous arena.
- Parameters and optimizer state buffers remain resident across training iterations with buffer donation semantics.
- 64-byte alignment guarantees aligned AVX2 vector memory instructions (`vmovaps`).

## 4. Sequence Modeling & Transformer Architecture

MiniDLC supports complete sequence modeling pipelines:
- **Batched Tensor Contractions**: 3D and 4D tensor matrix products ($[B, T, D] \times [B, D, T] \to [B, T, T]$) and sequence linear projections ($[B, T, D] \times [D, H] \to [B, T, H]$).
- **Causal Masking**: Additive triangular masking (`ops.triu_mask`) setting upper-triangular elements to $-1\times 10^4$ prior to softmax, strictly preventing future sequence positions from influencing prior representations.
- **Layer Normalization**: Normalization across feature dimensions with learnable scale ($\gamma$) and shift ($\beta$), backed by analytic backward VJPs in both C++ and Python.
- **GELU Non-linearity**: Smooth tanh-approximated activation $\frac{1}{2} x (1 + \tanh(\sqrt{2/\pi}(x + 0.044715 x^3)))$.
- **Autoregressive Generation**: Iterative character decoding with temperature-scaled softmax sampling.

## 5. Heterogeneous Execution Backends

1. **NumPy Reference Interpreter**:
   - Executes the untransformed or optimized Graph IR directly using double precision (`float64`) for numerical ground-truth and finite-difference gradient verification.
2. **LLVM CPU JIT Backend**:
   - Lowers Loop IR into LLVM IR via `IRBuilder`.
   - Emits 256-bit AVX2 SIMD vector operations (`<8 x float>`), fused multiply-adds (`llvm.fmuladd`), and NaN-propagating activations (`llvm.maximum`).
   - Dispatches parallel loop tiles across physical CPU cores via an integrated C++ `ThreadPool`.
3. **CUDA GPU Backend**:
   - Generates CUDA C++ kernels with 2D shared-memory tiling ($16 \times 16$), on-chip transposition, and register-fused epilogues.
   - Evaluates intra-warp tree reductions via `__shfl_down_sync` register intrinsics across 5 steps ($16 \to 8 \to 4 \to 2 \to 1$).
   - Fuses multi-tensor Adam updates into a single register pass, eliminating redundant DRAM round-trips.
   - Compiles kernels dynamically via NVRTC and launches asynchronously through the CUDA Driver API.

## 6. Verification Engine, Contract Testing & Randomized Fuzzing

MiniDLC uses a multi-tier testing strategy to ensure mathematical correctness, numerical stability, and robustness:
1. **Contract Testing Suite (`tests/python/test_contracts.py`)**:
   - Explicitly validates compiler and runtime invariant contracts:
     - IEEE-754 semantics: NaN and Inf propagation, with PyTorch NaN safety (`grad = 0.0` for `relu(NaN)`).
     - Log-Sum-Exp row-maximum shifting preventing float32 overflow under extreme logit magnitudes ($\pm 10,000$).
     - Multi-axis reductions and shape preservation under `keepdims=True`.
     - 4D batched tensor contractions and non-tile-multiple matrix shapes.
     - Multi-branch consumer fanout gradient accumulation ($\sum_i \frac{\partial L}{\partial y_i} \frac{\partial y_i}{\partial x}$).
2. **Randomized DAG Fuzzer (`tests/python/test_fuzzer.py`)**:
   - An automated graph synthesizer that builds randomized computational DAGs from the operator set.
   - Evaluates graphs in double precision (`float64`) and asserts analytic reverse-mode autograd gradients against two-sided numerical central finite differences:
     $$\frac{f(x + \epsilon) - f(x - \epsilon)}{2\epsilon}, \quad \epsilon = 10^{-5}$$
   - Fuzzes across random seeds to catch corner-case interactions between elementwise ops, reductions, normalizations, and matrix products.


