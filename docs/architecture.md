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
