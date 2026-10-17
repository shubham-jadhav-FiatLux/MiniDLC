# Loop Transformation Passes

## Purpose
Optimizes lowered `KernelNest` iteration spaces to exploit target hardware features on modern CPUs (AVX2 SIMD, caches, hardware multi-threading) and GPUs (shared-memory, warp scheduling).

## Passes & Transformations

### 1. Loop Tiling Pass (`LoopTilingPass`)
- **Mechanism**: Blocks nested loops along iteration dimensions:
  - Default parameters: `tileM = 32`, `tileN = 64`, `tileK = 32`.
- **Architectural Goal**: Maximizes L1/L2 data cache reuse on CPU and shared-memory staging on GPU, reducing high-latency main memory (DRAM) accesses.
- **Course Reference**: **CS6600 Computer Architecture** (Memory hierarchies, cache blocking).

### 2. Loop Vectorization Pass (`LoopVectorizationPass`)
- **Mechanism**: Inspects the innermost contiguous loop and stamps `vectorize(W)` attribute (default width $W=8$ for 256-bit AVX2 single-precision floats).
- **Codegen Impact**: Directs LLVM backend to emit SIMD vector loads (`<8 x float>`), fused multiply-adds (`llvm.fmuladd`), and vector stores (`vmovaps`).
- **Course Reference**: **CS6600 / CS6400 Computer Architecture** (SIMD vector architectures).

### 3. Loop Unrolling Pass (`LoopUnrollingPass`)
- **Mechanism**: Annotates innermost reduction/compute loops with `unroll(U)` (default factor $U=4$).
- **Architectural Goal**: Exposes Instruction-Level Parallelism (ILP), hides instruction latencies, and cuts loop counter/branching overhead.
- **Course Reference**: **CS6600 Computer Architecture** (Instruction-level parallelism, loop scheduling).

### 4. Loop Parallelization Pass (`LoopParallelizationPass`)
- **Mechanism**: Analyzes dependence vectors across iterations of outer loops (e.g. batch or output row loops). When iterations write to disjoint buffer slices, the loop is stamped `parallel`.
- **Runtime Dispatch**: The C++ thread pool splits outer loop ranges across physical CPU cores (`[begin, end)` chunks), eliminating OpenMP runtime overhead.
- **Course Reference**: **CS6843 Program Analysis** (Dependence analysis) and **ID5130 Parallel Scientific Computing** (Parallel scheduling).
