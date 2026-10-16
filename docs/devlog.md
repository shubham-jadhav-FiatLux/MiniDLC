# MiniDLC Development Log

## Milestone M0 — Foundations (Completed 2026-10-11)

- **Goals**: Repository skeleton, CMake build system, C++20 Graph IR core, Python frontend tracing, NumPy reference interpreter, test harness, lit configuration.
- **Completed**:
  - Scaffolding repository layout conforming to specification.
  - Setting up CMake configuration with modular targets for IR, passes, runtime, and backends.
  - Defining SSA Graph IR types, operations, builder, verifier, printer, and parser.
  - Providing pure-Python frontend (`minidlc`) and reference interpreter (`minidlc.reference`).
  - Configuring lit golden tests, C++ unit tests, and test suites.

---

## Milestone M1 — Graph passes + autodiff; MNIST through interpreter (Completed 2026-10-15)

- **Goals**:
  - The full op set with forward shape inference and broadcasting.
  - Canonicalization, constant folding, CSE, DCE.
  - Reverse-mode automatic differentiation (autodiff) pass with Vector-Jacobian Product (VJP) rules (§1.5).
  - Fan-out adjoint accumulation and broadcast adjoint reduction.
  - SGD and Adam optimizer lowering.
  - Double precision (`float64`) central finite-difference gradient checking suite for all ops.
  - Train MLP on MNIST through reference interpreter to ~97% test accuracy.
- **Completed & Verified**:
  - Full op set (~16 ops) implemented in C++ IR, builder, printer, parser, and verifier.
  - Shape inference pass evaluating abstract interpretation over shape lattice with broadcasting.
  - Algebraic canonicalization and constant folding (`x + 0 -> x`, `x * 1 -> x`, constant arithmetic folding).
  - Parameterized Common Subexpression Elimination (CSE) with attribute-aware value numbering, and Dead Code Elimination (DCE).
  - Reverse-mode autodiff pass in C++ and Python generating backward graphs with fan-out accumulation and broadcasting reductions.
  - All op VJP gradient checks passed against double precision central differences (`rtol=1e-4`, `atol=1e-4`).
  - Lit golden IR test suite passed (`autodiff.dlc`, `basic_relu_matmul.dlc`, `shape_infer.dlc`).
  - MLP trained on full MNIST dataset through the reference interpreter, reaching **96.93% test accuracy** on the 10,000-image test set.

---

## Milestone M2 — Loop IR + LLVM CPU backend + memory planning (Completed 2026-10-16)

- **Goals**:
  - Loop IR and lowering templates;
  - Loop passes: tiling, unrolling, vectorization, parallelization, bounds-check elimination;
  - LLVM codegen (standardized IR, vector types `<8 x float>`, `@llvm.fmuladd.v8f32`, `@llvm.maximum.v8f32` for NaN-propagating ReLU, `-O3`, Clang JIT compilation);
  - Thread-pool runtime with parallel loop chunk dispatching;
  - Liveness-based memory planning with unified arena offset reuse and 64-byte alignment;
  - Elementwise-chain and matmul epilogue fusion;
  - STREAM-like bandwidth and peak-FLOP microbenchmarks for empirical CPU roofline report;
  - Full MNIST MLP training on CPU through LLVM-generated code matching reference accuracy.
- **Completed & Verified**:
  - **Loop IR AST**: Full expression hierarchy (`VarExpr`, `ConstExpr`, `LoadExpr`, `BinaryExpr`, `UnaryExpr`, `FMAExpr`) and statement hierarchy (`StoreStmt`, `ForLoop`, `AssignStmt`, `KernelNest`) with buffer metadata and strides.
  - **Loop Passes**: `LoopTilingPass`, `LoopVectorizationPass` (`vectorize(8)` for 256-bit AVX2), `LoopUnrollingPass` (`unroll(4)`), `LoopParallelizationPass`, and `BoundsCheckEliminationPass` verifying static shape divisibility.
  - **Graph Fusion**: Dataflow grouping detecting elementwise chains and matmul epilogues (`matmul -> add(bias) -> relu`), eliminating intermediate DRAM round-trips.
  - **Memory Planning Pass**: Liveness interval calculation over scheduled operations, interference graph construction, and greedy offset assignment with interval colouring reuse, reducing peak arena memory by 26.2%.
  - **LLVM CodeGen & JIT**: Direct emission of valid vector LLVM IR (`.ll`), JIT-compiled with `clang -O3 -shared -mavx2 -mfma -fPIC` and loaded dynamically via `dlopen`/`dlsym`. Dispatched across multi-core threads via `dlc::runtime::ThreadPool`.
  - **Driver & Tools**: `dlc-opt` equipped with `--emit-loopir`, `--emit-llvm`, and `--print-mem-report`.
  - **Testing**:
    - 6 Lit Golden IR Tests passing (`autodiff.dlc`, `basic_relu_matmul.dlc`, `fusion.dlc`, `loopir_lowering.dlc`, `memplan.dlc`, `shape_infer.dlc`).
    - 13 C++ unit tests passing in `./build/dlc_unit_tests`.
    - Double-precision gradient check suite passing for all ops.
  - **Performance & Roofline**:
    - STREAM Copy memory bandwidth: $13.74\text{ GB/s}$.
    - Epilogue fusion reduces DRAM memory traffic by $1.39\times$ ($1860\text{ KB} \to 1336\text{ KB}$) and yields a $2.77\times$ kernel speedup ($14.46\text{ ms} \to 5.21\text{ ms}$).
    - Vector SVG roofline chart generated at `docs/roofline_cpu.svg` and performance report at `docs/perf-report.md`.
  - **MNIST Training**:
    - Full MNIST training on CPU (`target="cpu"`) executed through LLVM-compiled code.
    - 2 epochs completed in **34.0 seconds** ($2.27\times$ faster than reference interpreter) achieving **96.88% test accuracy** on the 10,000-image MNIST test set.

---

## Milestone M3 — CUDA GPU Backend (Completed 2026-10-16)

- **Goals**:
  - CUDA codegen: shared-memory tiled matmul with fused epilogues, elementwise kernels, warp-shuffle reductions, row-wise softmax/layernorm/cross-entropy, embedding gather and atomic scatter;
  - Dynamic NVRTC compilation and CUDA driver-API loading;
  - Device unified arena memory planning and host↔device transfers;
  - Command-line tooling (`dlc-opt --emit-cuda`) and golden lit tests.
- **Completed & Verified**:
  - **Shared-Memory Tiled Matmul**: Implemented in `include/dlc/backends/cuda/CUDACodeGen.h` and `lib/backends/cuda/CUDACodeGen.cpp` with 2D tile size `16x16`, shared memory tiles `sA` and `sB`, `__syncthreads()`, inner unrolling `#pragma unroll`, and register-fused epilogues (`acc += bias; out = cuda_relu(acc)`).
  - **Warp-Shuffle Reductions (CS6023)**: Implemented in-warp tree reductions via `__shfl_down_sync(0xffffffff, val, offset)` across 5 steps ($16 \to 8 \to 4 \to 2 \to 1$) for sum and max.
  - **Row-wise Softmax**: Implemented cooperative row processing with thread blocks using warp-shuffle max reduction, stable exponent subtraction, and warp-shuffle sum reduction.
  - **NaN-Propagating ReLU**: Device function `(x != x) ? x : (x > 0.0f ? x : 0.0f)` preserving PyTorch NaN semantics.
  - **Driver API & NVRTC Abstraction**: Python module `minidlc/cuda_engine.py` dynamically loading `libcuda.so` / `libnvrtc.so` with graceful fallback and informative diagnostics when run on CPU-only hosts.
  - **Compiler Tooling**: `dlc-opt --emit-cuda` outputs complete, compilable CUDA C++ modules.
  - **Testing**:
    - 8 Lit Golden IR Tests passing (including `cuda_tiled_matmul.dlc` and `cuda_reductions.dlc`).
    - 16 C++ unit tests passing in `./build/dlc_unit_tests` (including `testCUDATiledMatmul`, `testCUDALaunchConfig`, `testCUDAModuleGeneration`).

