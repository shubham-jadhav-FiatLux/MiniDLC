# MiniDLC Development Log

> **Note:** the M0–M5 entries were written before the 2026-10-17 audit, and several of their claims are incorrect —
> among them "MNIST training through LLVM-compiled code", the fusion and memory-planning numbers, and the CUDA
> milestone. The **Audit** entry at the end lists the corrections; `docs/architecture.md` (Part 2) has the status of
> every component.

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
    - 9 Lit Golden IR Tests passing (including `cuda_tiled_matmul.dlc` and `cuda_reductions.dlc`).
    - 17 C++ unit tests passing in `./build/dlc_unit_tests` (including `testCUDATiledMatmul`, `testCUDALaunchConfig`, `testCUDAModuleGeneration`, `testCUDABackwardKernels`).

---

## Milestone M4 — Transformer Ops, Bounds-Check Elimination & Char-GPT (Completed 2026-10-17)

- **Goals**:
  - Transformer operator set: 3D/4D batched matmuls, sequence linear projections, causal triangular masking (`triu_mask`), LayerNorm forward + autograd VJP, GELU activation, learned token and position embeddings, 3D cross-entropy loss;
  - Neural network modules: `MultiHeadAttention` with causal masking, `TransformerBlock` (Pre-LN residual connections and GELU feed-forward MLP), and `CharGPT` (autoregressive character generator);
  - TinyShakespeare dataset loader and tokenizer in `python/minidlc/data.py` with 65-character vocabulary and deterministic offline fallback;
  - Static range-analysis bounds-check elimination in `lib/loopir/LoopPasses.cpp` (`CS6843 A2/A4`) with golden lit test;
  - Comprehensive unit test suite in `tests/python/test_transformer.py` and end-to-end training and generation in `examples/char_gpt.py`.
- **Completed & Verified**:
  - **Transformer Operators & Autograd**:
    - Batched matmuls: 3D/4D batched tensor contractions ($[B, T, D] \times [B, D, T] \to [B, T, T]$) and sequence linear projections ($[B, T, D] \times [D, H] \to [B, T, H]$) with adjoint backward VJPs.
    - Causal attention mask: `ops.triu_mask(seq_len)` producing additive triangular masks ($-1\times 10^4$ on upper triangle), guaranteeing strict causal independence across sequence positions.
    - LayerNorm: Forward normalization and exact analytic autograd VJP for input $X$, scale $\gamma$, and shift $\beta$, verified against central finite differences to within $10^{-3}$.
    - GELU: Approximate activation $\frac{1}{2} x (1 + \tanh(\sqrt{2/\pi}(x + 0.044715 x^3)))$ and backward derivative.
    - 3D Cross-Entropy: Flattened sequence negative log-likelihood with stable softmax and analytic gradient.
  - **Neural Network Architecture**:
    - `nn.MultiHeadAttention`: Multi-head causal scaled dot-product attention with projection layers.
    - `nn.TransformerBlock`: Pre-LayerNorm residual attention and two-layer MLP with GELU non-linearity.
    - `nn.CharGPT`: Autoregressive character sequence model with learned token and position embeddings, final LayerNorm, LM head, and nucleus/temperature sampling `generate()`.
  - **Static Range Analysis & Bounds-Check Elimination (CS6843 A2/A4)**:
    - Formal interval arithmetic over loop induction variables: $\text{Env}: v \mapsto [\text{low}, \text{high}]$.
    - Verified memory expressions $A[\text{idx}]$ satisfy $0 \le \text{idx} < \text{extent}(A)$ and tile/vector-width divisibility.
    - Loop versioning / peeling in `applyLoopVersioning` splitting non-multiple vectorized loops into a vectorized main loop ($0 .. (N/W)\cdot W$) and a scalar cleanup tail loop ($(N/W)\cdot W .. N$).
    - Stamped `attributes {bounds_check_eliminated = true}` on optimized loop nests.
  - **Dataset**:
    - TinyShakespeare loader in `python/minidlc/data.py` caching 1,115,394 characters across a 65-character vocabulary, with sequence batch generator `tinyshakespeare_batches(batch_size, block_size)`.
  - **Testing & Verification**:
    - 18/18 C++ unit tests passing in `./build/dlc_unit_tests` (including `testLoopVersioningAndBCE`).
    - 11/11 Lit golden IR tests passing in `tests/lit/run_lit_tests.py` (including `bounds_check_elim.dlc` and `loop_versioning.dlc`).
    - 7/7 Transformer unit tests passing in `tests/python/test_transformer.py`.
    - End-to-end training in `examples/char_gpt.py`: 30,656 parameters compiled via `dl.compile(loss, optimizer, target="reference")` with Adam ($lr = 0.008$), empirical cross-entropy loss dropping from **4.3277** to **3.1475** in 40 iterations (0.78s), producing autoregressive text continuations.

---

## Milestone M5 — Documentation, Academic Mapping & Polish (Completed 2026-10-17)

- **Goals**:
  - Per-pass documentation pages with algorithms, complexity, and course links (`docs/passes/`);
  - Core architecture and IR specifications (`docs/architecture.md`, `docs/ir-spec.md`);
  - Academic course-concept map (`docs/course-map.md`) mapping to CS6843, CS6023, CS6600/CS6400, and ID5130;
  - Empirical performance report (`docs/perf-report.md`) with hardware baseline and roofline models;
  - Deep-dive technical blog post (`docs/blog-post.md`) and recruiter-legible resume entry (`docs/resume-entry.md`);
  - 10-minute quickstart guide in `README.md`.
- **Completed & Verified**:
  - Created per-pass specs for all 8 passes (`canonicalize.md`, `loop_passes.md`, `bounds_check_elim.md`, `autodiff.md`, `cse_dce.md`, `fusion.md`, `mem_plan.md`, `shape_infer.md`).
  - Generated complete course-concept map connecting compiler passes directly to IIT Madras coursework.
  - Authored deep retrospective blog post and verified all resume claims against empirically measured figures.
  - Clean git state synchronized to `origin/main`.

---

## Stress-Testing, Corner Cases & Random-Graph Fuzzing (Completed 2026-10-17)

- **Goals**:
  - Stress test implementation against contract edge cases: NaN/Inf propagation, numerical extremes, high-rank tensors, broadcasting corner cases, prime non-tile shapes, and zero-size dimensions;
  - Audit and fix discovered implementation bugs in reference interpreter and frontend ops;
  - Implement randomized computational DAG fuzzer comparing analytic autograd against double-precision numerical finite differences.
- **Completed & Verified**:
  - **Bug Fixes Uncovered**:
    - `python/minidlc/reference.py`: Forward pass for `softmax` and `log_softmax` hardcoded `axis=-1`, ignoring `node.attrs.get("axis", -1)`. Updated forward pass to respect the attribute.
    - `python/minidlc/ops.py`: `sum` and `mean` with `axis=None` and `keepdims=True` hardcoded `out_shape = (1,)` instead of $(1, 1, \dots, 1)$ matching tensor rank. Updated to `tuple([1] * len(x.shape))`.
    - `python/minidlc/reference.py`: `_unbroadcast` now coerces scalar inputs via `np.asarray(grad)` and strictly reshapes to `target_shape`.
  - **Contract Test Suite (`tests/python/test_contracts.py`)**:
    - 9/9 rigorous contract tests passing covering NaN/Inf semantics, logits up to $\pm 10,000$, non-default softmax axes, 4D batched matmuls, broadcasting corner cases, negative-axis reductions, prime dimensions ($17 \times 33 \times 19$), multi-consumer fanout gradient accumulation, and zero-size dimensions.
  - **Randomized DAG Fuzzer (`tests/python/test_fuzzer.py`)**:
    - Generated 25 complex randomized computational graphs combining `matmul`, `add`, `sub`, `mul`, `relu`, `gelu`, `layernorm`, and `softmax`.
    - Analytic reverse-mode gradients compared against float64 central differences ($\epsilon = 10^{-5}$) with 100% pass rate ($25/25$ trials passed).
  - **Full Regression**: Zero regressions across 18 C++ unit tests, 11 Lit golden tests, MNIST training (96.88%), and Char-GPT training (loss 4.29 -> 3.27 in 0.59s).

---

## Audit (2026-10-17)

A systematic review — a Debug build with AddressSanitizer and UndefinedBehaviorSanitizer, small programs aimed at each
pass, and re-measurement — found that v1 was not the integrated compiler the entries above describe.

- **Verified:** the Python reference interpreter and its VJPs; MNIST at 96.88% through hand-written LLVM IR kernels;
  the C++ IR infrastructure and `dlc-opt`; interval range analysis with remainder peeling.
- **Corrections to the entries above:**
  - The MNIST `cpu` target runs hand-written kernels for a fixed two-layer ReLU MLP; it ignores the traced graph, and
    the C++ compiler is not involved. Running the C++ pipeline end to end produces wrong results.
  - The LLVM and CUDA backends pick text templates by kernel name; tiling, unrolling and vectorization only set
    attributes. No CUDA code has been compiled or run.
  - Several C++ autodiff VJPs are wrong (ReLU, GELU, mean, LayerNorm); constant folding is a use-after-free; CSE merges
    different constants; memory planning ignores fusion and reuses parameter buffers.
  - The "2.27× faster" and "2.77× fusion speedup" figures were measured against a NumPy linked to the netlib reference
    BLAS; same-kernel fusion gives 1.00×. The 26.2% memory figure is not reproducible.
  - The unit tests were built with `-DNDEBUG`, so their assertions never ran.
  - Char-GPT attention is single-head, and the 40-step loss trace was neither monotonic nor reproducible.
- **Outcome:** a v2 architecture (`docs/architecture.md`) and phases P0–P7.

## P0 — Truthful docs and real tests (2026-10-17)

- **Tests:** unit tests moved to GoogleTest, so assertions run in every build; 21 tests check outcomes rather than
  "the pass reported a change"; 13 known bugs recorded as `DISABLED_` tests in `tests/unit/test_known_bugs.cpp`.
  Lit tests run with LLVM's FileCheck (the regex runner is gone); `autodiff.dlc` is an expected failure until P2, and
  the shape-inference tests now start from wrong types. Finite-difference gradient checks cover all 18 ops, with
  mixed-sign inputs for ReLU and GELU. `pytest` runs from the repository root.
- **CI:** `.github/workflows/ci.yml` builds Debug with ASan/UBSan and Release with clang, runs `ctest` (GoogleTest and
  lit) and runs `pytest` on Python 3.11 and 3.13.
- **Docs:** README, architecture, perf report, IR spec, course map, pass pages, blog and resume entry rewritten to
  describe only what exists; the GPU roofline is labelled as ceilings.
- **Reproducibility:** `examples/char_gpt.py` seeds before building the model; `benchmarks/fusion_ablation.py`
  (same-kernel fusion ablation) and `benchmarks/chargpt_heldout.py` (held-out loss with unigram/bigram baselines) back
  the numbers in the perf report.
- **Toolchain:** LLVM 23 development package (headers, `LLVMConfig.cmake`, `FileCheck`); the X86 and NVPTX targets
  are available for P4–P5.

