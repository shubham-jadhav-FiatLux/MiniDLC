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

## P1 — IR contract between Python and C++ (2026-10-18)

- **Op registry** (`lib/ir/OpRegistry.cpp`): operand count, attributes with kinds and defaults, and a type rule for each
  of the 21 ops. The builder, parser, verifier and shape inference all use it; `dlc-opt --print-op-registry` lists it.
- **Parser and printer:** a recursive-descent parser with a lexer that reports `line:col` errors (undefined or redefined
  values, unknown ops and attributes, wrong attribute kinds, arity, malformed tokens, oversized types). It reads
  attributes, argument roles (`input`, `param`, `const`, `state`), multi-value returns in both forms, comments and
  dotted annotations, and infers omitted types. The printer is canonical: results renumbered, defaults written out,
  attributes sorted, floats laid out like Python's `repr`. Print → parse → print is a fixpoint.
- **Verifier:** registry type rules and attributes, dominance, a single trailing `return` matching the signature, and
  use lists; `dlc-opt` verifies its input and the pass manager verifies after every pass, naming the pass that broke the
  module.
- **Use-list hygiene:** erasing an op unlinks it from its operands and refuses ops whose results are still used;
  `replaceAllUsesWith` replaces the per-pass helpers. Constant folding walks a snapshot of the op list (it was a
  use-after-free). Kernel buffers in the loop IR are named by the printer's slot names, so unnamed values no longer
  collide.
- **Python side** (`python/minidlc/ir.py`): `export` writes a traced graph as canonical IR (parameters named after the
  module attributes, e.g. `%block_0.attn.q_proj.weight`); `parse_module` reads `dlc-opt` output; `execute` runs it in
  the reference interpreter. `examples/export_ir.py` exports the MNIST MLP or Char-GPT for `dlc-opt`.
- **Tests:** 55 GoogleTest tests (round trips of every op, a table of malformed inputs, a mutation fuzzer that runs
  2,000 mutated modules through the parser, verifier and passes, registry rules, verifier checks, pass-manager errors),
  16 lit tests, and translation validation: each graph pass on 16 exported graphs (softmax regression, MLP, Char-GPT,
  ten random DAGs and more), executed before and after in float64. Five known bugs are fixed and moved into the regular
  tests; the translation-validation suite catches two more miscompiles (CSE merging nearby constants, `x * 0` changing
  a result type), recorded as strict expected failures for P2.

## P2 — The training graph in C++ (2026-10-18)

- **Backward primitives** in the op registry: `cmp_gt`, `select`, `broadcast_to`, `reduce_max`, `scatter_add`, `cast`,
  `tanh`, `sqrt`, `rsqrt`, and `transpose_a`/`transpose_b` flags on `matmul` (written only when set).
- **Activity analysis** (`lib/passes/Activity.cpp`): varied (forward from the params) ∩ useful (backward from the loss),
  along differentiable edges. No gradient is built for the input batch.
- **Autodiff** rewritten: VJPs for every differentiable op — ReLU as `select(cmp_gt(x, 0), dy, 0)`, GELU with `tanh`,
  LayerNorm with the full normalization Jacobian, reductions broadcast back with `broadcast_to`, cross-entropy divided
  by the number of rows, embedding through `scatter_add`, matmul gradients as matmuls with transpose flags (3-D × 2-D
  flattens the batch). Gradients are appended as `{grad = %param}` results.
- **Optimizer lowering** (`adam{...}`, `sgd{...}`): the update is part of the graph; moments and bias-correction powers
  are state arguments with an `init`, and new parameters and state are `{donates = %arg}` results. One graph is one
  training step.
- **Canonicalize** to a fixpoint: type-preserving, NaN-safe (no `x * 0 → 0`), constant folding of splats, transposes
  folded into matmul flags, transpose/reshape/broadcast chains collapsed, commutative operands ordered. **CSE** keys on
  operand identity, result type and exact attribute bits.
- **IR:** `{state, init = 0.0}` on arguments, `{grad = %w}` / `{donates = %w}` on results; globally unique value ids;
  the builder can insert before an op. Python's `ir.py` reads and prints all of it, and `ir.StepRunner` runs a step
  function repeatedly.
- **Tests:** 74 GoogleTest tests; 17 lit tests (`autodiff.dlc` now passes, `optimizer.dlc` is new); 41 new Python
  tests — every C++ VJP rule against float64 central finite differences (37 graphs) and the step graph of an MLP and a
  Char-GPT under Adam and SGD with momentum against eager training (losses equal to 1e-9 over four steps). The five
  P2 known bugs are fixed and moved into the regular tests; the two miscompiles translation validation caught are
  fixed too.

## P3 — Fusion and buffer assignment (2026-10-18)

- **Fusion** rewritten around kernel classes: a matmul with an elementwise epilogue (any elementwise op of its output
  shape that doesn't feed it), elementwise kernels with any number of outputs, and kernels of their own for row ops,
  reductions, gathers, scatters, transposes and constants read from memory. A merge is legal only if the kernel graph
  stays acyclic — checked on the quotient graph, since checking paths between single ops misses cycles between
  multi-op groups — and kernels are ordered topologically.
- **Cost model:** FLOPs and compulsory bytes per op and per kernel; `dlc-opt --print-cost`.
- **Buffer assignment:** params, optimizer state and consts in a persistent region; inputs, kernel outputs and results
  in a transient arena placed by interval colouring over per-kernel live ranges, 256-byte aligned; reshapes alias
  their operand; donated updates are written in place when the writing kernel reads the parameter only at the element
  it writes and runs after every other reader (else copied at the end). A checker verifies every assignment.
- **Measured** on the training step (`benchmarks/memory_plan.py`): the MNIST MLP step is 15 kernels with a 2.33 MiB
  persistent region, a 0.76 MiB transient arena and all 14 updates in place; fusion cuts its compulsory traffic from
  31.4 MB to 7.6 MB. Char-GPT: 157 kernels, arena 63% smaller than without reuse.
- **Removed from the build:** the v1 loop IR, lowering templates and LLVM/CUDA text-template backends, which depended
  on the old fusion groups; P4 and P5 replace them. The two P3 known bugs are fixed and tested; the v1 LLVM division
  bug went with its emitter and is tracked for P4 by an expected-failure lit test.
- **Tests:** 75 GoogleTest tests (fusion legality, cost model, buffer assignment; the fuzzer checks every mutant's
  assignment), 12 lit tests, 243 Python tests.

## P4 — The CPU backend (2026-10-29)

- **Loop IR v2** (`lib/loopir/LoopIR.cpp`): loops with a kind (serial, parallel, vectorize, unroll, bound), buffers with
  a scope, explicit guards, barriers and atomics, typed expressions with folding, a printer.
- **Lowering and CPU schedules** (`lib/loopir/Lowering.cpp`), one per kernel class: a 4×16 register-tiled matmul
  (outer-product form; dot-product form for a transposed B) with the epilogue applied to the tile in registers,
  partial tiles as separate branches; elementwise kernels over collapsed dimensions; row kernels; three reduction
  layouts; guarded gathers and scatters; transposes and fills.
- **Loop passes** (`lib/loopir/Analysis.cpp`): range analysis with branch refinement; bounds checking, with guards for
  indices read from memory (any input is memory-safe) and an error for any other unprovable access; bounds-check
  elimination; a dependence check that parallel iterations write disjoint elements and vectorized loops carry no
  memory dependence.
- **LLVM backend** (`lib/backends/llvm/`): `IRBuilder` emission with `noalias` buffers, emitter-side unrolling,
  `llvm.loop.vectorize` metadata, `reassoc` only on reduction updates, `llvm.maximum` for NaN-propagating ReLU; O3 via
  `PassBuilder` with the host TargetMachine; ORC LLJIT.
- **Runtime and bindings:** a backend-neutral plan (slots, launches, final copies), an executable with a persistent
  region and a transient arena, a chunking thread pool, per-kernel timers, a loop IR interpreter target, and the
  nanobind module `_minidlc`. `dl.compile(loss, optimizer, target="cpu")` compiles the whole training step.
- **Results:** MNIST trains 2 epochs through generated code in about 1–2 s to 96.8–96.9% (the hand-written v1 kernels
  took 31.6 s, the NumPy interpreter 5.4 s with OpenBLAS); the layer GEMMs run at 65–95 GFLOP/s. Char-GPT follows the
  reference loss trajectory step for step.
- **Tests:** 89 GoogleTest tests (loop IR, analyses, JIT, compiler), 16 lit tests (loop IR, LLVM IR, guards, tiles,
  memory plan; the LLVM division bug test passes), 63 differential tests of every op through the generated code, 10
  compiled-vs-eager training tests.

## P5 — The GPU backend (2026-10-29)

- **GPU schedules** (`lib/loopir/Lowering.cpp`): loops bound to `blockIdx`/`threadIdx`; matmuls in 64×64 output tiles
  per 16×16-thread block, k staged through shared memory 16 at a time (the A tile transposed and padded to 65 columns),
  a 4×4 register tile per thread with interleaved columns so that a warp reads consecutive words, and the epilogue
  applied in registers; elementwise, gather, transpose and fill kernels with a thread per element; row kernels
  (softmax, log-softmax, layernorm, cross-entropy) and suffix reductions with a warp per row and butterfly shuffles;
  scatters and the loss with a fill kernel plus atomic adds. Edge guards are emitted where the grid overshoots, and
  bounds-check elimination removes the matmul guards when M and N are multiples of 64 and K of 16.
- **Lockstep GPU interpreter** (`lib/loopir/Interpreter.cpp`): the threads of a block execute statement by statement;
  `if` splits the active set; a barrier reached by part of a block is an error, as is a shuffle that reads an inactive
  lane; shared memory starts as NaN. Blocks run on the host thread pool (atomics with `std::atomic_ref`). It is the
  `gpu-sim` target.
- **NVPTX emission** (`lib/backends/llvm/Emitter.cpp`, `Ptx.cpp`): `ptx_kernel` functions over `ptr addrspace(1)`
  buffers, shared buffers as addrspace(3) globals, register tiles as allocas that SROA scalarizes, `bar.sync`,
  `shfl.sync.bfly`, and the NVPTX TargetMachine's O3 pipeline and AsmPrinter. NVPTX has no libm, so `exp`/`log` use
  `ex2.approx`/`lg2.approx` (like CUDA's `__expf`/`__logf`) and `tanh` goes through `exp(-2|x|)`. LLVM 23 expands
  `atomicrmw fadd` into a compare-and-swap loop on every target tried, so float atomics are emitted as
  `red.global.add.f32`.
- **Artifacts and runtime:** `compileGpu` returns the plan and the PTX; the plan serializes to JSON. `minidlc/gpu.py`
  replays a plan through the CUDA driver API (ctypes; no CUDA toolkit or compiled module needed), or on a simulated
  device backed by the lockstep interpreter, which tests the runtime itself. Artifacts (plan, PTX, initial values,
  source graph) are saved to a directory; `examples/colab_gpu.py` runs the committed MNIST artifacts on a GPU and
  checks them against the NumPy reference.
- **Results:** every op matches the reference on `gpu-sim` (the 63 differential tests run on all three targets), and
  MLP and Char-GPT training trajectories match eager training. ptxas (sm_75) accepts every kernel with no spills; the
  matmul kernels use 64 registers and 8.25 KiB of shared memory. **Nothing has run on a GPU yet.**
- **Tests:** 95 GoogleTest tests (GPU interpreter semantics, guard elimination), 19 lit tests (three GPU tests:
  tiled matmul, warp reductions, a training step with atomics; loop IR, LLVM IR and PTX), 8 GPU runtime tests.

## P6 — The Transformer on both targets (2026-10-29)

- **Multi-head attention:** `MultiHeadAttention` now splits heads — projections reshaped to [B, T, H, Dh] and
  transposed to [B, H, T, Dh], 4-D batched matmuls for the scores and the context (canonicalization folds the key
  transpose into the matmul's `transpose_b`), heads merged back before the output projection. A test checks it
  against an explicit NumPy multi-head implementation and that it differs from single-head attention. Nothing else in
  the compiler changed: 4-D matmuls, reshapes and transposes were already supported on both targets.
- **PyTorch reference** (`benchmarks/torch_chargpt.py`): the same function in PyTorch ops, with MiniDLC's weights.
  In float64 the reference interpreter's loss and every gradient match PyTorch's to 1e-7; the compiled step's Adam
  trajectory and final weights match PyTorch's on `cpu` and `gpu-sim`. The one exception is the key projection's
  bias: softmax ignores it, so its gradient is zero up to rounding, which Adam amplifies differently in each
  implementation.
- **Held-out evaluation** (`benchmarks/chargpt_heldout.py`): trains on any target — the compiled step with a compiled
  evaluation function that receives its weights, PyTorch, or the reference — from the same weights on the same
  batches; n-gram baselines are now fitted on the training split and evaluated on the held-out split. 2,000 steps
  (d_model 64, 4 heads, 2 layers): `cpu` 1.9306 nats, PyTorch 1.9328, bigram 2.4819; the curves never differ by more
  than 0.0022.
- **GPU artifacts** for Char-GPT (d_model 128, 4 heads, 4 layers, context 64): a 316-kernel training step and a
  67-kernel evaluation function; evaluation artifacts carry no weights. The GPU run waits for Colab.

