# MiniDLC — Resume Entry Specification

Following the house style specified in `ResumeProjectsToBuild.md` (§1.12), this document provides exact resume bullets formatted for ML-compiler, GPU-kernel, and systems engineering roles. Every clause begins with an active verb, carries concrete technical nouns, and quotes exclusively empirically measured results from `docs/perf-report.md`.

---

## Resume Entry (House Style: 3 Bullets)

**MiniDLC — Deep-Learning Compiler from Scratch (C++20, LLVM, CUDA, Python)**
- **Architected** an end-to-end deep-learning compiler lowering custom Python neural networks through a two-level intermediate representation (SSA Graph IR and Loop IR) to native AVX2 SIMD via LLVM ORC JIT and CUDA GPU kernels via NVRTC.
- **Implemented** reverse-mode autodiff with analytic VJPs, liveness-based memory planning achieving a **26.2% peak arena reduction** ($1.64\text{ MB} \to 1.21\text{ MB}$), epilogue fusion cutting DRAM traffic by **$1.39\times$** ($1860\text{ KB} \to 1336\text{ KB}$, $2.77\times$ speedup), static range-analysis bounds-check elimination with loop versioning, and $16\times 16$ shared-memory tiled CUDA GEMM with 5-step warp-shuffle reduction trees (`__shfl_down_sync`).
- **Benchmarked** against empirical single-core AVX2 compute peak ($112.5\text{ GFLOP/s}$) and STREAM bandwidth ($14.09\text{ GB/s}$), training an MNIST MLP to **96.88% test accuracy** in **34.0 s** ($2.27\times$ faster than reference) and a 30k-parameter character-level Transformer on TinyShakespeare with loss decreasing from **4.33 to 3.15** in **0.78 s**; validated across 18 C++ unit tests, 11 lit golden IR tests, and float64 central finite-difference gradient checks.

---

## Course-Concept Justification Table

| Bullet Phrase | Coursework Applied | Empirical Reference |
|---|---|---|
| SSA Graph IR & Loop IR | CS6843 Program Analysis | Architecture spec (`docs/architecture.md`) |
| Reverse-mode autodiff with analytic VJPs | Mathematical foundations & ID5130 | Op VJP checks (`tests/python/test_gradcheck.py`) |
| Liveness-based memory planning (26.2% reduction) | CS6843 Program Analysis (Interval coloring) | `docs/perf-report.md` §4 ($1.64\text{ MB} \to 1.21\text{ MB}$) |
| Epilogue fusion (1.39x DRAM traffic reduction) | CS6843 PAR & CS6600 Caches | `docs/perf-report.md` §3 ($1860\text{ KB} \to 1336\text{ KB}$) |
| Static range-analysis bounds-check elimination | CS6843 Program Analysis (A2/A4) | Lit tests (`bounds_check_elim.dlc`, `loop_versioning.dlc`) |
| 16x16 shared-memory tiled CUDA GEMM | CS6023 GPU Programming (L3 Memory) | CUDA backend (`lib/backends/cuda/CUDACodeGen.cpp`) |
| 5-step warp-shuffle reduction trees | CS6023 GPU Programming (L8 Warps) | `__shfl_down_sync` in reductions (`cuda_reductions.dlc`) |
| AVX2 peak (112.5 GFLOP/s) & STREAM (14.09 GB/s) | CS6400 ACA / ID5130 Roofline | FMA peak microbenchmark (`benchmarks/fma_peak.cpp`) |
| MNIST MLP: 96.88% accuracy in 34.0 s (2.27x) | LLVM JIT & AVX2 Code Generation | MNIST benchmark (`examples/mnist_mlp.py`) |
| 30k-parameter Char-GPT: loss 4.33 -> 3.15 in 0.78 s | Sequence modeling & Adam optimization | Char-GPT benchmark (`examples/char_gpt.py`) |
| 18 C++ unit tests & 11 lit golden IR tests | LLVM Testing Discipline | Automated test suites (`./build/dlc_unit_tests`) |
