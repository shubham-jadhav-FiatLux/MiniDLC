# Changelog

Notable changes to MiniDLC. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project uses [Semantic Versioning](https://semver.org/). The development log (`docs/devlog.md`) has the details of
each phase.

## [Unreleased]

### Added
- **v2 compiler, phases P1–P8** (`docs/architecture.md`):
  - P1: an op registry, a round-tripping parser and a verifier for the graph IR, and IR export from Python checked by
    translation validation.
  - P2: the training graph in C++: backward primitives, activity analysis, reverse-mode autodiff, and the Adam/SGD
    update lowered to ops.
  - P3: fusion with legality checks, a cost model, and buffer assignment with a checker.
  - P4: the CPU backend: loop IR v2 with schedules, range-analysis bounds-check elimination, a dependence check, an
    LLVM emitter with ORC LLJIT, a runtime with a thread pool, and Python bindings.
  - P5: the GPU backend: GPU schedules, a lockstep GPU interpreter (`target="gpu-sim"`), NVPTX emission to PTX,
    artifacts and a CUDA driver-API runtime.
  - P6: multi-head attention, a cross-check against PyTorch, and held-out evaluation on every target.
  - P7: baselines, ablations, and the performance report.
  - P8: race detection in the GPU interpreter; GPU loop passes (bank-conflict padding, coalescing statistics, a
    resource check, partial-tile versioning, block-per-row reductions); loop simplification; an on-disk kernel cache;
    CPU artifacts (`dl.compile_cpu`); checkpoints that move between targets; CUDA-event timing and PyTorch/cuBLAS
    baselines on the GPU; a whole-pipeline fuzzer; nightly end-to-end tests.
- Contributor guide, code of conduct, security policy, and issue and pull-request templates.

### Removed
- The v1 template backends and the other v1 sources that v2 no longer built: `lib/backends/cuda/`,
  `LLVMCodeGen.cpp`, `LLVMBackend.cpp`, `LoopPasses.cpp`, `ExecutionPlan.cpp`, `MemoryPlanning.cpp`, their headers,
  `tests/unit/test_cuda.cpp` and `python/minidlc/cuda_engine.py`.

### Fixed
- The thread pool waits for every chunk before rethrowing an exception.
- MNIST accuracy is measured on all 10,000 test images.

## [0.1.0] — v1

The first version: a tracing frontend, a graph IR with passes, a loop IR with range analysis, template-based LLVM
and CUDA backends, and the MNIST and Char-GPT examples. Superseded by v2; the audit that led to the rewrite is in
`docs/devlog.md`.
