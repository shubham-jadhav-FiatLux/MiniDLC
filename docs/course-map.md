# Course-Concept Mapping

This document tracks how specific compiler and systems concepts from academic coursework are implemented within MiniDLC:

| Concept | Location in MiniDLC | Course Reference |
|---|---|---|
| SSA, IR design, verifier, textual IR | `include/dlc/ir/`, `lib/ir/` | CS6843 Program Analysis (LLVM IR foundations) |
| Dataflow analysis: available expressions, liveness | `include/dlc/passes/CSE.h`, `DCE.h`, `MemoryPlanning.h` | CS6843 Program Analysis (DFA frameworks) |
| Abstract interpretation | `include/dlc/passes/ShapeInference.h` | CS6843 Program Analysis (Abstract Interpretation) |
| Range analysis & bounds-check elimination | `include/dlc/loopir/LoopPasses.h` | CS6843 Program Analysis (Range Analysis & Instrumentation) |
| Dependence analysis, legality of reordering | `include/dlc/passes/Fusion.h` | CS6843 Program Analysis (Loop Parallelism & Dependences) |
| lit / FileCheck testing discipline | `tests/lit/` | CS6843 Program Analysis (LLVM testing workflow) |
| Caches, tiling, SIMD vectorization, ILP | `include/dlc/loopir/`, `lib/backends/llvm/` | CS6600 Computer Architecture |
| Roofline model, arithmetic intensity | `benchmarks/roofline.py`, `docs/perf-report.md` | CS6400 Advanced Computer Architecture, ID5130 |
| Thread pool, parallel loop scheduling | `include/dlc/runtime/ThreadPool.h` | ID5130 Parallel Scientific Computing |
| CUDA memory hierarchy, shared memory tiling | `lib/backends/cuda/` | CS6023 GPU Programming (Memory hierarchy) |
| Warp shuffles, parallel reduction trees | `lib/backends/cuda/` | CS6023 GPU Programming (Warp primitives) |
| Numerical stability (log-sum-exp, fp32 accumulation) | `lib/ir/Ops.cpp`, `python/minidlc/reference.py` | ID5130 Parallel Scientific Computing |
| Reverse-mode automatic differentiation | `include/dlc/passes/Autodiff.h` | Mathematical foundations (Vector-Jacobian Products) |
