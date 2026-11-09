# CPU Schedules and Loop Passes

## Purpose
Turn each fused kernel into a loop nest that uses the CPU: cache-friendly tiles, SIMD, instruction-level parallelism
and every core. A schedule is a rule per kernel class, written in `lib/loopir/Lowering.cpp`; the emitter only
translates it.

## CPU schedules

| Kernel class | Schedule |
|---|---|
| Matmul (+ epilogue) | Parallel loop over 4-row blocks (× batch). In each block, 16-column tiles of a 4×16 register tile accumulated over k, with A broadcast and B streamed by rows. The epilogue runs on the tile in registers. With a transposed B, a 4×4 tile with the k loop vectorized (dot products). Partial tiles are separate branches (`if (p < full)`), not per-element guards. |
| Elementwise (multi-output) | Dimensions collapsed where every operand is contiguous or broadcast alike; the outer loop parallel (chunks of 8192 when flat), the inner one `vectorize(8)`. Kernels with under 2¹⁵ elements × ops stay on one thread. |
| Row (softmax, log_softmax, LayerNorm, cross-entropy) | Parallel over rows, a vectorized pass per reduction over each row (max, sum of exponentials; mean, variance). |
| Reduction | Reduced axes contiguous at the end: a register accumulator per output, vectorized. Leading or middle: each output row accumulated in place, streaming contiguous rows. Anything else: initialize, accumulate, finalize. |
| Gather / scatter | Per index: a guarded, vectorized row copy (embedding, zero row for an invalid id) or row accumulation (scatter, serial over indices). |
| Transpose, fill | Parallel outer loop, vectorized inner loop. |

## Loop passes
Per kernel, in order (`lowerFunction`): range analysis, bounds checking with guard insertion and bounds-check
elimination ([bounds_check_elim.md](bounds_check_elim.md)); the dependence check of every parallel and vectorized loop;
simplification — constant propagation, division folding with range facts, loop-invariant code motion
([simplify.md](simplify.md)). GPU kernels add bank-conflict padding before and a resource check after
([gpu_passes.md](gpu_passes.md)). See also the [IR spec](../ir-spec.md#5-loop-ir-v2).

## What LLVM does
Constant folding of index arithmetic as expressions are built, constant propagation, range-based division folding and
LICM of `let`s are ours. Strength reduction of induction variables is left to LLVM's O3 pipeline (it would hide the
affine indices our analyses read), as is the mechanical widening of `vectorize(8)` loops (loop vectorizer) and
unrolled register tiles (SLP vectorizer) into `<8 x float>` instructions; `tests/lit/loopir_lowering.dlc` checks that
the matmul tile becomes `llvm.fmuladd.v8f32`.

## Compiled code and the kernel cache
The emitter's module goes through O3 and code generation to an object file, which ORC LLJIT links into the process.
The object file is cached on disk, keyed by a SHA-256 of the kernels' loop IR, the host CPU (triple, name, features),
the LLVM version and the build of the code generator (`lib/backends/llvm/KernelCache.cpp`): compiling the Char-GPT
training step takes 4.4 s the first time and 0.09 s from the cache. `MINIDLC_CACHE_DIR` moves it, `MINIDLC_CACHE=off`
disables it, `minidlc.cache_info()` and `minidlc.clear_cache()` inspect and empty it. The same object file, saved with
the plan by `dl.compile_cpu`, is a CPU artifact: `minidlc.cpu.load` links it later, or on another machine with the
same triple and CPU features, without compiling.

## Status (P4)
Differential tests (`tests/python/test_cpu_backend.py`) run every op through the schedules on shapes that are not
multiples of the tile or the vector width, with broadcasting, zero-size dimensions and NaN/Inf, against the reference
interpreter, both compiled and in the loop IR interpreter. MNIST and Char-GPT train through the generated code
(`tests/python/test_cpu_training.py`, the examples).

## Course link
Cache blocking, register tiling, SIMD and ILP (CS6600); work sharing across threads and Amdahl's law (ID5130).
