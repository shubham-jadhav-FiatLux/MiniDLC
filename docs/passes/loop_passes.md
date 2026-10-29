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
Range analysis, bounds checking with guard insertion, bounds-check elimination and schedule verification; see
[bounds_check_elim.md](bounds_check_elim.md) and the [IR spec](../ir-spec.md#5-loop-ir-v2).

## What LLVM does
Index simplification as expressions are built is ours. Loop-invariant code motion and strength reduction are left to
LLVM's O3 pipeline, as is the mechanical widening of `vectorize(8)` loops (loop vectorizer) and unrolled register tiles
(SLP vectorizer) into `<8 x float>` instructions; `tests/lit/loopir_lowering.dlc` checks that the matmul tile becomes
`llvm.fmuladd.v8f32`.

## Status (P4)
Differential tests (`tests/python/test_cpu_backend.py`) run every op through the schedules on shapes that are not
multiples of the tile or the vector width, with broadcasting, zero-size dimensions and NaN/Inf, against the reference
interpreter, both compiled and in the loop IR interpreter. MNIST and Char-GPT train through the generated code
(`tests/python/test_cpu_training.py`, the examples).

## Course link
Cache blocking, register tiling, SIMD and ILP (CS6600); work sharing across threads and Amdahl's law (ID5130).
