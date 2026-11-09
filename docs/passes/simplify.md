# Loop Simplification

## Purpose
Clean up the loop IR after the schedules and the bounds checks: constants the schedules bind to names, index
arithmetic that the ranges of loops and branches decide, and computations repeated in every iteration although they
do not change. It runs on both targets after the bounds and dependence checks (`simplify` in
`lib/loopir/Transforms.cpp`), so its rewrites never hide anything from them.

## Algorithm
1. **Constant propagation.** `let x = 7` (an integer constant, bound once in the kernel and never assigned) is
   substituted into its uses and removed; the expression constructors fold what becomes constant (`(o * 100 + l) + q`
   with `q = 0` is `o * 100 + l`).
2. **Division and remainder folding with range facts.** Walking the kernel with the same range environment as
   bounds-check elimination (loop variables, integer locals, branch conditions), `x / c` and `x % c` (c > 0, x ≥ 0)
   become `q` and `x − q·c` when the quotient `q` is the same over x's whole range — e.g. `j / 8 = 0` and `j % 8 = j`
   inside `for j in 0..8`.
3. **Loop-invariant code motion.** A `let` whose value is pure (no loads, no shuffles, no integer division except by
   a nonzero constant — a hoisted value runs even if the loop runs zero times, as with zero-size dimensions) and uses
   nothing the loop defines or assigns moves out of a serial, vectorized or unrolled loop that runs at least once,
   innermost first (so it can keep moving out).
   It must be bound once in the kernel and never assigned — an accumulator re-initialized each iteration
   (`let acc = 0` followed by `acc = acc + ...`) stays. Nothing moves out of a parallel loop (the runtime splits its
   range) or a bound loop (the GPU launch). In the GPU matmul, the row index `by*64 + ty + 16i` moves out of the
   column loop of the epilogue.

**Strength reduction** of induction variables (`i * c` → an accumulator advanced by `c`) is left to LLVM's loop
strength reduction: in the loop IR it would replace the affine indices that range and dependence analysis read with
loop-carried variables they cannot bound.

## Status
`--print-loop-report` prints what the pass changed per kernel. On the CPU it changes nothing measurable — LLVM's O3
does the same work: the MNIST step's fastest time is 0.82 ms with it and 0.83 ms without (`benchmarks/ablations.py`,
the `simplify: False` ablation; perf report, end of Section 8). It matters for the loop IR itself (what the reports,
the interpreters and the GPU analyses read). `tests/unit/test_loopir.cpp` checks each rewrite and
that the interpreter computes the same results before and after; every differential and training test runs the
simplified kernels.

## Course link
CS6843 Program Analysis: constant propagation, range analysis applied to expression folding, and loop-invariant code
motion (dataflow over the loop nest: a definition is invariant if none of its operands is defined in the loop).
