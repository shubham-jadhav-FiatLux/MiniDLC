# Range Analysis, Guards and Bounds-Check Elimination

## Purpose
Prove that every memory access of a loop kernel stays inside its buffer, so the generated code needs no runtime check.
Where an index comes from data and cannot be proven, the generated code checks it, so that it is memory-safe for any
input.

## Algorithm (`lib/loopir/Analysis.cpp`)
1. **Range analysis:** interval arithmetic over the loop variables (`[0, extent − 1]`), integer locals, `+ − ×`,
   division and remainder by constants, `min`/`max`. Branch conditions refine ranges: inside `if (x < c)`, x ≤ c − 1;
   in the else branch, x ≥ c; a branch whose range becomes empty is infeasible and skipped.
2. **Bounds checking:** every load and store index must lie in `[0, elements − 1]`.
   - An index computed from data (loaded from memory) gets a **guard**: a load becomes
     `select(in bounds, load(clamped index), 0)`, a store runs only when in bounds.
   - The schedules guard their own data indices explicitly, saying what an invalid index means: a zero embedding row,
     a skipped scatter row, a NaN loss. Inside those guards, refinement proves the accesses.
   - Any other unprovable access is a lowering bug, reported as an error.
3. **Bounds-check elimination:** an `if` whose condition range analysis decides is replaced by the branch it takes.
4. **Tiling without peeling:** where a tile doesn't divide the iteration space, the schedule emits the full tile and
   the partial tile as two branches (`tests/lit/loop_versioning.dlc`). This is the "unguarded interior, separate
   boundary" form; on the GPU (P5) partial tiles are guarded and BCE removes the guards that tile-divisible shapes make
   redundant.

## Status (P4)
- Unit tests `RangeAnalysis.*` and `Bounds.*`: interval arithmetic, refinement, guards on indices read from memory
  (an out-of-range id reads 0), elimination of decided guards, and an out-of-bounds schedule rejected.
- Lit tests `bounds_check_elim.dlc` (guarded embedding and labels) and `loop_versioning.dlc` (partial tiles).
- `dlc-opt --print-loop-report` shows, per kernel, the accesses proven, guarded and eliminated. Every access of the
  MNIST and Char-GPT training steps is proven or guarded.

## Course link
Range analysis and bounds-check elimination: CS6843 assignments A2 and A4, with branch refinement — the part of A4
that was hardest to get right.
