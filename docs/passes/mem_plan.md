# Memory Planning

## Purpose
Assign every tensor an offset in one contiguous arena, reusing memory between tensors whose lifetimes don't overlap.

## Algorithm
1. Take the execution schedule (the program order of the ops).
2. Compute a live interval [definition, last use] for every value; values returned by the function stay live to the end.
3. Sort values by size, largest first.
4. For each value, place it at the lowest 64-byte-aligned offset that doesn't overlap any already placed value whose
   interval intersects its own (interval colouring; compare linear-scan register allocation, Poletto & Sarkar 1999).

## Status (v1)
- The algorithm is correct for an op-by-op schedule (`tests/unit/test_passes.cpp` checks that values live at the same
  time never share bytes).
- It runs on the unfused schedule, but a fused kernel executes several ops at once, so the kernel's output can be placed
  inside its own inputs (`KnownBugs.DISABLED_FusedKernelOutputDoesNotAliasItsInputs`).
- Function arguments, including parameters, are treated like transient values, so later intermediates reuse their memory
  (`KnownBugs.DISABLED_ParameterBuffersAreNeverReused`).
- No executed path uses the plan, and no reproducible saving is reported (`docs/perf-report.md` §4).

## v2 (P3)
Buffer assignment after fusion with per-kernel liveness, a persistent region for parameters and optimizer state, in-place
reuse for elementwise kernels, and 256-byte alignment for coalesced GPU access; the executed arena comes from this plan.
