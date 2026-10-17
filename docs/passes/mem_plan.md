# Buffer Assignment (memory planning)

## Purpose
Give every value that lives in memory a buffer: parameters and optimizer state in a **persistent region** allocated once,
everything else in one **transient arena** whose space is reused by values whose lifetimes don't overlap.

## Algorithm (`lib/passes/BufferAssignment.cpp`)
1. Run [fusion](fusion.md) for the kernel schedule. Only kernel outputs, arguments and returned values need memory;
   values inside a kernel stay in registers.
2. **Persistent region:** `param`, `state` and `const` arguments, back to back, 256-byte aligned.
3. **Aliases:** a `reshape` result shares its operand's buffer. A donated result (`{donates = %p}`) is written **in
   place**, into %p's buffer, if the kernel that computes it reads %p only at the element it writes (an elementwise
   kernel over %p's shape, or a matmul epilogue) and every other reader of %p runs before it. Otherwise it gets its
   own buffer and is copied over %p after the last kernel.
4. **Live ranges** over the kernel schedule: from the producing kernel (−1 for arguments) to the last kernel reading
   the value or anything aliasing it (K, past the end, for returned values).
5. **Transient arena:** largest buffer first, each at the lowest 256-byte-aligned offset that no buffer live at the
   same time occupies (interval colouring, as in linear-scan register allocation; Poletto & Sarkar 1999).
6. **Checker** (`BufferAssignment::verify`): no two buffers live at the same time overlap, every offset is aligned, and
   every in-place update is safe. It runs on every assignment.

`dlc-opt --passes=buffer-assign` annotates each kernel output's op with `mem.region`, `mem.offset` and `mem.size`;
`--print-mem-report` prints the regions, every buffer's offset and live range, and the reuse.

## Status (P3)
- The two v1 defects are fixed and tested: a fused kernel's output never overlaps its inputs
  (`BufferAssignment.FusedKernelOutputDoesNotAliasItsInputs`), and parameters live in the persistent region
  (`BufferAssignment.ParametersLiveInThePersistentRegion`).
- `BufferAssignment.*` also covers live-range reuse, in-place updates of a training step, the copy fallback and
  reshape aliases. The mutation fuzzer checks the assignment of every mutant that verifies.
- Measured on the training step (`benchmarks/memory_plan.py`, `docs/perf-report.md` §4): every Adam update is in place,
  and the Char-GPT arena is 63% smaller than without reuse.
- In-place reuse of a dying input by an elementwise output (beyond donated updates) is not done; the interval check
  treats a kernel's inputs and outputs as live together.

## Course link
Liveness (CS6843 DFA) and interval colouring — the register-allocation analogy.
