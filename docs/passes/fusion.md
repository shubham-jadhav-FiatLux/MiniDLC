# Operator Fusion

## Purpose
Group the ops of a function into kernels, so values computed and consumed inside a kernel never make a round trip to
memory. The kernels are what gets lowered and scheduled (P4/P5).

## Kernel classes (`lib/passes/Fusion.cpp`)

| Class | Members | Lowered as |
|---|---|---|
| `matmul` | one matmul plus an elementwise **epilogue** of its output shape | an accumulation, then the epilogue per output element, in registers |
| `elementwise` | elementwise ops (and `broadcast_to`) with one output shape; any number of outputs | one loop over the output shape |
| `reduction` | `sum`, `mean`, `reduce_max` | a reduction over the axes |
| `row` | `softmax`, `log_softmax`, `layernorm`, `cross_entropy` | one or two passes over each row |
| `gather` / `scatter` | `embedding`, `onehot` / `scatter_add` | indexed copies / atomic adds |
| `transpose` | `transpose` | a permuted copy |
| `fill` | a `constant` that a kernel reads from memory | a fill |

Splat constants read by elementwise code are inlined as immediates; `reshape` needs no kernel (it aliases its operand,
see [buffer assignment](mem_plan.md)).

## Algorithm
1. Seed one group per op that needs a kernel.
2. Visit the elementwise ops in program order. Each tries to join the group of every operand's producer (matmul groups
   first, so it joins an epilogue) when the operand has the op's output shape.
3. A merge is **legal** only if:
   - **iteration spaces match:** both groups have the op's output shape;
   - **epilogues never feed the matmul:** no elementwise member may be an ancestor of the matmul, since the epilogue
     runs after the accumulation;
   - **the kernel graph stays acyclic:** there is no path from one group to the other through a third group. This
     is a reachability check on the quotient graph; checking paths between single ops is not enough once groups hold
     several ops.
4. Order the kernels topologically (Kahn's algorithm, earliest member first), so every input of a kernel is written by
   an earlier kernel or is an argument.
5. A member's result is a kernel **output** if anything outside the kernel reads it (another kernel, a reshape or the
   return); the other results stay in registers.

`dlc-opt --passes=fusion` annotates each member with `fusion.group` and `fusion.kernel`; `--print-cost` prints the
kernels with the [cost model](cost_model.md).

## Status (P3)
- Unit tests: `Fusion.*` in `tests/unit/test_passes.cpp` — the MNIST layer as one `matmul_add_relu` kernel,
  multi-output elementwise kernels, a merge rejected because it would make the kernel graph cyclic, an elementwise op
  that feeds a matmul kept out of its epilogue, constants inlined or filled, and the topological order. Golden test:
  `tests/lit/fusion.dlc`.
- Translation validation runs the pass on 16 exported graphs.
- On the MNIST training step it forms 15 kernels: each Adam update lands in the epilogue of the matmul that computes
  its gradient, and fusion cuts the compulsory traffic from 31.4 MB to 7.6 MB (`docs/perf-report.md` §4).
- Profitability is not modelled: every legal merge is taken. On a CPU, that is never worse than not fusing. A GPU
  register-pressure limit on epilogue size can come with P5 if needed.

## Course link
Dependence analysis and the legality of reordering (CS6843 PAR): a fusion is legal when it creates no cycle in the
dependence graph, the same condition as for loop fusion.
