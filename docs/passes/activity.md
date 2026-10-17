# Activity Analysis

## Purpose
Decide which values need an adjoint before autodiff builds anything, so the backward graph contains only useful work.
In the MNIST MLP, v1 built the gradient of the loss with respect to the input images — a 128×256×784 GEMM per step,
about half of the backward FLOPs — and threw it away.

## Algorithm (`lib/passes/Activity.cpp`)
Two dataflow problems over the SSA block, both along **differentiable edges** only (an `f32` operand of an op whose
derivative with respect to it can be nonzero: not integer operands, not the operands of `cmp_gt`, `onehot` or `cast`,
not the condition of `select`, not the indices of `embedding`, `scatter_add` or `cross_entropy`):

1. **Varied** (forward): the `param` arguments are varied; an op's result is varied if a differentiable operand is.
2. **Useful** (backward): the loss is useful; the differentiable operands of an op with a useful result are useful.
3. **Active** = varied ∩ useful. Autodiff applies VJP rules only to ops with an active result, and only towards active
   operands.

Both are single passes: the block is in SSA form and topologically ordered, so program order (or its reverse) visits
every definition before its uses (or every use before its definition).

`dlc-opt --passes=activity` marks the active ops with an `ad.active = 1` annotation.

## Status (P2)
Tested in `Activity.ActiveValuesDependOnAParameterAndReachTheLoss` (an input-only path, a comparison, an embedding
that never reaches the loss and a parameter-only dead path are all inactive) and through autodiff:
`Autodiff.BuildsGradientsOnlyForParameters` checks that no value of the input's shape is built.

## Course link
The varied/useful formulation is the standard one in AD tools (Hascoët & Pascual, Tapenade); each half is a dataflow
analysis — forward "may depend on", backward "may reach" (CS6843 DFA).
