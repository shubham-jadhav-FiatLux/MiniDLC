# Reverse-Mode Automatic Differentiation

## Purpose
Turn a forward graph with a scalar loss into a graph that also computes gradients, using vector-Jacobian products
(VJPs).

## Algorithm
1. Traverse the forward ops in reverse program order (a reverse topological order of the SSA block), starting from the
   loss.
2. Seed the loss adjoint with 1.0.
3. For each op whose result has an adjoint, apply its VJP rule to produce adjoints for its operands.
4. Sum the contributions when a value has several consumers (fan-out).
5. Return the loss together with the gradients.

## Status (v1)
- **Python reference interpreter** (`python/minidlc/reference.py`): reverse-mode autodiff over the traced graph,
  evaluated eagerly in NumPy, for all 18 ops. Every op is checked against float64 central finite differences
  (`tests/python/test_gradcheck.py`). This is the autodiff used for training today.
- **C++ graph pass** (`lib/passes/Autodiff.cpp`): builds a backward graph, but several VJPs are wrong. ReLU and GELU
  multiply the upstream gradient by the input; `sum` and `mean` don't broadcast the gradient back (and `mean` misses
  1/n); LayerNorm skips the normalization Jacobian; `softmax` and `log_softmax` ignore `axis`; 3-D cross-entropy divides
  by the wrong N; `embedding` and 3-D × 2-D matmul get no gradient. It differentiates with respect to every argument,
  ignores its parameter list, and appends no optimizer update.

## v2 (P2)
Backward primitives (`cmp_gt`, `select`, `reduce_sum(axes)`, `broadcast_to`, `scatter_add`, …) make every VJP
expressible. Activity analysis restricts the backward graph to values that depend on a parameter and reach the loss.
Optimizer lowering appends the Adam/SGD update with donated parameter buffers. The C++ gradients are checked by
evaluating the backward graph in the reference interpreter against finite differences.

## Course link
Chain rule and the adjoint method; activity analysis is a pair of dataflow problems (CS6843).
