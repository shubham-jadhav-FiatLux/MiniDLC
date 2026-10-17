# Reverse-Mode Automatic Differentiation

## Purpose
Turn a forward graph with a scalar loss into a graph that also computes the gradient of the loss with respect to every
parameter, using vector-Jacobian products (VJPs).

## Algorithm (`lib/passes/Autodiff.cpp`)
1. The loss is the function's first result and must be `f32[]`; the parameters are the arguments with the `param` role.
2. [Activity analysis](activity.md) finds the active values: those that depend on a parameter and reach the loss.
   Nothing is built for the rest — for the MNIST MLP, no gradient for the input images (a 128×256×784 GEMM per step
   in v1).
3. Seed the loss adjoint with 1.0 and visit the forward ops in reverse program order (a reverse topological order of
   the SSA block). For each op whose result has an adjoint, its VJP rule adds contributions to the adjoints of its
   active operands; contributions to a value with several consumers are summed.
4. Append each parameter's gradient to the results, annotated `{grad = %param}`; a parameter the loss doesn't depend on
   gets a zero constant. [Optimizer lowering](optimizer.md) turns these into parameter updates.

The rules (with the backward primitives of the [IR spec](../ir-spec.md#3-operations)):

| Op | VJP |
|---|---|
| `add`, `sub`, `mul`, `div` | product rule; `unbroadcast` = `sum` over broadcast axes + `reshape` |
| `relu` | `select(cmp_gt(x, 0), dy, 0)` |
| `gelu` | tanh form: `dy · [½(1+t) + ½x(1−t²)·c(1+3k x²)]` |
| `exp`, `log`, `tanh`, `sqrt`, `rsqrt` | from the result where possible (`dy·y`, `dy·(1−y²)`, …) |
| `matmul` | `dA = dY B̂ᵀ`, `dB = Âᵀ dY` as matmuls with transpose flags; 3-D × 2-D flattens the batch for `dB` |
| `softmax`, `log_softmax` | `y ⊙ (dy − Σ dy⊙y)`, `dy − exp(y) Σ dy` along `axis` |
| `cross_entropy` | `(softmax(z) − onehot(y)) · dL / N`, N = rows (B·T for 3-D logits) |
| `layernorm` | recomputes μ, rstd, x̂; `dβ = Σ dy`, `dγ = Σ dy⊙x̂`, `dx = (rstd/D)(D g − Σ g − x̂ Σ g⊙x̂)`, `g = dy⊙γ` |
| `sum`, `mean`, `reduce_max` | `broadcast_to` back over the reduced axes (÷n for mean; ties share for max) |
| `transpose`, `reshape`, `broadcast_to` | inverse permutation, reshape back, unbroadcast |
| `embedding` / `scatter_add` | `scatter_add(idx, dy)` / `embedding(dy, idx)` |
| `select` | `select(c, dy, 0)` and `select(c, 0, dy)`, unbroadcast |

Result types are inferred as the backward graph is built, so a rule that builds an ill-typed op fails loudly.

## Status (P2)
- Every rule is checked against float64 central finite differences (`tests/python/test_training_graph.py`, 37
  graphs, including every combination of matmul transpose flags), and the gradients of whole models — softmax
  regression, an MLP, Char-GPT, a graph of the remaining ops and ten random DAGs — against the reference interpreter's
  Python VJPs (`tests/python/test_translation_validation.py`).
- With [optimizer lowering](optimizer.md), the training step built in C++ follows eager training step by step for an
  MLP and Char-GPT under Adam and SGD with momentum.
- The **Python reference interpreter** (`python/minidlc/reference.py`) keeps its own eager VJPs as the oracle.

## Course link
Chain rule and the adjoint method; activity analysis is a pair of dataflow problems (CS6843).
