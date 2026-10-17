# Optimizer Lowering

## Purpose
Append the parameter update to a differentiated function, so that **one graph is one training step**: the step reads
the batch, the parameters and the optimizer state, and returns the loss together with the new parameters and state.

## Algorithm (`lib/passes/Optimizer.cpp`)
The pass consumes every `{grad = %p}` result that [autodiff](autodiff.md) produced:

- **SGD** (`sgd{lr=,momentum=}`): `θ ← θ − lr·g`; with momentum, a state argument `%p.velocity` (init 0) and
  `v ← μv + g`, `θ ← θ − lr·v`.
- **Adam** (`adam{lr=,beta1=,beta2=,eps=}`): state arguments `%p.m`, `%p.v` (init 0) per parameter and the scalars
  `%adam.beta1_pow`, `%adam.beta2_pow` (init 1) for bias correction:
  `m ← β1 m + (1−β1) g`, `v ← β2 v + (1−β2) g²`, `m̂ = m/(1−β1ᵗ)`, `v̂ = v/(1−β2ᵗ)`, `θ ← θ − lr·m̂/(√v̂ + ε)`.

Every new parameter and state value is returned as `{donates = %arg}`: it takes over the argument's buffer, so the
runtime updates parameters in place instead of copying them. The non-gradient results (the loss, and any other forward
outputs) stay first. Hyperparameters are baked into the graph as constants.

```
dlc-opt --passes="autodiff,adam{lr=0.001},canonicalize,cse,dce" model.dlc
```

## Status (P2)
`tests/lit/optimizer.dlc` checks the generated signature for Adam and SGD with momentum and the error messages;
`tests/python/test_training_graph.py` runs the step function of an MLP and a Char-GPT for several steps in the
reference interpreter, feeding the donated results back, and matches eager training with `optim.Adam` / `optim.SGD`.

## Course link
Graph construction; in-place updates through donation are the input-output aliasing of XLA and the buffer donation of
JAX, and they are what [buffer assignment](mem_plan.md) relies on to keep parameters in a persistent region.
