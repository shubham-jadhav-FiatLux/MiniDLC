# Reverse-Mode Automatic Differentiation Pass

## Purpose
Transforms a forward computation graph targeting a scalar loss into an augmented training graph that evaluates gradients with respect to parameters and inputs via Vector-Jacobian Products (VJPs).

## Algorithm
1. Traverse forward operations in reverse topological order starting from the loss node.
2. Initialize the loss adjoint \( dL = 1.0 \).
3. For each op, look up its registered VJP derivation rule (§1.5 of specification).
4. For nodes with multiple fan-out uses, sum incoming adjoint contributions.
5. Append optimizer update operations (SGD or Adam) consuming gradients and updating persistent parameter buffers.
