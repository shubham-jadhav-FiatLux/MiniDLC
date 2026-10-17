# Canonicalize and Constant Folding Pass

## Purpose
Simplifies algebraic expressions, eliminates identity operations, and evaluates constant subgraphs at compile-time to reduce instruction count and normalize the intermediate representation before dataflow optimization.

## Implemented Simplifications

1. **Additive Identity Elimination**:
   - `x + 0` $\to$ `x`
   - `0 + x` $\to$ `x`
   - `x - 0` $\to$ `x`

2. **Multiplicative Identity & Annihilator**:
   - `x * 1` $\to$ `x`
   - `1 * x` $\to$ `x`
   - `x * 0` $\to$ `0`
   - `x / 1` $\to$ `x`

3. **Constant Folding**:
   - Statically evaluates operations whose operands are all compile-time constants (e.g. constant folding of scalar weights or hyperparameter scalars).

4. **Commutative Normalization**:
   - Standardizes argument ordering for commutative operations (`add`, `mul`) by value ID to maximize Common Subexpression Elimination (CSE) hit rates.

## Algorithm & Convergence
- The pass performs iterative pattern-matching rewrites in topological order.
- Fixpoint iteration: rewrites continue until no further rules apply, ensuring maximal simplification.

## Academic Link
- **Course Reference**: Compilers / Program Analysis algebraic normalization.
