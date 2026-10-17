# Range Analysis, Bounds-Check Elimination and Loop Peeling

## Purpose
Prove with static interval analysis that every memory access in a loop nest stays in bounds, so the accesses need no
runtime checks; where a vectorized loop's trip count is not a multiple of the vector width, split it so that both
parts provably stay in bounds.

## Interval range analysis (CS6843 A2)

### 1. Induction-variable environment
For each loop in the nest:
$$\text{ForLoop}(v, \text{start}, \text{end}, \text{step}) \implies \text{Env}[v] = [\text{start}, \text{end} - 1]$$

### 2. Interval arithmetic
- **Constant**: $\text{Interval}(c) = [c, c]$
- **Variable**: $\text{Interval}(v) = \text{Env}[v]$
- **Addition**: $[l_1, u_1] + [l_2, u_2] = [l_1 + l_2, u_1 + u_2]$
- **Subtraction**: $[l_1, u_1] - [l_2, u_2] = [l_1 - u_2, u_1 - l_2]$
- **Multiplication**: $[\min(p_i), \max(p_i)]$ for $p_i \in \{l_1 l_2, l_1 u_2, u_1 l_2, u_1 u_2\}$

### 3. Safety check
For every load or store into buffer $B$ with $N = \prod \text{shape}$ elements, the index interval must satisfy
$idx_{\min} \ge 0$ and $idx_{\max} < N$.

## Loop peeling (CS6843 A4)
When a loop is vectorized with width $W > 1$ and its trip count $R$ is not divisible by $W$:
1. **Vector main loop:** $[\text{start}, \text{split})$ with $\text{split} = \text{start} + \lfloor R / W \rfloor \cdot W$.
2. **Scalar tail loop:** $[\text{split}, \text{end})$.

If every access is proven in bounds, the kernel is stamped `attributes {bounds_check_eliminated = true}`. With static
shapes no runtime test is needed to choose between the two parts, so this is peeling rather than versioning.

## Status (v1)
- The analysis and the peeling work and are tested (`tests/lit/bounds_check_elim.dlc`, `tests/lit/loop_versioning.dlc`,
  `tests/unit/test_runtime.cpp`).
- v1 never emits bounds checks, so there is nothing to eliminate: the attribute is printed but no code generator reads
  it, and the LLVM template ignores the main/tail split.
- The access check ignores the vector width, relying on peeling to make trip counts divisible; the division rule is
  unsound for negative intervals.

## v2 (P4–P5)
The loop IR gets explicit `if` guards, so the pass deletes the guards it proves redundant. On the GPU, partial tiles are
split into unguarded interior blocks and guarded boundary blocks, and the cost of the guards is measured.
