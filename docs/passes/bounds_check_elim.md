# Static Range-Analysis Bounds-Check Elimination & Loop Versioning Pass

## Purpose
Eliminates redundant runtime bounds checks and guard branches inside loop nests using formal static interval range analysis. When static dimensions are not exact multiples of vector widths, loop versioning / peeling is applied to produce a fast vectorized path and a scalar cleanup path.

## Interval Range Analysis (`CS6843 A2`)

### 1. Induction Variable Environment
For each loop in the nest:
$$\text{ForLoop}(v, \text{start}, \text{end}, \text{step}) \implies \text{Env}[v] = [\text{start}, \text{end} - 1]$$

### 2. Expression Interval Arithmetic
- **Constant**: $\text{Interval}(c) = [c, c]$
- **Variable**: $\text{Interval}(v) = \text{Env}[v]$
- **Addition**: $[l_1, u_1] + [l_2, u_2] = [l_1 + l_2, u_1 + u_2]$
- **Subtraction**: $[l_1, u_1] - [l_2, u_2] = [l_1 - u_2, u_1 - l_2]$
- **Multiplication**: $[l_1, u_1] \times [l_2, u_2] = [\min(p_1, p_2, p_3, p_4), \max(p_1, p_2, p_3, p_4)]$ where $p_i \in \{l_1 l_2, l_1 u_2, u_1 l_2, u_1 u_2\}$

### 3. Safety Verification
For every memory access (e.g. `LoadExpr` or `StoreStmt`) referencing buffer $B$ with capacity $N = \prod \text{shape}$:
$$\text{Evaluate } \text{idx} \in [idx_{\min}, idx_{\max}]$$
If $idx_{\min} \ge 0$ and $idx_{\max} < N$, the access is statically proven to never exceed buffer bounds.

## Loop Versioning / Peeling (`CS6843 A4`)

When a loop is vectorized with width $W > 1$ and the range $R = \text{end} - \text{start}$ is not divisible by $W$ ($R \pmod W \ne 0$):
1. **Vectorized Main Loop**: Spans $[0, \text{split})$ where $\text{split} = \text{start} + \lfloor R / W \rfloor \cdot W$, vectorized with width $W$.
2. **Scalar Cleanup Tail Loop**: Spans $[\text{split}, \text{end})$ with width $W=1$ (scalar execution).

Once split, both the main loop and the tail loop provably satisfy bounds checks without out-of-bounds access or remainder overrun. The loop nest is stamped with:
```
attributes {bounds_check_eliminated = true}
```

## Course Link
- **CS6843 Program Analysis**: Assignment A2 (Range Analysis) and Assignment A4 (Bounds-Check Elimination and Loop Versioning).
