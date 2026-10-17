# MiniDLC Graph IR Specification

This describes the v1 graph IR and loop IR as implemented. The v2 changes — an op registry, backward primitives,
explicit guards, thread binding and memory scopes in the loop IR — are described in [`architecture.md`](architecture.md).

## 1. Syntax and Grammar

The Graph IR is a textual, SSA-based language representing dataflow graphs of tensor operations.

### Example Function

```dlc
func @mlp_forward(%x: f32[128,784], %w1: f32[784,256], %b1: f32[256]) -> f32[128,256] {
  %0 = matmul(%x, %w1) : f32[128,256]
  %1 = add(%0, %b1) : f32[128,256]
  %2 = relu(%1) : f32[128,256]
  return %2
}
```

## 2. Type System

- **Primitive Types**:
  - `f32`: 32-bit IEEE 754 floating point.
  - `i32`: 32-bit signed integer (indices, categorical labels).
- **Tensor Types**:
  - `f32[d0, d1, ...]` represents a multidimensional tensor with static integer dimensions.
  - `f32[]`: 0-D scalar tensor.

## 3. Operations

| Op Category | Operations | Description |
|-------------|------------|-------------|
| Linear Algebra | `matmul` | Matrix multiplication (including batched 3D) |
| Binary Elementwise | `add`, `sub`, `mul`, `div` | Elementwise with NumPy-style broadcasting |
| Unary Elementwise | `relu`, `gelu`, `exp`, `log` | Unary functions |
| Reduction | `sum`, `mean` | Reduction along specified axes |
| Normalization | `softmax`, `log_softmax`, `layernorm` | Normalization along specified axes |
| Loss Functions | `cross_entropy` | Categorical cross entropy with integer targets |
| Manipulation | `transpose`, `reshape` | Tensor transposition and dimension reshapes |
| Indexing & Memory | `embedding`, `constant`, `onehot` | Table gather, constant literals, one-hot encoding |

Planned for P2 (needed to express backward graphs): `cmp_gt`, `select`, `broadcast_to`, `reduce_sum(axes)`,
`reduce_max(axes)`, `scatter_add`, `cast`, `tanh`, `rsqrt`, and `transpose_a`/`transpose_b` flags on `matmul`.

## 4. Verification Rules

Checked by the v1 verifier (`lib/ir/Verifier.cpp`):

1. **Def before use:** every operand is a function argument or the result of an earlier op.
2. **Terminator:** a non-empty function contains a `return`.

Not yet checked (P1): per-op type and shape rules (broadcast compatibility, matmul inner dimensions, non-negative
dimensions), the returned values against the declared signature, and that `return` is the last op.

### Parser limitations (v1)

- Attribute dictionaries such as `{axis = 0}` are skipped. Ops get default attributes (`axis = -1`, `eps = 1e-5`), and
  ops without a dedicated builder (`transpose`, `reshape`, `constant`, `onehot`) are created with no attributes.
- An operand name that was never defined is dropped silently.
- `return` takes a single value written as `return %v`. The printer's `return(%v)` form does not re-parse, and
  `return %a, %b` makes the parser loop forever.

These are fixed in P1, together with a print → parse → print round-trip test.

## 5. Loop IR Specification

Lowered from fused Graph IR clusters, the Loop IR models loop nests, memory buffers, affine index access patterns, and hardware mapping directives.

### Syntax Example

```dlc
kernel @fused_matmul_bias_relu {
  buffer %arg0 : offset 802816 bytes, size 401408 bytes, shape [128, 784]
  buffer %arg1 : offset 0 bytes, size 802816 bytes, shape [784, 256]
  buffer %arg2 : offset 1335296 bytes, size 1024 bytes, shape [256]
  buffer %2 : offset 131072 bytes, size 131072 bytes, shape [128, 256]
  attributes {bounds_check_eliminated = true}
  parallel for i in 0..128 {
    for j in 0..256 {
      acc = 0.0;
      for k in 0..784 vectorize(8) {
        acc = fma(arg0[((i * 784) + k)], arg1[((k * 256) + j)], acc);
      }
      2[((i * 256) + j)] = max((acc + arg2[j]), 0.0);
    }
  }
}
```

> **Two v1 defects are visible in this dump.** The output buffer `%2` (bytes 131072–262143) lies inside the weight
> buffer `%arg1` (bytes 0–802815) that the same kernel reads, because memory planning runs on the unfused op
> schedule; executing this kernel gives wrong results. And `vectorize(8)` is placed on the reduction loop `k`, whose
> accesses to `arg1` have stride 256, so it is not a legal contiguous vectorization (the LLVM template ignores the
> annotation and vectorizes over `j`). The v2 loop IR addresses both.

### Loop IR AST Constructs

1. **KernelNest**: Container containing:
   - `name`: Kernel symbol name.
   - `buffers`: List of memory buffers allocated in the contiguous arena with byte offsets and shapes.
   - `rootStmts`: Root statement sequence.
   - `boundsCheckEliminated`: Boolean attribute indicating static range-analysis verification. v1 emits no guards,
     so the flag does not change generated code.

2. **Statements (`LoopStmt`)**:
   - `ForLoop(var, start, end, step)`: Iteration construct with attributes:
     - `parallel`: Enables multi-threaded or multi-block scheduling.
     - `vectorize(W)`: Directs SIMD generation with vector width $W$ (e.g. 8 for AVX2).
     - `unroll(U)`: Directs loop unrolling with factor $U$.
   - `AssignStmt(var, expr)`: Local register/variable assignment.
   - `StoreStmt(buffer, indexExpr, valueExpr)`: Memory store into a designated arena buffer.

3. **Expressions (`Expr`)**:
   - `VarExpr(name)`: Variable or induction symbol reference.
   - `ConstExpr(val)`: Floating-point or integer constant.
   - `LoadExpr(buffer, indexExpr)`: Memory read from an arena buffer at an affine index.
   - `BinaryExpr(op, lhs, rhs)`: Arithmetic operations (`+`, `-`, `*`, `/`, `max`, `min`).
   - `UnaryExpr(op, operand)`: Unary functions (`exp`, `log`, `relu`, `sqrt`, `neg`).
   - `FMAExpr(a, b, c)`: Fused multiply-accumulate evaluating $a \cdot b + c$.

