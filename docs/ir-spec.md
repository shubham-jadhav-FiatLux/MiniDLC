# MiniDLC Graph IR Specification

The textual graph IR is the contract between the Python frontend and the C++ compiler: `minidlc.ir.export` writes it,
`dlc-opt` reads, optimizes and prints it, and `minidlc.ir.execute` runs it in the NumPy reference interpreter. This
document describes the graph IR as of phase P1 and the v1 loop IR (section 5); the v2 changes to the loop IR are in
[`architecture.md`](architecture.md).

## 1. Example

```dlc
func @mlp(%x: f32[128,784], %y: i32[128], %fc1.weight: f32[784,256] {param}, %fc1.bias: f32[256] {param},
          %fc2.weight: f32[256,10] {param}, %fc2.bias: f32[10] {param}) -> (f32[], f32[128,10]) {
  %0 = matmul(%x, %fc1.weight) : f32[128,256]
  %1 = add(%0, %fc1.bias) : f32[128,256]
  %2 = relu(%1) : f32[128,256]
  %3 = matmul(%2, %fc2.weight) : f32[128,10]
  %4 = add(%3, %fc2.bias) : f32[128,10]
  %5 = cross_entropy(%4, %y) : f32[]
  return %5, %4
}
```

This is `examples/export_ir.py mlp` (the printer writes the signature on one line).

## 2. Grammar

```
module     ::= function*
function   ::= 'func' '@' name '(' [arg (',' arg)*] ')' '->' results '{' statement* '}'
arg        ::= '%' name ':' type ['{' role '}']
role       ::= 'input' | 'param' | 'const' | 'state'
results    ::= type | '(' [type (',' type)*] ')'
type       ::= ('f32' | 'i32') '[' [dim (',' dim)*] ']'
statement  ::= '%' name '=' op '(' [value (',' value)*] ')' [':' type] [attributes]
             | 'return' [value (',' value)* | '(' [value (',' value)*] ')']
attributes ::= '{' [key '=' attr (',' key '=' attr)*] '}'
attr       ::= integer | float | string | 'true' | 'false' | '[' [number (',' number)*] ']'
```

- Value and function names are `[A-Za-z0-9_.$]+`; attribute keys are `[A-Za-z_][A-Za-z0-9_.]*`.
- Numbers: an optional `-`, digits, an optional fraction and exponent; also `nan`, `inf` and `-inf`. Strings use double
  quotes with the escapes `\"`, `\\`, `\n` and `\t`. `//` starts a comment that runs to the end of the line.
- A value is defined once per function (arguments and results share one namespace) and used only after its definition.
  Function names are unique within a module.
- An omitted result type is inferred from the operands and attributes; a written one must equal the inferred one.
- Missing optional attributes get their defaults when the op is created, so the printed form lists every attribute.
- Keys containing a `.` (`fusion.group`, `fusion.kernel`, `mem.offset`, `mem.size`) are **annotations**: passes add them
  for IR dumps, the parser keeps them, and they don't change what the function computes.

### Argument roles

| Role | Holds | Example |
|---|---|---|
| `input` (default, not printed) | data fed on every step | images, labels, token ids |
| `param` | trainable weights | `%fc1.weight` |
| `const` | data fixed when the graph is built | `%causal_mask`, `%positions` |
| `state` | optimizer state that persists across steps (P2) | Adam moments |

Autodiff with activity analysis (P2) will differentiate only `param` arguments; buffer assignment (P3) will keep
`param` and `state` arguments out of the reused arena.

### Types and size limits

`f32` is a 32-bit IEEE float, `i32` a 32-bit signed integer (indices and labels). A tensor type lists its static
dimensions; `f32[]` is a scalar. Dimensions are non-negative, and the product of the dimensions (counting a zero
dimension as one) is at most 2⁴⁸, so element counts, strides and byte sizes cannot overflow.

## 3. Operations

The op registry (`include/dlc/ir/OpRegistry.h`, `lib/ir/OpRegistry.cpp`) holds each op's operand count, attributes and
type rule. The builder, parser, verifier and shape inference all use it, and `dlc-opt --print-op-registry` lists it;
`minidlc.ir.OPS` mirrors it and a test compares the two.

| Op | Operands | Attributes (default) | Result |
|---|---|---|---|
| `add`, `sub`, `mul`, `div` | two `f32` | — | NumPy broadcast of the operand shapes |
| `relu`, `gelu`, `exp`, `log` | one `f32` | — | the operand's type |
| `matmul` | two `f32` | — | `[M,K]×[K,N]→[M,N]`, `[B,M,K]×[K,N]→[B,M,N]`, `[B,M,K]×[B,K,N]→[B,M,N]`, `[B,H,M,K]×[B,H,K,N]→[B,H,M,N]` |
| `softmax`, `log_softmax` | one `f32`, rank ≥ 1 | `axis` int (`-1`) | the operand's type |
| `layernorm` | `x` `f32` of rank ≥ 1; `gamma`, `beta` `f32[D]`, D = last dim of `x` | `eps` float > 0 (`1e-05`) | `x`'s type |
| `cross_entropy` | logits `f32[N,C]` or `f32[B,T,C]`; targets `i32` of the logits' shape without C | — | `f32[]`, the mean over rows |
| `sum`, `mean` | one `f32` | `axes` distinct ints (all axes); `keepdims` 0 or 1 (`0`) | the reduced shape |
| `transpose` | one tensor | `perm` a permutation (reversed axes) | the permuted shape, same element type |
| `reshape` | one tensor | `shape` ints, **required**, same element count | `shape` |
| `embedding` | table `f32[V,D]`, indices `i32` | — | indices' shape + `[D]` |
| `onehot` | indices `i32` | `num_classes` int > 0, **required** | `f32`, indices' shape + `[C]` |
| `constant` | none | `value` float, **required** | the written type (required) |
| `return` | any number | — | none; must be the function's last op |

Negative axes count from the end. Planned for P2, to express backward graphs: `cmp_gt`, `select`, `broadcast_to`,
`reduce_max`, `scatter_add`, `cast`, `tanh`, `rsqrt`, and `transpose_a`/`transpose_b` flags on `matmul`.

## 4. Printing, verification and errors

### Canonical form

The printer (`Function::toString`, `lib/ir/SlotTracker.cpp`) writes:

- arguments under their own names when those are valid (`[A-Za-z_][A-Za-z0-9_.$]*`) and unique, otherwise `argN`;
- op results renumbered `%0`, `%1`, … in program order;
- every result type, and attributes sorted by key;
- floats in the shortest form that reads back as the same double, laid out like Python's `repr` (`0.5`, `1e-05`,
  `1e+16`, `nan`, `-inf`);
- `-> T` for one result and `-> (T1, T2, ...)` otherwise; `return %a, %b` (the parser also accepts `return(%a, %b)`).

Printing, parsing and printing again gives the same text (`tests/lit/roundtrip.dlc`; `IRParser.*` and the mutation
fuzzer in `tests/unit/test_ir.cpp`). The exporter writes this form, so `dlc-opt` with no passes prints an exported
function back unchanged.

### Verifier

`lib/ir/Verifier.cpp` checks:

1. every op is registered, with the registered number of operands and results and no null operand;
2. required attributes are present, every attribute has its declared kind, and other keys are rejected unless they are
   annotations;
3. each result type equals the type the registry infers (broadcasting, matmul dimensions, axes, permutations, reshape
   counts, index types);
4. operands are arguments or results of earlier ops;
5. there is exactly one `return`, it is the last op, and it matches the declared result count and types;
6. each value's use list names exactly the ops that use it.

`dlc-opt` verifies its input (skip with `--no-verify-input`) and, through the pass manager, the module after every pass
(skip both with `--no-verify`).

### Errors

Malformed input raises `IRError` with the position of the offending token; `dlc-opt` prints it and exits with status 1.
`tests/lit/errors.dlc` covers each kind:

```
error: 2:16: use of undefined value '%typo'
error: 2:8: 'add' expects 2 operand(s), got 1
error: 2:30: attribute 'axis' of 'softmax' must be int
error: 4:6: redefinition of function '@f'
error: input failed verification:
  [@f] '%0 = matmul(%a, %b) : f32[4,8]': matmul operand shapes don't match: f32[4,8] x f32[4,8]
```

A pass that leaves the module invalid is named: `verification failed after pass 'canonicalize': ...`; an error raised
inside a pass is prefixed `pass 'autodiff': ...`.

### Python side

`python/minidlc/ir.py`:

- `export(outputs, name, module=None)` turns a traced graph into an `IRFunction`. Placeholders become `input` arguments,
  `nn.Parameter`s `param` arguments (named by `module.named_parameters()`, e.g. `%block_0.attn.q_proj.weight`),
  tensor-valued constants `const` arguments, and `f32` scalars `constant` ops. The data of `param` and `const`
  arguments is kept in `IRFunction.bindings`.
- `parse_module(text)` / `parse_function(text)` read IR text such as `dlc-opt` output; `IRFunction.to_text()` prints it
  the way the C++ printer does.
- `execute(func, feeds)` runs a function in the reference interpreter (`ReferenceInterpreter.forward`).

### Translation validation

`tests/python/test_translation_validation.py` exports 16 graphs (softmax regression, an MLP, a two-layer Char-GPT, a
graph of the remaining ops and patterns, ten random DAGs, and two graphs that trigger known bugs) and runs each through
`shape-infer`, `canonicalize`, `cse`, `dce`, `fusion`, `mem-plan` and the full pipeline. The output of `dlc-opt` is parsed
and executed in float64, and must match the unoptimized function within 1e-12. Autodiff is checked for keeping the loss
and, on the linear model, against the reference gradients. Known miscompiles are strict expected failures, so fixing
one fails the test until the marker is removed.

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

