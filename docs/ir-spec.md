# MiniDLC Graph IR Specification

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

## 4. Verification Rules

1. **SSA Dominance**: Every operand SSA value `%v` must be defined before its use in execution order.
2. **Type Consistency**: Binary elementwise operations require operand shapes compatible under broadcasting.
3. **Static Shapes**: Tensor dimensions must be strictly non-negative integers.
4. **Single Return**: A function must terminate with a valid `return` operation matching the declared signature.
