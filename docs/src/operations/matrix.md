# Matrix Operations

Matrix operations for linear algebra computations.

## Overview

Tenzor provides optimized matrix operations that form the backbone of neural network computations:

| Operation | Description |
|-----------|-------------|
| `matmul` | Matrix multiplication |
| `transpose` | Axis permutation |

## Matrix Multiplication

The most important operation in deep learning.

```zig
const A = tz.Tensor(f32, .{ 3, 4 });  // 3x4 matrix
const B = tz.Tensor(f32, .{ 4, 5 });  // 4x5 matrix

var a = A{};
var b = B{};

const C = a.matmul(b);  // 3x5 matrix

comptime {
    std.debug.assert(C.shape[0] == 3);
    std.debug.assert(C.shape[1] == 5);
}
```

See [Matrix Multiplication](./matmul.md) for detailed coverage.

## Transpose

Permute tensor dimensions.

```zig
const A = tz.Tensor(f32, .{ 3, 4 });  // 3x4 matrix
// Conceptual: A.transpose() would give 4x3
```

See [Transpose](./transpose.md) for details.

## Shape Requirements

### Matmul

Inner dimensions must match:

```zig
// [M, K] @ [K, N] -> [M, N]
const A = tz.Tensor(f32, .{ M, K });
const B = tz.Tensor(f32, .{ K, N });
const C = a.matmul(b);  // Shape: { M, N }
```

Invalid matmul produces compile error:

```zig
const A = tz.Tensor(f32, .{ 3, 4 });
const B = tz.Tensor(f32, .{ 5, 6 });
// const C = a.matmul(b);  // Error: inner dimensions 4 != 5
```

## Common Patterns

### Linear Layer

```zig
// y = Wx + b
const linear = input.matmul(weights).add(bias);
```

### Attention Scores

```zig
// scores = Q @ K^T / sqrt(d_k)
const scores = query.matmul(key_transposed).div(scale);
```

### Batch Matmul

For batched operations:

```zig
const A = tz.Tensor(f32, .{ batch, M, K });
const B = tz.Tensor(f32, .{ batch, K, N });
const C = a.matmul(b);  // Shape: { batch, M, N }
```

## Performance Considerations

Matrix operations are heavily optimized:

1. **Tiled execution** - Cache-friendly blocking
2. **SIMD vectorization** - Multiple elements per instruction
3. **Epilogue fusion** - Bias and activation fused with matmul

```zig
// This is a single fused kernel:
const output = input.matmul(weights).add(bias).relu();
```

## Memory Layout

Matrices use row-major layout:

```
[3, 4] matrix:
  Row 0: [0,0] [0,1] [0,2] [0,3]
  Row 1: [1,0] [1,1] [1,2] [1,3]
  Row 2: [2,0] [2,1] [2,2] [2,3]

Memory: [0,0][0,1][0,2][0,3][1,0][1,1]...
```

This layout is optimal for matmul's access patterns.

## Next Steps

- [Matrix Multiplication](./matmul.md) - Detailed matmul reference
- [Transpose](./transpose.md) - Dimension permutation
- [Fusion Engine](../fusion/overview.md) - Matmul epilogue fusion
