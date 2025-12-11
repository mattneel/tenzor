# Transpose

Transpose operations permute tensor dimensions.

## Basic Transpose

For 2D matrices, transpose swaps rows and columns:

```zig
const A = tz.Tensor(f32, .{ 3, 4 });  // 3x4

// Conceptual transpose (implementation pending)
// const B = a.transpose();  // Would be 4x3

comptime {
    // Result shape would be { 4, 3 }
}
```

## Memory Implications

Transpose changes how elements are accessed:

```
Original [3, 4]:
Memory: [0,0][0,1][0,2][0,3][1,0][1,1][1,2][1,3][2,0][2,1][2,2][2,3]

Transposed [4, 3]:
Logical view of same memory with different strides
```

### Stride-Based Transpose

A "view" transpose just changes strides without copying:

```zig
// Original strides: { 4, 1 }  (step 4 for rows, 1 for cols)
// Transposed strides: { 1, 4 } (step 1 for rows, 4 for cols)
```

This is memory-efficient but may not be SIMD-friendly.

### Copy Transpose

For performance-critical paths, explicit copy may be better:

```zig
// Explicit copy ensures contiguous memory
// Better for subsequent matmul operations
```

## N-Dimensional Transpose

For higher dimensions, specify axis permutation:

```zig
const A = tz.Tensor(f32, .{ 2, 3, 4 });  // Shape: { 2, 3, 4 }

// Conceptual: transpose with axis order
// const B = a.transpose(.{ 0, 2, 1 });  // Shape: { 2, 4, 3 }
// const C = a.transpose(.{ 2, 1, 0 });  // Shape: { 4, 3, 2 }
```

## Common Patterns

### Matrix Transpose for Matmul

```zig
// Attention: Q @ K^T
const A = tz.Tensor(f32, .{ seq, dim });
const B = tz.Tensor(f32, .{ seq, dim });

// Need B transposed to [dim, seq] for matmul
// scores = A @ B^T
```

### Batch Dimension Manipulation

```zig
// NCHW to NHWC
const nchw = tz.Tensor(f32, .{ N, C, H, W });
// nhwc = nchw.transpose(.{ 0, 2, 3, 1 });  // Shape: { N, H, W, C }
```

### Channel-First to Channel-Last

```zig
// For compatibility with different frameworks
const chw = tz.Tensor(f32, .{ C, H, W });
// hwc = chw.transpose(.{ 1, 2, 0 });  // Shape: { H, W, C }
```

## Performance Considerations

### Cache Behavior

Transpose can cause cache misses if accessed row-wise on transposed data:

```
Original [1024, 1024]:
  Row access: sequential, cache-friendly
  Col access: strided, cache-unfriendly

Transposed view:
  "Row" access: now strided
  "Col" access: now sequential
```

### When to Copy

Copy transpose when:
- Data will be used multiple times
- Subsequent operations need contiguous memory
- Matmul performance is critical

Use view transpose when:
- Single-use access
- Memory is constrained
- Access pattern is already strided

## Implementation Status

Transpose is defined in the expression system but full evaluation support is implementation-dependent. The type system validates transpose operations at compile time.

## Next Steps

- [Matrix Multiplication](./matmul.md) - Using transposed matrices
- [Memory Layout](../core/memory-layout.md) - Understanding strides
