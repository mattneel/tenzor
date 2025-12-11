# Shape and Dimensions

Shapes are central to tenzor's compile-time guarantees. This chapter covers shape representation, validation, and algebra.

## Shape Representation

Shapes are compile-time tuples:

```zig
const tz = @import("tenzor");

const Scalar = tz.Tensor(f32, .{});          // 0-D: scalar
const Vector = tz.Tensor(f32, .{4});          // 1-D: vector
const Matrix = tz.Tensor(f32, .{ 3, 4 });     // 2-D: matrix
const Tensor3D = tz.Tensor(f32, .{ 2, 3, 4 }); // 3-D: tensor
```

## Dimension Properties

### Rank (ndim)

The number of dimensions:

```zig
comptime {
    std.debug.assert(Scalar.ndim == 0);
    std.debug.assert(Vector.ndim == 1);
    std.debug.assert(Matrix.ndim == 2);
    std.debug.assert(Tensor3D.ndim == 3);
}
```

### Shape Array

Access individual dimensions:

```zig
const T = tz.Tensor(f32, .{ 2, 3, 4 });

comptime {
    std.debug.assert(T.shape[0] == 2);
    std.debug.assert(T.shape[1] == 3);
    std.debug.assert(T.shape[2] == 4);
}
```

### Number of Elements

Total element count:

```zig
const T = tz.Tensor(f32, .{ 2, 3, 4 });

comptime {
    std.debug.assert(T.numel() == 24);  // 2 * 3 * 4
}
```

## Shape Algebra

### Broadcasting

Tenzor implements NumPy-style broadcasting rules at compile time:

```zig
// Broadcasting rules:
// 1. Shapes are compared right-to-left
// 2. Dimensions match if equal or one is 1
// 3. Missing dimensions are treated as 1

const A = tz.Tensor(f32, .{ 3, 4 });    // [3, 4]
const B = tz.Tensor(f32, .{4});          // [4]
const C = A.add(B);                       // [3, 4]

const D = tz.Tensor(f32, .{ 3, 1 });     // [3, 1]
const E = A.mul(D);                       // [3, 4]
```

#### Broadcasting Examples

| Shape A | Shape B | Result | Valid? |
|---------|---------|--------|--------|
| `{3, 4}` | `{4}` | `{3, 4}` | Yes |
| `{3, 4}` | `{3, 1}` | `{3, 4}` | Yes |
| `{3, 4}` | `{1, 4}` | `{3, 4}` | Yes |
| `{3, 4}` | `{3, 4}` | `{3, 4}` | Yes |
| `{3, 4}` | `{5}` | - | No |
| `{3, 4}` | `{2, 4}` | - | No |

### Matrix Multiplication Shapes

```zig
const A = tz.Tensor(f32, .{ M, K });
const B = tz.Tensor(f32, .{ K, N });
const C = A.matmul(B);  // Shape: { M, N }

// Batched matmul
const D = tz.Tensor(f32, .{ batch, M, K });
const E = tz.Tensor(f32, .{ batch, K, N });
const F = D.matmul(E);  // Shape: { batch, M, N }
```

### Reduction Shapes

```zig
const T = tz.Tensor(f32, .{ 2, 3, 4 });

// Full reduction
const sum_all = T.sum(.{});           // Shape: {}

// Partial reduction
const sum_0 = T.sum(.{0});            // Shape: { 3, 4 }
const sum_1 = T.sum(.{1});            // Shape: { 2, 4 }
const sum_2 = T.sum(.{2});            // Shape: { 2, 3 }

// Multi-axis reduction
const sum_01 = T.sum(.{ 0, 1 });      // Shape: { 4 }

// Keepdims (not yet implemented)
// const sum_k = T.sum(.{0}, .{ .keepdims = true });  // Shape: { 1, 3, 4 }
```

## Shape Validation

### Compile-Time Errors

Invalid shapes produce clear compile errors:

```zig
const A = tz.Tensor(f32, .{ 3, 4 });
const B = tz.Tensor(f32, .{ 5, 6 });

// This would produce:
// error: Shapes not broadcast compatible: { 3, 4 } vs { 5, 6 }
const C = A.add(B);
```

### Matmul Dimension Checks

```zig
const A = tz.Tensor(f32, .{ 3, 4 });
const B = tz.Tensor(f32, .{ 5, 6 });

// error: Matmul inner dimensions must match: 4 vs 5
const C = A.matmul(B);
```

## Shape Utilities

### Computing Broadcast Shape

```zig
const core = @import("tenzor").core;

const ShapeA = core.shape.Shape(.{ 3, 1, 4 });
const ShapeB = core.shape.Shape(.{ 5, 4 });

const Result = core.shape.BroadcastShape(ShapeA, ShapeB);
// Result.dimensions == .{ 3, 5, 4 }
```

### Checking Compatibility

```zig
const compatible = core.shape.broadcastCompatible(
    core.shape.Shape(.{ 3, 4 }),
    core.shape.Shape(.{ 4 }),
);
// compatible == true
```

### Computing Strides

```zig
const strides = core.strides.contiguousStrides(.{ 2, 3, 4 });
// strides == .{ 12, 4, 1 }
```

## Common Shape Patterns

### Flattening

```zig
// Conceptual: [2, 3, 4] -> [24]
const T3D = tz.Tensor(f32, .{ 2, 3, 4 });
const Flat = tz.Tensor(f32, .{24});

comptime {
    std.debug.assert(T3D.numel() == Flat.numel());
}
```

### Reshaping

```zig
// Conceptual: [12] -> [3, 4] -> [2, 6]
const A = tz.Tensor(f32, .{12});
const B = tz.Tensor(f32, .{ 3, 4 });
const C = tz.Tensor(f32, .{ 2, 6 });

comptime {
    std.debug.assert(A.numel() == B.numel());
    std.debug.assert(B.numel() == C.numel());
}
```

### Adding Dimensions

```zig
// Conceptual: [4] -> [1, 4] for row vector
// Conceptual: [4] -> [4, 1] for column vector
const Vec = tz.Tensor(f32, .{4});
const Row = tz.Tensor(f32, .{ 1, 4 });
const Col = tz.Tensor(f32, .{ 4, 1 });
```

## Shape in Expression Types

Shapes flow through expressions:

```zig
const A = tz.Tensor(f32, .{ 3, 4 });
const B = tz.Tensor(f32, .{ 4, 5 });

var a = A{};
var b = B{};

const expr = a.matmul(b).relu();

comptime {
    std.debug.assert(expr.shape[0] == 3);
    std.debug.assert(expr.shape[1] == 5);
}
```

## Next Steps

- [Data Types](./dtypes.md) - Supported element types
- [Memory Layout](./memory-layout.md) - How shapes map to memory
- [Broadcasting](../advanced/broadcasting.md) - Advanced broadcasting
