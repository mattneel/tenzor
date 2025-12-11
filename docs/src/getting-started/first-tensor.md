# Your First Tensor

This chapter explores tensor creation and manipulation in depth.

## What is a Tensor?

In tenzor, a **Tensor** is a compile-time type that describes:
- The element type (e.g., `f32`, `f64`)
- The shape (e.g., `{3, 4}` for a 3x4 matrix)
- The memory layout (row-major by default)

```zig
const tz = @import("tenzor");

// A 3x4 matrix of 32-bit floats
const Matrix = tz.Tensor(f32, .{ 3, 4 });

// Access compile-time properties
const ndim = Matrix.ndim;        // 2
const shape = Matrix.shape;      // .{ 3, 4 }
const numel = Matrix.numel();    // 12
const T = Matrix.ElementType;    // f32
```

## Creating Tensor Instances

### Stack Allocation

For small tensors, allocate on the stack:

```zig
const Vec4 = tz.Tensor(f32, .{4});

var vec = Vec4{};
vec.data = .{ 1.0, 2.0, 3.0, 4.0 };
```

### With Initialization

```zig
const Vec4 = tz.Tensor(f32, .{4});

var zeros = Vec4{};
@memset(&zeros.data, 0.0);

var ones = Vec4{};
@memset(&ones.data, 1.0);

// Copy from a slice
var from_slice = Vec4{};
const source = [_]f32{ 1, 2, 3, 4 };
@memcpy(&from_slice.data, &source);
```

### Multidimensional Tensors

Higher-dimensional tensors work the same way:

```zig
const Image = tz.Tensor(f32, .{ 3, 224, 224 });  // CHW format
const Batch = tz.Tensor(f32, .{ 32, 3, 224, 224 }); // NCHW format

var image = Image{};
// image.data is a flat array of 3 * 224 * 224 = 150528 elements
```

## Understanding Memory Layout

Tensors use **row-major** (C-style) ordering. For a `{3, 4}` matrix:

```
Logical view:          Memory layout:
┌─────────────────┐    [0,0] [0,1] [0,2] [0,3] [1,0] [1,1] ...
│ 0,0  0,1  0,2  0,3 │
│ 1,0  1,1  1,2  1,3 │
│ 2,0  2,1  2,2  2,3 │
└─────────────────┘
```

Access elements via linear indexing:

```zig
const Mat = tz.Tensor(f32, .{ 3, 4 });
var mat = Mat{};

// Element at [row, col] is at index: row * 4 + col
mat.data[0 * 4 + 2] = 5.0;  // Set [0, 2] = 5.0
mat.data[2 * 4 + 3] = 7.0;  // Set [2, 3] = 7.0
```

## Strides

Strides describe the memory step between elements along each dimension:

```zig
const Mat = tz.Tensor(f32, .{ 3, 4 });
const strides = Mat.strides;  // .{ 4, 1 }

// To move one row: step by 4 elements
// To move one column: step by 1 element
```

For higher dimensions:

```zig
const Tensor3D = tz.Tensor(f32, .{ 2, 3, 4 });
const strides = Tensor3D.strides;  // .{ 12, 4, 1 }

// dim 0: step by 12 (3 * 4)
// dim 1: step by 4
// dim 2: step by 1
```

## Scalar Tensors

Zero-dimensional tensors represent scalars:

```zig
const Scalar = tz.Tensor(f32, .{});

var s = Scalar{};
s.data[0] = 42.0;

comptime {
    std.debug.assert(Scalar.ndim == 0);
    std.debug.assert(Scalar.numel() == 1);
}
```

## Type Introspection

Query tensor properties at compile time:

```zig
const Mat = tz.Tensor(f32, .{ 3, 4 });

comptime {
    // Dimensionality
    std.debug.assert(Mat.ndim == 2);

    // Shape
    std.debug.assert(Mat.shape[0] == 3);
    std.debug.assert(Mat.shape[1] == 4);

    // Total elements
    std.debug.assert(Mat.numel() == 12);

    // Element type
    std.debug.assert(Mat.ElementType == f32);

    // Size in bytes
    std.debug.assert(Mat.size_bytes() == 12 * 4);
}
```

## Working with Data

### Iterating Elements

```zig
const Vec = tz.Tensor(f32, .{4});
var vec = Vec{};
vec.data = .{ 1, 2, 3, 4 };

var sum: f32 = 0;
for (vec.data) |x| {
    sum += x;
}
```

### Applying Transformations

```zig
const Vec = tz.Tensor(f32, .{4});
var vec = Vec{};

for (&vec.data) |*x| {
    x.* = @sin(x.*);
}
```

### SIMD-Friendly Access

```zig
const Vec = tz.Tensor(f32, .{16});
var vec = Vec{};

// Process in SIMD-width chunks
const simd_width = 4;
var i: usize = 0;
while (i < 16) : (i += simd_width) {
    const chunk = vec.data[i..][0..simd_width];
    // Process chunk with @Vector operations
}
```

## Common Patterns

### Creating Test Data

```zig
fn linspace(comptime N: usize) [N]f32 {
    var result: [N]f32 = undefined;
    for (&result, 0..) |*x, i| {
        x.* = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(N - 1));
    }
    return result;
}

const Vec = tz.Tensor(f32, .{10});
var vec = Vec{};
vec.data = linspace(10);  // [0.0, 0.111, 0.222, ..., 1.0]
```

### Reshaping (Conceptual)

Tenzor validates reshape compatibility at compile time:

```zig
const Flat = tz.Tensor(f32, .{12});
const Mat = tz.Tensor(f32, .{ 3, 4 });
const Cube = tz.Tensor(f32, .{ 2, 2, 3 });

// All have the same numel (12), so reshaping is valid
comptime {
    std.debug.assert(Flat.numel() == Mat.numel());
    std.debug.assert(Mat.numel() == Cube.numel());
}
```

## Next Steps

- [Shape and Dimensions](../core/shapes.md) - Deep dive into shape algebra
- [Expression Graphs](../core/expression-graphs.md) - Build computations
- [Operations](../operations/overview.md) - Available operations
