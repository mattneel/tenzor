# Tensors

Tensors are the fundamental data structure in tenzor. Unlike runtime tensor libraries, tenzor encodes tensor properties in the type system.

## The Tensor Type

```zig
pub fn Tensor(comptime T: type, comptime shape_tuple: anytype) type
```

This function generates a unique type for each combination of element type and shape:

```zig
const tz = @import("tenzor");

const Vec3 = tz.Tensor(f32, .{3});           // Different types
const Vec4 = tz.Tensor(f32, .{4});           // Different types
const Mat2x3 = tz.Tensor(f32, .{ 2, 3 });    // Different types
const Mat2x3_f64 = tz.Tensor(f64, .{ 2, 3 }); // Different types
```

## Tensor Properties

Every tensor type has these compile-time properties:

| Property | Type | Description |
|----------|------|-------------|
| `ElementType` | `type` | The scalar element type |
| `ndim` | `comptime_int` | Number of dimensions |
| `shape` | `[ndim]usize` | Size along each dimension |
| `strides` | `[ndim]usize` | Memory stride per dimension |
| `numel()` | `usize` | Total number of elements |

```zig
const Mat = tz.Tensor(f32, .{ 3, 4 });

comptime {
    _ = Mat.ElementType;  // f32
    _ = Mat.ndim;         // 2
    _ = Mat.shape;        // .{ 3, 4 }
    _ = Mat.strides;      // .{ 4, 1 }
    _ = Mat.numel();      // 12
}
```

## Tensor Instances

A tensor instance contains the actual data:

```zig
const Vec4 = tz.Tensor(f32, .{4});

var v1 = Vec4{};                           // Uninitialized
var v2 = Vec4{ .data = .{ 1, 2, 3, 4 } };  // Initialized
```

The `data` field is a fixed-size array:

```zig
const Mat = tz.Tensor(f32, .{ 2, 3 });
// Mat.data is [6]f32
```

## Tensor Operations

Tensors support operations through methods:

```zig
const A = tz.Tensor(f32, .{ 2, 3 });
const B = tz.Tensor(f32, .{ 2, 3 });

var a = A{};
var b = B{};

// Operations return expression types, not tensors
const sum = a.add(b);        // BinaryExpr(.add, A, B)
const prod = a.mul(b);       // BinaryExpr(.mul, A, B)
const activated = a.relu();  // UnaryExpr(.relu, A)
```

## Tensor vs Expression

It's important to understand the distinction:

| Concept | Contains Data | Lazy | Evaluatable |
|---------|--------------|------|-------------|
| Tensor | Yes | No | Direct access |
| Expression | No | Yes | Via `eval()` |

```zig
const Vec = tz.Tensor(f32, .{4});
var tensor = Vec{};            // Has data
tensor.data[0] = 1.0;          // Direct access

const expr = tensor.relu();    // No data, just describes operation
const result = tz.eval(expr);  // Evaluates and returns data
```

## Type-Level Guarantees

The type system enforces correctness:

```zig
const A = tz.Tensor(f32, .{ 3, 4 });
const B = tz.Tensor(f32, .{ 4, 5 });

// Valid: inner dimensions match (4 == 4)
const C = A.matmul(B);  // Result: Tensor(f32, .{ 3, 5 })

// Invalid: would fail at compile time
// const D = B.matmul(A);  // Error: 5 != 3
```

## Memory Considerations

Tensors with large shapes should be heap-allocated:

```zig
// Stack allocation (okay for small tensors)
const Small = tz.Tensor(f32, .{ 4, 4 });
var small = Small{};

// For large tensors, consider external storage
const Large = tz.Tensor(f32, .{ 1024, 1024 });
// Large.data would be 4MB - too large for stack
```

## Tensor Aliasing

Multiple tensor instances can share the same type:

```zig
const Vec = tz.Tensor(f32, .{4});

var v1 = Vec{};
var v2 = Vec{};
var v3 = Vec{};

// All have the same type, different storage
```

## Generic Programming

Write functions that work with any tensor:

```zig
fn printShape(comptime T: type) void {
    if (@hasDecl(T, "shape")) {
        std.debug.print("Shape: {any}\n", .{T.shape});
    }
}

const A = tz.Tensor(f32, .{ 2, 3, 4 });
printShape(A);  // Prints: Shape: { 2, 3, 4 }
```

## Next Steps

- [Creating Tensors](./creating-tensors.md) - Tensor initialization patterns
- [Shapes](./shapes.md) - Shape algebra and validation
- [Data Types](./dtypes.md) - Supported element types
