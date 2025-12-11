# Comptime Magic

Tenzor leverages Zig's comptime capabilities for zero-cost abstractions.

## Compile-Time Computation

### Type Generation

```zig
// Tensor generates a unique type for each shape
pub fn Tensor(comptime T: type, comptime shape: anytype) type {
    return struct {
        pub const ElementType = T;
        pub const ndim = shape.len;
        pub const shape = shape;

        data: [numel()]T,

        pub fn numel() comptime_int {
            var n: comptime_int = 1;
            for (shape) |dim| {
                n *= dim;
            }
            return n;
        }
    };
}
```

### Constant Folding

```zig
// Computed at compile time
const Mat = Tensor(f32, .{ 3, 4 });

comptime {
    // All of these are compile-time constants
    const n = Mat.numel();       // 12
    const s0 = Mat.shape[0];     // 3
    const s1 = Mat.shape[1];     // 4
}
```

## Type-Level Programming

### Expression Types

Operations return types that encode computation:

```zig
const A = Tensor(f32, .{ 3, 4 });
const B = Tensor(f32, .{4});

var a = A{};
var b = B{};

const expr = a.add(b);
// @TypeOf(expr) == BinaryExpr(.add, A, B)
```

### Type Introspection

```zig
fn analyzeExpr(comptime E: type) void {
    comptime {
        if (E.kind == .binary) {
            @compileLog("Binary op:", @tagName(E.operation));
            @compileLog("LHS shape:", E.LhsType.shape);
            @compileLog("RHS shape:", E.RhsType.shape);
        }
    }
}
```

## Comptime Functions

### Shape Computation

```zig
fn BroadcastShape(comptime A: type, comptime B: type) type {
    const a_shape = A.shape;
    const b_shape = B.shape;
    const result_ndim = @max(a_shape.len, b_shape.len);

    var result: [result_ndim]usize = undefined;

    comptime {
        for (0..result_ndim) |i| {
            const a_dim = if (i < a_shape.len) a_shape[a_shape.len - 1 - i] else 1;
            const b_dim = if (i < b_shape.len) b_shape[b_shape.len - 1 - i] else 1;

            if (a_dim != b_dim and a_dim != 1 and b_dim != 1) {
                @compileError("Shapes not broadcastable");
            }

            result[result_ndim - 1 - i] = @max(a_dim, b_dim);
        }
    }

    return struct {
        pub const ndim = result_ndim;
        pub const dimensions = result;
    };
}
```

### Stride Calculation

```zig
fn contiguousStrides(comptime shape: anytype) [shape.len]usize {
    var strides: [shape.len]usize = undefined;

    comptime {
        var stride: usize = 1;
        var i = shape.len;
        while (i > 0) {
            i -= 1;
            strides[i] = stride;
            stride *= shape[i];
        }
    }

    return strides;
}
```

## Inline Loops

### `inline for`

Unrolls loops at compile time:

```zig
fn applyOps(comptime ops: []const OpTag, v: anytype) @TypeOf(v) {
    var result = v;
    inline for (ops) |op| {
        result = applyOp(op, result);  // No loop overhead
    }
    return result;
}
```

### `inline while`

```zig
fn sumDimensions(comptime shape: anytype) comptime_int {
    var sum: comptime_int = 0;
    comptime var i: usize = 0;
    inline while (i < shape.len) : (i += 1) {
        sum += shape[i];
    }
    return sum;
}
```

## Compile-Time Validation

### Shape Checking

```zig
pub fn MatmulExpr(comptime A: type, comptime B: type) type {
    comptime {
        if (A.shape[A.ndim - 1] != B.shape[B.ndim - 2]) {
            @compileError(std.fmt.comptimePrint(
                "Matmul dimension mismatch: {} vs {}",
                .{ A.shape[A.ndim - 1], B.shape[B.ndim - 2] },
            ));
        }
    }

    // Generate result type...
}
```

### Type Checking

```zig
pub fn BinaryExpr(comptime op: OpTag, comptime A: type, comptime B: type) type {
    comptime {
        if (A.ElementType != B.ElementType) {
            @compileError("Element type mismatch");
        }
    }

    // Generate result type...
}
```

## Generic Type Functions

### Type Extraction

```zig
pub fn ElementTypeOf(comptime T: type) type {
    if (@hasDecl(T, "ElementType")) {
        return T.ElementType;
    }
    @compileError("Type has no ElementType");
}

pub fn ShapeOf(comptime T: type) @TypeOf(T.shape) {
    return T.shape;
}

pub fn RankOf(comptime T: type) comptime_int {
    return T.ndim;
}
```

### Type Predicates

```zig
pub fn isExprType(comptime T: type) bool {
    const info = @typeInfo(T);
    if (info != .@"struct") return false;
    return @hasDecl(T, "ExpressionMarker") and T.ExpressionMarker;
}

pub fn isTensorType(comptime T: type) bool {
    const info = @typeInfo(T);
    if (info != .@"struct") return false;
    return @hasDecl(T, "ElementType") and
           @hasDecl(T, "shape") and
           @hasDecl(T, "numel");
}
```

## Comptime Strings

### Error Messages

```zig
comptime {
    @compileError(std.fmt.comptimePrint(
        "Invalid shape: {any}",
        .{invalid_shape},
    ));
}
```

### Type Names

```zig
comptime {
    @compileLog("Type:", @typeName(SomeType));
}
```

## Performance Implications

All comptime code:
- Runs during compilation only
- Has zero runtime cost
- May increase compile time
- Generates specialized code

## Next Steps

- [Shape Algebra](./shape-algebra.md) - Comptime shape operations
- [Type-Level Encoding](../core/type-level-encoding.md) - Expression types
