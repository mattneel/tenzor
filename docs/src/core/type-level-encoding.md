# Type-Level Encoding

Tenzor encodes expression graphs in the type system. This enables compile-time validation, optimization, and zero-cost abstractions.

## Types as Computation Graphs

Each operation returns a new type that describes:
- The operation performed
- The input types
- The result shape
- The element type

```zig
const A = tz.Tensor(f32, .{ 3, 4 });
const B = tz.Tensor(f32, .{4});

var a = A{};
var b = B{};

const expr = a.add(b);

// expr has type: BinaryExpr(.add, A, B)
```

## Anatomy of an Expression Type

### UnaryExpr

```zig
pub fn UnaryExpr(comptime op: OpTag, comptime Input: type) type {
    return struct {
        pub const ExpressionMarker = true;
        pub const kind: NodeKind = .unary;
        pub const operation: OpTag = op;
        pub const InputType = Input;
        pub const ElementType = ElementTypeOf(Input);
        pub const ndim = RankOf(Input);
        pub const shape = ShapeOf(Input);

        input: Input,
    };
}
```

### BinaryExpr

```zig
pub fn BinaryExpr(comptime op: OpTag, comptime Lhs: type, comptime Rhs: type) type {
    // Compile-time validation
    comptime {
        if (ElementTypeOf(Lhs) != ElementTypeOf(Rhs)) {
            @compileError("Type mismatch");
        }
        if (!broadcastCompatible(ShapeOf(Lhs), ShapeOf(Rhs))) {
            @compileError("Shapes not compatible");
        }
    }

    return struct {
        pub const ExpressionMarker = true;
        pub const kind: NodeKind = .binary;
        pub const operation: OpTag = op;
        pub const LhsType = Lhs;
        pub const RhsType = Rhs;
        pub const ElementType = ElementTypeOf(Lhs);
        pub const shape = BroadcastShape(Lhs, Rhs);

        lhs: Lhs,
        rhs: Rhs,
    };
}
```

## Type Nesting

Expressions nest naturally:

```zig
const a = tensor_a;                    // Tensor(f32, .{4})
const b = a.relu();                    // UnaryExpr(.relu, Tensor)
const c = b.add(tensor_b);             // BinaryExpr(.add, UnaryExpr, Tensor)
const d = c.mul(tensor_c);             // BinaryExpr(.mul, BinaryExpr, Tensor)

// Type of d:
// BinaryExpr(.mul,
//   BinaryExpr(.add,
//     UnaryExpr(.relu, Tensor(f32, .{4})),
//     Tensor(f32, .{4})
//   ),
//   Tensor(f32, .{4})
// )
```

## Compile-Time Properties

All properties are available at compile time:

```zig
const expr = tensor.matmul(weights).add(bias).relu();

comptime {
    // Navigate the type structure
    const MatmulType = @TypeOf(expr).InputType.InputType.LhsType;

    // Access properties
    const result_shape = @TypeOf(expr).shape;
    const elem_type = @TypeOf(expr).ElementType;

    // Validate
    std.debug.assert(result_shape[0] == 1);
    std.debug.assert(elem_type == f32);
}
```

## Type-Based Dispatch

Functions can dispatch based on expression types:

```zig
fn evaluate(expr: anytype) void {
    const T = @TypeOf(expr);

    switch (T.kind) {
        .tensor => evaluateTensor(expr),
        .unary => {
            evaluate(expr.input);
            applyUnary(T.operation);
        },
        .binary => {
            evaluate(expr.lhs);
            evaluate(expr.rhs);
            applyBinary(T.operation);
        },
        .matmul => evaluateMatmul(expr),
        .reduce => evaluateReduce(expr),
    }
}
```

## Zero-Cost Abstraction

The type encoding has no runtime overhead:

```zig
// At compile time: complex nested type
const expr = a.add(b).mul(c).relu();

// At runtime: just function calls
// The type structure is erased, only the operations remain
const result = tz.eval(expr, allocator);
```

## Type Introspection

Examine expressions programmatically:

```zig
fn printExprTree(comptime T: type, indent: usize) void {
    const spaces = "                    "[0..indent];

    if (T.kind == .tensor) {
        @compileLog(spaces, "Tensor", T.shape);
    } else if (T.kind == .unary) {
        @compileLog(spaces, "Unary", @tagName(T.operation));
        printExprTree(T.InputType, indent + 2);
    } else if (T.kind == .binary) {
        @compileLog(spaces, "Binary", @tagName(T.operation));
        printExprTree(T.LhsType, indent + 2);
        printExprTree(T.RhsType, indent + 2);
    }
}
```

## Expression Markers

Types are identified as expressions via markers:

```zig
pub fn isExprType(comptime T: type) bool {
    if (@typeInfo(T) != .@"struct") return false;
    return @hasDecl(T, "ExpressionMarker") and T.ExpressionMarker;
}

pub fn isTensorType(comptime T: type) bool {
    if (@typeInfo(T) != .@"struct") return false;
    return @hasDecl(T, "ElementType") and
           @hasDecl(T, "shape") and
           @hasDecl(T, "numel");
}
```

## Generic Expression Handling

Write code that works with any expression:

```zig
fn getResultShape(comptime E: type) []const usize {
    if (isExprType(E) or isTensorType(E)) {
        return &E.shape;
    }
    @compileError("Not an expression type");
}

fn getElementType(comptime E: type) type {
    return E.ElementType;
}

fn countOps(comptime E: type) usize {
    if (E.kind == .tensor or E.kind == .constant) {
        return 0;
    } else if (E.kind == .unary) {
        return 1 + countOps(E.InputType);
    } else if (E.kind == .binary) {
        return 1 + countOps(E.LhsType) + countOps(E.RhsType);
    }
    return 1;
}
```

## Benefits of Type-Level Encoding

1. **Compile-time validation** - Errors caught before runtime
2. **Zero overhead** - Types erased at runtime
3. **Full visibility** - Optimizer sees entire graph
4. **Type safety** - Operations on correct types only
5. **IDE support** - Type information for tooling

## Limitations

1. **Compile time** - Complex graphs increase compilation
2. **Fixed topology** - Graph structure is static
3. **No runtime dispatch** - Can't choose operations at runtime

## Next Steps

- [Fusion Engine](../fusion/overview.md) - How types enable optimization
- [Comptime Magic](../advanced/comptime.md) - Advanced type-level techniques
