# Expression Graphs

Tenzor uses expression graphs to represent computations. Unlike eager execution, operations build a graph that is evaluated on demand.

## What is an Expression Graph?

An expression graph is a directed acyclic graph (DAG) where:
- **Nodes** represent operations or data
- **Edges** represent data flow

```zig
const a = tensor_a;           // Leaf node
const b = tensor_b;           // Leaf node
const c = a.add(b);           // Operation node: Add(a, b)
const d = c.relu();           // Operation node: ReLU(c)
const e = d.mul(tensor_e);    // Operation node: Mul(d, e)
```

Visualized:

```
    tensor_a    tensor_b    tensor_e
        \          /            |
         \        /             |
          Add(a,b)              |
              |                 |
           ReLU(c)              |
              \                /
               \              /
                Mul(d, e)
                    |
                    e
```

## Expression Types

Each operation returns a specific expression type:

### UnaryExpr

Single-input operations:

```zig
const UnaryExpr = @import("tenzor").ops.expr.UnaryExpr;

const input = tensor.relu();
// Type: UnaryExpr(.relu, TensorType)

comptime {
    std.debug.assert(input.kind == .unary);
    std.debug.assert(input.operation == .relu);
}
```

### BinaryExpr

Two-input operations:

```zig
const BinaryExpr = @import("tenzor").ops.expr.BinaryExpr;

const sum = tensor_a.add(tensor_b);
// Type: BinaryExpr(.add, TensorTypeA, TensorTypeB)

comptime {
    std.debug.assert(sum.kind == .binary);
    std.debug.assert(sum.operation == .add);
}
```

### MatmulExpr

Matrix multiplication:

```zig
const MatmulExpr = @import("tenzor").ops.expr.MatmulExpr;

const product = matrix_a.matmul(matrix_b);
// Type: MatmulExpr(MatrixA, MatrixB)

comptime {
    std.debug.assert(product.kind == .matmul);
}
```

### ReduceExpr

Reduction operations:

```zig
const ReduceExpr = @import("tenzor").ops.expr.ReduceExpr;

const sum = tensor.sum(.{0});
// Type: ReduceExpr(.sum, TensorType, .{0}, false)

comptime {
    std.debug.assert(sum.kind == .reduce);
    std.debug.assert(sum.operation == .sum);
}
```

## Node Kinds

Every expression has a `kind` that identifies its type:

```zig
const NodeKind = enum {
    tensor,    // Leaf: concrete tensor with data
    constant,  // Leaf: compile-time constant
    unary,     // Unary operation
    binary,    // Binary operation
    matmul,    // Matrix multiplication
    reduce,    // Reduction operation
    reshape,   // Shape manipulation
    transpose, // Axis permutation
};
```

## Composing Expressions

Expressions can be freely composed:

```zig
// Build a complex expression
const expr = input
    .matmul(weights)        // MatmulExpr
    .add(bias)              // BinaryExpr(.add, ...)
    .relu()                 // UnaryExpr(.relu, ...)
    .mul(scale)             // BinaryExpr(.mul, ...)
    .sum(.{1});             // ReduceExpr(.sum, ...)

// All compile-time, no computation yet
```

## Type-Level Encoding

The expression graph is encoded in types:

```zig
const A = tz.Tensor(f32, .{ 3, 4 });
const B = tz.Tensor(f32, .{4});

var a = A{};
var b = B{};

const expr = a.add(b);

// expr has type: BinaryExpr(.add, A, B)
// This type contains:
// - The operation (.add)
// - Left operand type (A)
// - Right operand type (B)
// - Result shape (.{ 3, 4 })
```

## Inspecting Expressions

Query expression properties at compile time:

```zig
const expr = tensor.relu().add(other);

comptime {
    // Expression kind
    std.debug.assert(expr.kind == .binary);

    // Operation
    std.debug.assert(expr.operation == .add);

    // Result properties
    std.debug.assert(expr.ElementType == f32);
    std.debug.assert(expr.ndim == 2);

    // Input types
    const LhsType = expr.LhsType;  // UnaryExpr(.relu, ...)
    const RhsType = expr.RhsType;  // TensorType
}
```

## Expression vs Tensor

| Feature | Tensor | Expression |
|---------|--------|------------|
| Has data | Yes | No |
| Lazy | No | Yes |
| Shape known | Compile-time | Compile-time |
| Can chain ops | Yes | Yes |
| Evaluatable | Direct | Via `eval()` |

## Sharing Subexpressions

Expressions can share common subexpressions:

```zig
const a = tensor.exp();
const b = a.add(one);      // Uses a
const c = a.mul(two);      // Reuses a

// When evaluated, 'a' is computed once
```

## Next Steps

- [Lazy Evaluation](./lazy-evaluation.md) - When computation happens
- [Type-Level Encoding](./type-level-encoding.md) - How types encode the graph
- [Fusion Engine](../fusion/overview.md) - Optimizing expression graphs
