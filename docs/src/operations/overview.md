# Operations Overview

Tenzor provides a comprehensive set of operations for tensor computation.

## Operation Categories

### Unary Operations

Single-input operations that transform each element:

```zig
const y = x.relu();    // Activation
const y = x.exp();     // Math function
const y = x.neg();     // Negation
```

### Binary Operations

Two-input operations with broadcasting:

```zig
const c = a.add(b);    // Addition
const c = a.mul(b);    // Multiplication
const c = a.pow(b);    // Power
```

### Matrix Operations

Linear algebra operations:

```zig
const c = a.matmul(b); // Matrix multiplication
```

### Reductions

Aggregate values along dimensions:

```zig
const s = x.sum(.{});      // Total sum
const m = x.mean(.{0});    // Mean along axis 0
```

## Method Chaining

Operations return expression types that support further operations:

```zig
const result = input
    .matmul(weights)      // Matrix multiply
    .add(bias)            // Add bias
    .relu()               // Activation
    .mul(scale)           // Scale
    .sum(.{1});           // Reduce
```

## Available Operations

### Activation Functions

| Method | Description |
|--------|-------------|
| `.relu()` | ReLU: max(0, x) |
| `.sigmoid()` | Sigmoid: 1/(1+exp(-x)) |
| `.tanh()` | Hyperbolic tangent |
| `.gelu()` | Gaussian Error Linear Unit |
| `.silu()` | Sigmoid Linear Unit (Swish) |
| `.softplus()` | Softplus: log(1+exp(x)) |
| `.leaky_relu()` | Leaky ReLU |

### Math Functions

| Method | Description |
|--------|-------------|
| `.exp()` | Exponential |
| `.log()` | Natural logarithm |
| `.sqrt()` | Square root |
| `.rsqrt()` | Reciprocal square root |
| `.sin()` | Sine |
| `.cos()` | Cosine |
| `.abs()` | Absolute value |
| `.neg()` | Negation |
| `.floor()` | Floor |
| `.ceil()` | Ceiling |
| `.round()` | Round to nearest |

### Arithmetic Operations

| Method | Description |
|--------|-------------|
| `.add(other)` | Element-wise addition |
| `.sub(other)` | Element-wise subtraction |
| `.mul(other)` | Element-wise multiplication |
| `.div(other)` | Element-wise division |
| `.pow(other)` | Element-wise power |
| `.max(other)` | Element-wise maximum |
| `.min(other)` | Element-wise minimum |

### Comparison Operations

| Method | Description |
|--------|-------------|
| `.eq(other)` | Equal |
| `.ne(other)` | Not equal |
| `.lt(other)` | Less than |
| `.le(other)` | Less than or equal |
| `.gt(other)` | Greater than |
| `.ge(other)` | Greater than or equal |

### Matrix Operations

| Method | Description |
|--------|-------------|
| `.matmul(other)` | Matrix multiplication |

### Reductions

| Method | Description |
|--------|-------------|
| `.sum(axes)` | Sum reduction |
| `.mean(axes)` | Mean reduction |
| `.prod(axes)` | Product reduction |
| `.reduce_max(axes)` | Maximum reduction |
| `.reduce_min(axes)` | Minimum reduction |
| `.variance(axes)` | Variance |

## Operation Properties

### Type Preservation

Operations preserve the element type:

```zig
const A = tz.Tensor(f32, .{ 3, 4 });
const expr = a.relu().add(b);

comptime {
    std.debug.assert(expr.ElementType == f32);
}
```

### Shape Rules

Each operation has specific shape requirements:

| Operation | Shape Rule |
|-----------|-----------|
| Unary | Output = Input |
| Binary | Output = Broadcast(A, B) |
| Matmul | `[M, K] @ [K, N] = [M, N]` |
| Reduce | Removes or keeps specified axes |

## Compile-Time Validation

Invalid operations produce compile errors:

```zig
// Type mismatch
const a = tz.Tensor(f32, .{4}){};
const b = tz.Tensor(f64, .{4}){};
// const c = a.add(b);  // Error: type mismatch f32 vs f64

// Shape mismatch
const x = tz.Tensor(f32, .{ 3, 4 }){};
const y = tz.Tensor(f32, .{ 5, 6 }){};
// const z = x.add(y);  // Error: shapes not compatible
```

## Next Steps

- [Unary Operations](./unary.md) - Detailed unary operation reference
- [Binary Operations](./binary.md) - Detailed binary operation reference
- [Matrix Operations](./matrix.md) - Matrix multiplication and transpose
- [Reductions](./reductions.md) - Aggregation operations
