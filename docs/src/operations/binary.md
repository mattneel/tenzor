# Binary Operations

Binary operations combine two tensors element-wise with automatic broadcasting.

## Broadcasting Rules

Binary operations follow NumPy-style broadcasting:

1. Shapes are compared right-to-left
2. Dimensions match if equal or one is 1
3. Missing dimensions are treated as 1

```zig
const A = tz.Tensor(f32, .{ 3, 4 });    // [3, 4]
const B = tz.Tensor(f32, .{4});          // [4]

const C = a.add(b);  // Result: [3, 4]
```

### Broadcasting Examples

| Shape A | Shape B | Result | Valid |
|---------|---------|--------|-------|
| `{3, 4}` | `{4}` | `{3, 4}` | ✓ |
| `{3, 4}` | `{3, 1}` | `{3, 4}` | ✓ |
| `{3, 4}` | `{1, 4}` | `{3, 4}` | ✓ |
| `{2, 3, 4}` | `{3, 4}` | `{2, 3, 4}` | ✓ |
| `{3, 4}` | `{3, 4}` | `{3, 4}` | ✓ |
| `{3, 4}` | `{5}` | - | ✗ |
| `{3, 4}` | `{2, 4}` | - | ✗ |

## Arithmetic Operations

### add() - Addition

```zig
const c = a.add(b);  // c = a + b
```

### sub() - Subtraction

```zig
const c = a.sub(b);  // c = a - b
```

### mul() - Multiplication

```zig
const c = a.mul(b);  // c = a * b
```

### div() - Division

```zig
const c = a.div(b);  // c = a / b
```

### pow() - Power

```zig
const c = a.pow(b);  // c = a^b
```

## Comparison Operations

See [Comparisons](./comparisons.md) for details.

### max() - Element-wise Maximum

```zig
const c = a.max(b);  // c = max(a, b)
```

### min() - Element-wise Minimum

```zig
const c = a.min(b);  // c = min(a, b)
```

## Scalar Operations

Binary operations support scalar broadcasting:

```zig
const A = tz.Tensor(f32, .{ 3, 4 });
var a = A{};

// Scalar wrapped in expression
const scaled = a.mul(tz.scalar(f32, 2.0));  // Multiply by 2
const shifted = a.add(tz.scalar(f32, 1.0)); // Add 1
```

## Type Requirements

Both operands must have the same element type:

```zig
const A = tz.Tensor(f32, .{ 3, 4 });
const B = tz.Tensor(f32, .{4});
const C = a.add(b);  // OK: both f32

const D = tz.Tensor(f64, .{4});
// const E = a.add(d);  // Error: f32 vs f64
```

## Result Shape

The result shape is the broadcast of input shapes:

```zig
const A = tz.Tensor(f32, .{ 1, 3, 1 });
const B = tz.Tensor(f32, .{ 2, 1, 4 });
const C = a.add(b);

comptime {
    std.debug.assert(C.shape[0] == 2);
    std.debug.assert(C.shape[1] == 3);
    std.debug.assert(C.shape[2] == 4);
}
```

## Common Patterns

### Residual Connection

```zig
const residual = input.add(transformed);
```

### Normalization

```zig
const centered = x.sub(mean);
const normalized = centered.div(std);
```

### Scaling

```zig
const scaled = x.mul(scale).add(shift);
```

### Gradient Clipping

```zig
const clipped = gradient.max(neg_limit).min(pos_limit);
```

## SIMD Implementation

Binary operations are SIMD-vectorized:

```zig
// Internal implementation (simplified)
pub fn binaryOp(comptime op: OpTag, a: []const f32, b: []const f32, out: []f32) void {
    const vec_len = simd.suggestVectorLength(f32);
    var i: usize = 0;

    // SIMD loop
    while (i + vec_len <= a.len) : (i += vec_len) {
        const va = simd.load(f32, a[i..]);
        const vb = simd.load(f32, b[i..]);
        const result = applyBinary(op, va, vb);
        simd.store(f32, result, out[i..]);
    }

    // Scalar remainder
    while (i < a.len) : (i += 1) {
        out[i] = applyScalar(op, a[i], b[i]);
    }
}
```

## Next Steps

- [Arithmetic](./arithmetic.md) - Detailed arithmetic reference
- [Comparisons](./comparisons.md) - Comparison operations
- [Broadcasting](../advanced/broadcasting.md) - Advanced broadcasting
