# Comparison Operations

Comparison operations produce boolean-like results for element-wise comparisons.

## Available Comparisons

### eq() - Equal

```zig
const mask = a.eq(b);  // mask[i] = (a[i] == b[i]) ? 1.0 : 0.0
```

### ne() - Not Equal

```zig
const mask = a.ne(b);  // mask[i] = (a[i] != b[i]) ? 1.0 : 0.0
```

### lt() - Less Than

```zig
const mask = a.lt(b);  // mask[i] = (a[i] < b[i]) ? 1.0 : 0.0
```

### le() - Less Than or Equal

```zig
const mask = a.le(b);  // mask[i] = (a[i] <= b[i]) ? 1.0 : 0.0
```

### gt() - Greater Than

```zig
const mask = a.gt(b);  // mask[i] = (a[i] > b[i]) ? 1.0 : 0.0
```

### ge() - Greater Than or Equal

```zig
const mask = a.ge(b);  // mask[i] = (a[i] >= b[i]) ? 1.0 : 0.0
```

## Result Type

Comparisons return float tensors with values 0.0 or 1.0:

```zig
const A = tz.Tensor(f32, .{4});
const B = tz.Tensor(f32, .{4});

var a = A{};
var b = B{};

const mask = a.gt(b);

comptime {
    std.debug.assert(mask.ElementType == f32);  // Same as input
}
```

## Common Patterns

### Masking

Use comparisons to create masks for selective operations:

```zig
// Zero out negative values (manual ReLU)
const positive_mask = x.gt(zero);
const result = x.mul(positive_mask);
```

### Conditional Selection

Implement where/select using arithmetic:

```zig
// where(condition, a, b) = condition * a + (1 - condition) * b
const one = tz.scalar(f32, 1.0);
const condition = x.gt(threshold);
const result = condition.mul(a).add(one.sub(condition).mul(b));
```

### Counting

Count elements satisfying a condition:

```zig
const mask = x.gt(threshold);
const count = mask.sum(.{});  // Sum of 1s = count
```

### Threshold Detection

```zig
// Find elements above threshold
const above = values.gt(threshold);

// Find elements in range
const in_lower = values.ge(lower);
const in_upper = values.le(upper);
const in_range = in_lower.mul(in_upper);
```

## Broadcasting

Comparisons support broadcasting like other binary operations:

```zig
const A = tz.Tensor(f32, .{ 3, 4 });
const B = tz.Tensor(f32, .{4});        // Broadcasts

const mask = a.gt(b);  // Result shape: { 3, 4 }
```

## Chaining with Arithmetic

Comparison masks integrate with arithmetic operations:

```zig
// Clamp negative values to zero
const positive_mask = x.ge(zero);
const clamped = x.mul(positive_mask);

// Apply different scales based on sign
const is_positive = x.gt(zero);
const is_negative = x.lt(zero);
const result = x.mul(is_positive).mul(pos_scale)
              .add(x.mul(is_negative).mul(neg_scale));
```

## Numerical Considerations

### Floating Point Equality

Direct equality comparison with floats can be unreliable:

```zig
// Risky: exact equality
const exact_match = a.eq(b);

// Safer: approximate equality
const diff = a.sub(b).abs();
const approx_match = diff.lt(epsilon);
```

### NaN Handling

Comparisons with NaN follow IEEE 754 rules:
- NaN != NaN
- NaN comparisons return false (except !=)

## Next Steps

- [Reductions](./reductions.md) - Aggregate comparison results
- [Arithmetic](./arithmetic.md) - Use masks in arithmetic
