# Arithmetic Operations

Detailed reference for element-wise arithmetic operations.

## add() - Addition

Element-wise addition with broadcasting.

```zig
const c = a.add(b);  // c[i] = a[i] + b[i]
```

**Properties:**
- Commutative: a + b = b + a
- Associative: (a + b) + c = a + (b + c)
- Identity: a + 0 = a

**Common Uses:**
- Bias addition
- Residual connections
- Accumulation

```zig
// Add bias to linear output
const biased = linear_output.add(bias);

// Residual connection
const output = residual.add(transformed);
```

## sub() - Subtraction

Element-wise subtraction with broadcasting.

```zig
const c = a.sub(b);  // c[i] = a[i] - b[i]
```

**Properties:**
- Not commutative: a - b ≠ b - a
- Identity: a - 0 = a
- Inverse: a - a = 0

**Common Uses:**
- Computing differences
- Centering data
- Gradient computation

```zig
// Center data around mean
const centered = data.sub(mean);

// Compute error
const error = predicted.sub(target);
```

## mul() - Multiplication

Element-wise multiplication with broadcasting.

```zig
const c = a.mul(b);  // c[i] = a[i] * b[i]
```

**Properties:**
- Commutative: a * b = b * a
- Associative: (a * b) * c = a * (b * c)
- Identity: a * 1 = a
- Zero: a * 0 = 0

**Common Uses:**
- Scaling
- Masking
- Hadamard product
- Attention weights

```zig
// Scale by learned parameter
const scaled = features.mul(scale);

// Apply mask (zeros out elements)
const masked = values.mul(mask);

// Gating mechanism
const gated = candidate.mul(gate);
```

## div() - Division

Element-wise division with broadcasting.

```zig
const c = a.div(b);  // c[i] = a[i] / b[i]
```

**Properties:**
- Not commutative: a / b ≠ b / a
- Identity: a / 1 = a
- Division by zero: undefined behavior

**Common Uses:**
- Normalization
- Averaging
- Rescaling

```zig
// Normalize by standard deviation
const normalized = centered.div(std);

// Compute average
const avg = sum.div(count);

// Softmax denominator
const probs = exp_x.div(sum_exp);
```

## pow() - Power

Element-wise power with broadcasting.

```zig
const c = a.pow(b);  // c[i] = a[i]^b[i]
```

**Properties:**
- Not commutative: a^b ≠ b^a
- Identity: a^1 = a
- Zero power: a^0 = 1 (for a ≠ 0)

**Common Uses:**
- Polynomial features
- Distance metrics
- Loss functions

```zig
// Square elements
const squared = x.pow(tz.scalar(f32, 2.0));

// Square root (equivalent to x^0.5)
const sqrt_x = x.pow(tz.scalar(f32, 0.5));

// L2 loss component
const l2_term = weights.pow(two).sum(.{});
```

## max() - Element-wise Maximum

Returns the larger element at each position.

```zig
const c = a.max(b);  // c[i] = max(a[i], b[i])
```

**Properties:**
- Commutative: max(a, b) = max(b, a)
- Associative: max(max(a, b), c) = max(a, max(b, c))
- Idempotent: max(a, a) = a

**Common Uses:**
- ReLU implementation
- Clipping upper bound
- Piecewise functions

```zig
// Manual ReLU
const relu = x.max(tz.scalar(f32, 0.0));

// Clamp to minimum value
const clamped = values.max(min_value);
```

## min() - Element-wise Minimum

Returns the smaller element at each position.

```zig
const c = a.min(b);  // c[i] = min(a[i], b[i])
```

**Properties:**
- Commutative: min(a, b) = min(b, a)
- Associative: min(min(a, b), c) = min(a, min(b, c))
- Idempotent: min(a, a) = a

**Common Uses:**
- Clipping upper bound
- Soft constraints
- Piecewise functions

```zig
// Clamp to maximum value
const clamped = values.min(max_value);

// Clip to range
const clipped = values.max(lower).min(upper);
```

## Combined Patterns

### Clamp/Clip

```zig
const clipped = x.max(lower_bound).min(upper_bound);
```

### Leaky ReLU

```zig
const positive = x.max(zero);
const negative = x.min(zero).mul(alpha);
const leaky = positive.add(negative);
```

### Huber Loss

```zig
const error = predicted.sub(target);
const abs_error = error.abs();
const quadratic = error.mul(error).mul(half);
const linear = delta.mul(abs_error.sub(half_delta));
const loss = quadratic.min(linear);
```

## Numerical Considerations

### Division Safety

```zig
// Add small epsilon to prevent division by zero
const safe_div = numerator.div(denominator.add(epsilon));
```

### Overflow Prevention

```zig
// Log-space computation for products
const log_product = log_a.add(log_b);  // Instead of a * b

// Subtract max for numerical stability
const shifted = x.sub(x.max());
const stable_softmax = shifted.exp();
```

## Next Steps

- [Comparisons](./comparisons.md) - Comparison operations
- [Broadcasting](../advanced/broadcasting.md) - How broadcasting works
