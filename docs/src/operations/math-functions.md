# Math Functions

Tenzor provides SIMD-optimized implementations of common mathematical functions.

## Exponential Family

### exp() - Exponential

Computes e^x for each element.

```zig
const y = x.exp();
```

**Domain:** All real numbers
**Range:** (0, ∞)

### log() - Natural Logarithm

Computes ln(x) for each element.

```zig
const y = x.log();
```

**Domain:** x > 0
**Range:** All real numbers

### exp2() - Base-2 Exponential

Computes 2^x for each element.

```zig
const y = x.exp2();
```

### log2() - Base-2 Logarithm

Computes log₂(x) for each element.

```zig
const y = x.log2();
```

### log10() - Base-10 Logarithm

Computes log₁₀(x) for each element.

```zig
const y = x.log10();
```

## Power Functions

### sqrt() - Square Root

Computes √x for each element.

```zig
const y = x.sqrt();
```

**Domain:** x ≥ 0
**Range:** [0, ∞)

### rsqrt() - Reciprocal Square Root

Computes 1/√x for each element.

```zig
const y = x.rsqrt();
```

**Domain:** x > 0
**Range:** (0, ∞)

This is often faster than `x.sqrt().reciprocal()` due to hardware support.

### pow() - Power (Binary)

Computes x^y element-wise.

```zig
const z = x.pow(y);
```

## Trigonometric Functions

### sin() - Sine

Computes sin(x) in radians.

```zig
const y = x.sin();
```

### cos() - Cosine

Computes cos(x) in radians.

```zig
const y = x.cos();
```

### tan() - Tangent

Computes tan(x) in radians.

```zig
const y = x.tan();
```

## Hyperbolic Functions

### sinh() - Hyperbolic Sine

Computes sinh(x) = (e^x - e^(-x)) / 2.

```zig
const y = x.sinh();
```

### cosh() - Hyperbolic Cosine

Computes cosh(x) = (e^x + e^(-x)) / 2.

```zig
const y = x.cosh();
```

### tanh() - Hyperbolic Tangent

Computes tanh(x) = sinh(x) / cosh(x).

```zig
const y = x.tanh();
```

**Range:** (-1, 1)

Commonly used as an activation function in neural networks.

## Inverse Trigonometric

### asin() - Arcsine

Computes arcsin(x) in radians.

```zig
const y = x.asin();
```

**Domain:** [-1, 1]
**Range:** [-π/2, π/2]

### acos() - Arccosine

Computes arccos(x) in radians.

```zig
const y = x.acos();
```

**Domain:** [-1, 1]
**Range:** [0, π]

### atan() - Arctangent

Computes arctan(x) in radians.

```zig
const y = x.atan();
```

**Range:** (-π/2, π/2)

### atan2() - Two-Argument Arctangent (Binary)

Computes atan2(y, x) in radians.

```zig
const z = y.atan2(x);
```

**Range:** (-π, π]

## Rounding Functions

### floor()

Rounds toward negative infinity.

```zig
const y = x.floor();
// floor(2.7) = 2.0
// floor(-2.3) = -3.0
```

### ceil()

Rounds toward positive infinity.

```zig
const y = x.ceil();
// ceil(2.3) = 3.0
// ceil(-2.7) = -2.0
```

### round()

Rounds to nearest integer (ties to even).

```zig
const y = x.round();
// round(2.5) = 2.0 (ties to even)
// round(3.5) = 4.0 (ties to even)
```

## Sign and Absolute Value

### abs() - Absolute Value

Computes |x|.

```zig
const y = x.abs();
```

### neg() - Negation

Computes -x.

```zig
const y = x.neg();
```

### sign() - Sign Function

Returns the sign of each element.

```zig
const y = x.sign();
// sign(x) = -1 if x < 0
//           0 if x = 0
//           1 if x > 0
```

## Common Patterns

### Softmax Numerator

```zig
// exp(x - max(x)) for numerical stability
const max_x = x.reduce_max(.{});
const shifted = x.sub(max_x);
const exp_x = shifted.exp();
```

### Log-Sum-Exp

```zig
const max_x = x.reduce_max(.{});
const shifted = x.sub(max_x);
const sum_exp = shifted.exp().sum(.{});
const lse = max_x.add(sum_exp.log());
```

### Normalized Euclidean Distance

```zig
const diff = a.sub(b);
const squared = diff.mul(diff);
const sum_sq = squared.sum(.{});
const dist = sum_sq.sqrt();
```

## Next Steps

- [Activation Functions](./activations.md) - Neural network activations
- [SIMD Optimization](../backend/simd.md) - How functions are vectorized
