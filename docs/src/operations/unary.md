# Unary Operations

Unary operations transform each element independently.

## Shape Behavior

Unary operations preserve the input shape:

```zig
const A = tz.Tensor(f32, .{ 3, 4 });
var a = A{};

const b = a.relu();

comptime {
    std.debug.assert(b.shape[0] == 3);
    std.debug.assert(b.shape[1] == 4);
}
```

## Available Operations

### Basic Math

#### neg() - Negation

```zig
const y = x.neg();  // y = -x
```

#### abs() - Absolute Value

```zig
const y = x.abs();  // y = |x|
```

### Exponential and Logarithmic

#### exp() - Exponential

```zig
const y = x.exp();  // y = e^x
```

#### log() - Natural Logarithm

```zig
const y = x.log();  // y = ln(x)
```

### Power Functions

#### sqrt() - Square Root

```zig
const y = x.sqrt();  // y = √x
```

#### rsqrt() - Reciprocal Square Root

```zig
const y = x.rsqrt();  // y = 1/√x
```

### Trigonometric

#### sin() - Sine

```zig
const y = x.sin();  // y = sin(x)
```

#### cos() - Cosine

```zig
const y = x.cos();  // y = cos(x)
```

#### tanh() - Hyperbolic Tangent

```zig
const y = x.tanh();  // y = tanh(x)
```

### Rounding

#### floor() - Floor

```zig
const y = x.floor();  // y = ⌊x⌋
```

#### ceil() - Ceiling

```zig
const y = x.ceil();  // y = ⌈x⌉
```

#### round() - Round to Nearest

```zig
const y = x.round();  // y = round(x)
```

## Activation Functions

See [Activation Functions](./activations.md) for detailed coverage.

### relu() - Rectified Linear Unit

```zig
const y = x.relu();  // y = max(0, x)
```

### sigmoid() - Sigmoid

```zig
const y = x.sigmoid();  // y = 1 / (1 + e^(-x))
```

### gelu() - Gaussian Error Linear Unit

```zig
const y = x.gelu();  // y ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
```

### silu() - Sigmoid Linear Unit (Swish)

```zig
const y = x.silu();  // y = x * sigmoid(x)
```

### softplus()

```zig
const y = x.softplus();  // y = log(1 + e^x)
```

## Chaining Unary Operations

```zig
const result = input
    .abs()      // |x|
    .log()      // log|x|
    .neg()      // -log|x|
    .exp();     // e^(-log|x|) = 1/|x|
```

## SIMD Implementation

Unary operations are SIMD-vectorized:

```zig
// Internal implementation (simplified)
pub fn unaryOp(comptime op: OpTag, input: []const f32, output: []f32) void {
    const vec_len = simd.suggestVectorLength(f32);
    var i: usize = 0;

    // SIMD loop
    while (i + vec_len <= input.len) : (i += vec_len) {
        const v = simd.load(f32, input[i..]);
        const result = applyUnary(op, v);
        simd.store(f32, result, output[i..]);
    }

    // Scalar remainder
    while (i < input.len) : (i += 1) {
        output[i] = applyScalar(op, input[i]);
    }
}
```

## Next Steps

- [Math Functions](./math-functions.md) - Detailed math function reference
- [Activation Functions](./activations.md) - Neural network activations
