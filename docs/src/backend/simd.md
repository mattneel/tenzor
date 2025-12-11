# SIMD Optimization

Tenzor uses SIMD (Single Instruction, Multiple Data) to process multiple elements simultaneously.

## Overview

SIMD allows processing multiple data elements with a single instruction:

```
Scalar:  a[0] + b[0], a[1] + b[1], a[2] + b[2], ...  (N instructions)
SIMD:    a[0:8] + b[0:8]                              (1 instruction for 8 elements)
```

## Vector Types

Tenzor uses Zig's `@Vector` type:

```zig
const simd = @import("tenzor").backend.cpu.simd;

// Get optimal vector length for type
const vec_len = simd.suggestVectorLength(f32);  // e.g., 8 on AVX

// Vector type
const Vec = simd.Vec(f32);  // @Vector(8, f32)
```

### Platform Vector Widths

| Platform | f32 width | f64 width |
|----------|-----------|-----------|
| SSE | 4 | 2 |
| AVX | 8 | 4 |
| AVX-512 | 16 | 8 |
| NEON | 4 | 2 |
| WASM SIMD | 4 | 2 |

## Core Operations

### Load and Store

```zig
// Load vector from memory
const vec = simd.load(f32, slice[i..]);

// Store vector to memory
simd.store(f32, result_vec, output[i..]);
```

### Splat

Create vector with repeated value:

```zig
const ones = simd.splat(f32, 1.0);  // [1.0, 1.0, 1.0, ...]
```

### Arithmetic

```zig
const sum = simd.add(f32, a, b);
const diff = simd.sub(f32, a, b);
const prod = simd.mul(f32, a, b);
const quot = simd.div(f32, a, b);
```

### Comparisons

```zig
const max_vec = simd.max(f32, a, b);
const min_vec = simd.min(f32, a, b);
```

### Math Functions

```zig
const exp_v = simd.exp(f32, x);
const log_v = simd.log(f32, x);
const sqrt_v = simd.sqrt(f32, x);
const sin_v = simd.sin(f32, x);
```

### Reductions

```zig
const sum = simd.reduceAdd(f32, vec);   // Sum all elements
const prod = simd.reduceMul(f32, vec);  // Product
const max = simd.reduceMax(f32, vec);   // Maximum
const min = simd.reduceMin(f32, vec);   // Minimum
```

## Kernel Pattern

Standard SIMD kernel structure:

```zig
pub fn unaryOp(comptime op: OpTag, input: []const f32, output: []f32) void {
    const vec_len = simd.suggestVectorLength(f32);
    var i: usize = 0;

    // SIMD loop - process vec_len elements at a time
    while (i + vec_len <= input.len) : (i += vec_len) {
        const v = simd.load(f32, input[i..]);
        const result = applyOp(op, v);
        simd.store(f32, result, output[i..]);
    }

    // Scalar remainder - handle leftover elements
    while (i < input.len) : (i += 1) {
        output[i] = applyScalar(op, input[i]);
    }
}
```

## Activation Implementations

### ReLU

```zig
pub fn relu(comptime T: type, a: Vec(T)) Vec(T) {
    const zero: Vec(T) = @splat(0.0);
    return @max(a, zero);
}
```

### Sigmoid

```zig
pub fn sigmoid(comptime T: type, a: Vec(T)) Vec(T) {
    const one: Vec(T) = @splat(1.0);
    return one / (one + @exp(-a));
}
```

### GELU

```zig
pub fn gelu(comptime T: type, a: Vec(T)) Vec(T) {
    const half: Vec(T) = @splat(0.5);
    const one: Vec(T) = @splat(1.0);
    const sqrt_2_over_pi: Vec(T) = @splat(0.7978845608);
    const coeff: Vec(T) = @splat(0.044715);

    const x3 = a * a * a;
    const inner = sqrt_2_over_pi * (a + coeff * x3);
    return half * a * (one + tanh(T, inner));
}
```

## Scalar Fallback

For operations without direct SIMD support:

```zig
pub const scalar = struct {
    pub fn sigmoid(x: anytype) @TypeOf(x) {
        return 1.0 / (1.0 + @exp(-x));
    }

    pub fn gelu(x: anytype) @TypeOf(x) {
        const T = @TypeOf(x);
        const sqrt_2_over_pi: T = 0.7978845608;
        const coeff: T = 0.044715;
        const x3 = x * x * x;
        const inner = sqrt_2_over_pi * (x + coeff * x3);
        return 0.5 * x * (1.0 + scalar.tanh(inner));
    }
};
```

## Performance Tips

### Alignment

Ensure data is aligned for optimal loads:

```zig
// Aligned allocation
const data = try allocator.alignedAlloc(f32, 32, count);
```

### Loop Unrolling

Process multiple vectors per iteration:

```zig
while (i + 4 * vec_len <= n) : (i += 4 * vec_len) {
    // Process 4 vectors
    const v0 = simd.load(f32, input[i..]);
    const v1 = simd.load(f32, input[i + vec_len..]);
    const v2 = simd.load(f32, input[i + 2*vec_len..]);
    const v3 = simd.load(f32, input[i + 3*vec_len..]);
    // ...
}
```

### Prefetching

Zig handles prefetching automatically in most cases.

## Next Steps

- [Vector Types](./vector-types.md) - Detailed vector type reference
- [Vectorized Kernels](./vectorized-kernels.md) - Kernel implementations
