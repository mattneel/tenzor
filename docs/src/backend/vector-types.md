# Vector Types

Zig's `@Vector` type provides portable SIMD abstractions.

## Vector Type Basics

```zig
// Declare a vector type
const Vec4f32 = @Vector(4, f32);

// Create vectors
const a: Vec4f32 = .{ 1.0, 2.0, 3.0, 4.0 };
const b: Vec4f32 = @splat(2.0);  // { 2.0, 2.0, 2.0, 2.0 }

// Arithmetic
const sum = a + b;   // { 3.0, 4.0, 5.0, 6.0 }
const prod = a * b;  // { 2.0, 4.0, 6.0, 8.0 }
```

## Tenzor's Vector Abstraction

```zig
const simd = @import("tenzor").backend.cpu.simd;

// Automatic width selection
pub fn Vec(comptime T: type) type {
    return @Vector(suggestVectorLength(T), T);
}

// Usage
const F32Vec = simd.Vec(f32);  // Platform-optimal width
```

## Vector Operations

### Element-wise Arithmetic

```zig
const a: Vec(f32) = // ...
const b: Vec(f32) = // ...

const add = a + b;
const sub = a - b;
const mul = a * b;
const div = a / b;
const neg = -a;
```

### Comparisons

```zig
const eq = a == b;   // Vector of bool
const lt = a < b;
const gt = a > b;
```

### Built-in Functions

```zig
const abs_a = @abs(a);
const sqrt_a = @sqrt(a);
const min_ab = @min(a, b);
const max_ab = @max(a, b);
const floor_a = @floor(a);
const ceil_a = @ceil(a);
```

### Math Functions

```zig
const exp_a = @exp(a);
const log_a = @log(a);
const sin_a = @sin(a);
const cos_a = @cos(a);
```

### Reductions

```zig
const sum = @reduce(.Add, vec);
const prod = @reduce(.Mul, vec);
const max = @reduce(.Max, vec);
const min = @reduce(.Min, vec);
const and_all = @reduce(.And, bool_vec);
const or_any = @reduce(.Or, bool_vec);
```

## Loading and Storing

### From Slice

```zig
const vec_len = simd.suggestVectorLength(f32);

// Load
const slice: []const f32 = data[i..];
const vec: @Vector(vec_len, f32) = slice[0..vec_len].*;

// Store
var output: []f32 = out[i..];
output[0..vec_len].* = vec;
```

### With Wrapper Functions

```zig
const vec = simd.load(f32, data[i..]);
simd.store(f32, vec, output[i..]);
```

## Masking and Selection

### Select

```zig
const mask: @Vector(4, bool) = .{ true, false, true, false };
const a: @Vector(4, f32) = .{ 1, 2, 3, 4 };
const b: @Vector(4, f32) = .{ 5, 6, 7, 8 };

const result = @select(f32, mask, a, b);  // { 1, 6, 3, 8 }
```

### Application: Leaky ReLU

```zig
pub fn leakyRelu(comptime T: type, a: Vec(T), alpha: T) Vec(T) {
    const zero: Vec(T) = @splat(0.0);
    const alpha_vec: Vec(T) = @splat(alpha);
    const mask = a > zero;
    return @select(T, mask, a, a * alpha_vec);
}
```

## Type Conversions

### Integer to Float

```zig
const ints: @Vector(4, i32) = .{ 1, 2, 3, 4 };
const floats: @Vector(4, f32) = @as(@Vector(4, f32), @floatFromInt(ints));
```

### Float to Integer

```zig
const floats: @Vector(4, f32) = .{ 1.5, 2.7, 3.2, 4.9 };
const ints: @Vector(4, i32) = @intFromFloat(floats);  // { 1, 2, 3, 4 }
```

## Vector Width Considerations

### Automatic Selection

```zig
pub fn suggestVectorLength(comptime T: type) comptime_int {
    return std.simd.suggestVectorLength(T) orelse defaultVectorLength(T);
}

fn defaultVectorLength(comptime T: type) comptime_int {
    const target_bytes = 32;  // 256 bits (AVX)
    return @max(1, target_bytes / @sizeOf(T));
}
```

### Result by Type

| Type | Typical Width | Elements |
|------|--------------|----------|
| `f32` | 256-bit | 8 |
| `f64` | 256-bit | 4 |
| `i32` | 256-bit | 8 |
| `i8` | 256-bit | 32 |

## Performance Notes

### Alignment

Aligned loads/stores are faster:

```zig
// Ensure alignment for vector width
const alignment = @alignOf(Vec(f32));
const data = try allocator.alignedAlloc(f32, alignment, count);
```

### Avoid Scalar Gather/Scatter

Sequential access is much faster:

```zig
// Good: sequential
for (0..n / vec_len) |i| {
    const v = load(data[i * vec_len..]);
}

// Bad: strided (causes gather)
for (0..n) |i| {
    const v = data[i * stride];
}
```

## Next Steps

- [Vectorized Kernels](./vectorized-kernels.md) - Using vectors in kernels
- [SIMD Optimization](./simd.md) - Optimization strategies
