# Data Types

Tenzor supports various numeric types for tensor elements.

## Supported Types

### Floating Point

| Type | Size | Range | Use Case |
|------|------|-------|----------|
| `f16` | 2 bytes | ±65504 | Memory-constrained, inference |
| `f32` | 4 bytes | ±3.4e38 | General purpose (default) |
| `f64` | 8 bytes | ±1.8e308 | High precision |

```zig
const F16Tensor = tz.Tensor(f16, .{ 3, 4 });
const F32Tensor = tz.Tensor(f32, .{ 3, 4 });
const F64Tensor = tz.Tensor(f64, .{ 3, 4 });
```

### Integer Types

| Type | Size | Range |
|------|------|-------|
| `i8` | 1 byte | -128 to 127 |
| `i16` | 2 bytes | -32768 to 32767 |
| `i32` | 4 bytes | -2^31 to 2^31-1 |
| `i64` | 8 bytes | -2^63 to 2^63-1 |
| `u8` | 1 byte | 0 to 255 |
| `u16` | 2 bytes | 0 to 65535 |
| `u32` | 4 bytes | 0 to 2^32-1 |
| `u64` | 8 bytes | 0 to 2^64-1 |

```zig
const I32Tensor = tz.Tensor(i32, .{ 3, 4 });
const U8Tensor = tz.Tensor(u8, .{ 224, 224, 3 });  // Image data
```

## Type Properties

### Size and Alignment

```zig
const dtype = @import("tenzor").core.dtype;

const f32_info = dtype.DType(f32);
comptime {
    std.debug.assert(f32_info.size == 4);
    std.debug.assert(f32_info.alignment == 4);
}
```

### Type Categories

```zig
const dtype = @import("tenzor").core.dtype;

comptime {
    std.debug.assert(dtype.isFloat(f32));
    std.debug.assert(!dtype.isFloat(i32));
    std.debug.assert(dtype.isSignedInt(i32));
    std.debug.assert(!dtype.isSignedInt(u32));
    std.debug.assert(dtype.isUnsignedInt(u32));
}
```

## Type Compatibility

### Same-Type Operations

Most operations require matching element types:

```zig
const A = tz.Tensor(f32, .{ 3, 4 });
const B = tz.Tensor(f32, .{ 3, 4 });
const C = A.add(B);  // OK: both f32

const D = tz.Tensor(f64, .{ 3, 4 });
// const E = A.add(D);  // Error: f32 vs f64
```

### Type Inference

Expression types preserve the element type:

```zig
const A = tz.Tensor(f32, .{ 3, 4 });
var a = A{};

const expr = a.relu().exp().add(a);

comptime {
    std.debug.assert(expr.ElementType == f32);
}
```

## Choosing Data Types

### f32 (Recommended Default)

- Best balance of precision and performance
- Optimal SIMD width on most hardware
- Standard for ML inference

```zig
const Model = tz.Tensor(f32, .{ 1000, 1000 });
```

### f16 (Half Precision)

- 2x memory savings
- Faster on hardware with f16 support
- Reduced precision (3-4 decimal digits)

```zig
const Weights = tz.Tensor(f16, .{ 4096, 4096 });  // 32MB vs 64MB
```

### f64 (Double Precision)

- Scientific computing
- Accumulation in reductions
- Numerical stability for ill-conditioned problems

```zig
const Precision = tz.Tensor(f64, .{ 100, 100 });
```

## Memory Considerations

### Tensor Size Calculation

```zig
const T = tz.Tensor(f32, .{ 1024, 1024 });

comptime {
    const bytes = T.numel() * @sizeOf(T.ElementType);
    std.debug.assert(bytes == 4 * 1024 * 1024);  // 4 MB
}
```

### Type Size Comparison

| Shape | f16 | f32 | f64 |
|-------|-----|-----|-----|
| `{1024}` | 2 KB | 4 KB | 8 KB |
| `{1024, 1024}` | 2 MB | 4 MB | 8 MB |
| `{32, 3, 224, 224}` | 9.2 MB | 18.4 MB | 36.8 MB |

## SIMD Considerations

Different types have different vector widths:

| Type | 256-bit Vector | 512-bit Vector |
|------|----------------|----------------|
| `f32` | 8 elements | 16 elements |
| `f64` | 4 elements | 8 elements |
| `f16` | 16 elements | 32 elements |

```zig
const simd = @import("tenzor").backend.cpu.simd;

comptime {
    const f32_width = simd.suggestVectorLength(f32);  // 8 on AVX
    const f64_width = simd.suggestVectorLength(f64);  // 4 on AVX
}
```

## Type Casting (Future)

Currently, explicit type conversion requires manual handling:

```zig
// Future API (not yet implemented):
// const f32_tensor = f16_tensor.cast(f32);
```

## Next Steps

- [Memory Layout](./memory-layout.md) - Data organization in memory
- [SIMD Optimization](../backend/simd.md) - Type-specific vectorization
