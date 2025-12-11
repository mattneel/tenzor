# Memory Layout

Understanding how tensor data is organized in memory is crucial for performance.

## Row-Major Order

Tenzor uses **row-major** (C-style) ordering, where the last dimension varies fastest in memory:

```
2D Tensor [3, 4]:

Logical:                    Memory:
┌──────────────────────┐
│ [0,0] [0,1] [0,2] [0,3] │   [0,0][0,1][0,2][0,3][1,0][1,1][1,2][1,3][2,0][2,1][2,2][2,3]
│ [1,0] [1,1] [1,2] [1,3] │   ◄─── Row 0 ───►◄─── Row 1 ───►◄─── Row 2 ───►
│ [2,0] [2,1] [2,2] [2,3] │
└──────────────────────┘
```

## Strides

Strides describe the memory step (in elements) between consecutive indices along each dimension:

```zig
const T = tz.Tensor(f32, .{ 3, 4 });

// T.strides == .{ 4, 1 }
// - Stride 4 along dim 0: moving to next row skips 4 elements
// - Stride 1 along dim 1: moving to next column skips 1 element
```

### Stride Calculation

For a contiguous tensor with shape `[d0, d1, ..., dn]`:

```
stride[i] = d[i+1] * d[i+2] * ... * d[n]
stride[n] = 1  (last dimension)
```

```zig
const T = tz.Tensor(f32, .{ 2, 3, 4 });

// strides = .{ 12, 4, 1 }
// stride[0] = 3 * 4 = 12
// stride[1] = 4
// stride[2] = 1
```

## Linear Indexing

Convert multi-dimensional indices to linear offset:

```zig
fn linearIndex(indices: anytype, strides: anytype) usize {
    var offset: usize = 0;
    inline for (indices, strides) |idx, stride| {
        offset += idx * stride;
    }
    return offset;
}

// For tensor [2, 3, 4]:
// Element [1, 2, 0] is at offset: 1*12 + 2*4 + 0*1 = 20
```

### Manual Access

```zig
const Mat = tz.Tensor(f32, .{ 3, 4 });
var mat = Mat{};

// Access [row, col]
const row = 1;
const col = 2;
const offset = row * Mat.strides[0] + col * Mat.strides[1];
mat.data[offset] = 5.0;  // Set [1, 2] = 5.0
```

## Contiguous Memory

A tensor is **contiguous** when its strides follow the standard row-major pattern:

```zig
// Contiguous: strides match product of trailing dimensions
const T = tz.Tensor(f32, .{ 2, 3, 4 });
// strides = .{ 12, 4, 1 }  ✓ Contiguous

// After transpose (conceptual), strides would be different:
// strides = .{ 1, 4, 12 }  ✗ Not contiguous
```

### Benefits of Contiguity

1. **Sequential access** - Cache-friendly
2. **SIMD vectorization** - Load/store aligned chunks
3. **Memory efficiency** - No gaps in storage

## Cache Considerations

### Spatial Locality

Access patterns that follow memory order are fastest:

```zig
const Mat = tz.Tensor(f32, .{ 1000, 1000 });
var mat = Mat{};

// Good: row-wise access (follows memory order)
for (0..1000) |row| {
    for (0..1000) |col| {
        const idx = row * 1000 + col;
        mat.data[idx] *= 2.0;
    }
}

// Bad: column-wise access (cache misses)
for (0..1000) |col| {
    for (0..1000) |row| {
        const idx = row * 1000 + col;
        mat.data[idx] *= 2.0;  // Strided access
    }
}
```

### Cache Line Size

Typical cache lines are 64 bytes:

| Type | Elements per Cache Line |
|------|------------------------|
| `f32` | 16 |
| `f64` | 8 |
| `f16` | 32 |

## Alignment

Tensor data is aligned for SIMD:

```zig
const T = tz.Tensor(f32, .{16});

// data is aligned to at least @alignOf(f32) = 4
// For SIMD, we prefer 32-byte (AVX) or 64-byte (AVX-512) alignment
```

### SIMD Alignment

```zig
const simd = @import("tenzor").backend.cpu.simd;

// Load requires proper alignment
const vec_len = simd.suggestVectorLength(f32);  // e.g., 8
const vec = simd.load(f32, data[0..]);  // Assumes aligned

// Remainder handling for misaligned tails
var i: usize = 0;
while (i + vec_len <= data.len) : (i += vec_len) {
    // SIMD processing
}
while (i < data.len) : (i += 1) {
    // Scalar remainder
}
```

## Memory Layout Visualization

### 1D Tensor

```
[4] = { a, b, c, d }

Memory: [ a ][ b ][ c ][ d ]
Index:    0    1    2    3
```

### 2D Tensor

```
[2, 3] = {{ a, b, c },
          { d, e, f }}

Memory: [ a ][ b ][ c ][ d ][ e ][ f ]
Index:    0    1    2    3    4    5
```

### 3D Tensor

```
[2, 2, 3] = {{{ a, b, c },
              { d, e, f }},
             {{ g, h, i },
              { j, k, l }}}

Memory: [ a ][ b ][ c ][ d ][ e ][ f ][ g ][ h ][ i ][ j ][ k ][ l ]
Index:    0    1    2    3    4    5    6    7    8    9   10   11
```

## Broadcasting and Memory

When broadcasting, dimensions with size 1 have stride 0:

```zig
// Conceptual: Broadcasting [3, 1] to [3, 4]
// Original strides: { 1, 1 }
// Broadcast strides: { 1, 0 }  // stride 0 repeats the value
```

This is handled automatically during expression evaluation.

## Next Steps

- [Expression Graphs](./expression-graphs.md) - How operations are composed
- [SIMD Optimization](../backend/simd.md) - Memory-efficient vectorization
