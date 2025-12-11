# Broadcasting

Broadcasting allows operations between tensors of different shapes.

## Rules

NumPy-compatible broadcasting:

1. Align dimensions from the right
2. Dimensions must match OR one must be 1
3. Missing dimensions are treated as 1

## Visual Example

```
Shape A:     [3, 4]
Shape B:        [4]
              ↓ align right
Shape A: [3, 4]
Shape B: [1, 4]  (prepend 1)
              ↓ broadcast
Result:  [3, 4]
```

## Implementation

### Compile-Time Check

```zig
pub fn broadcastCompatible(comptime A: type, comptime B: type) bool {
    const a = A.dimensions;
    const b = B.dimensions;
    const max_ndim = @max(a.len, b.len);

    comptime {
        for (0..max_ndim) |i| {
            // Index from right
            const ai = if (i < a.len) a[a.len - 1 - i] else 1;
            const bi = if (i < b.len) b[b.len - 1 - i] else 1;

            if (ai != bi and ai != 1 and bi != 1) {
                return false;
            }
        }
        return true;
    }
}
```

### Result Shape

```zig
pub fn BroadcastShape(comptime A: type, comptime B: type) type {
    const a = A.dimensions;
    const b = B.dimensions;
    const result_ndim = @max(a.len, b.len);

    var result: [result_ndim]usize = undefined;

    comptime {
        for (0..result_ndim) |i| {
            const ai = if (i < a.len) a[a.len - 1 - i] else 1;
            const bi = if (i < b.len) b[b.len - 1 - i] else 1;

            if (ai != bi and ai != 1 and bi != 1) {
                @compileError("Shapes not broadcastable");
            }

            result[result_ndim - 1 - i] = @max(ai, bi);
        }
    }

    return Shape(result);
}
```

## Broadcasting Scenarios

### Scalar Broadcast

```zig
const A = Tensor(f32, .{ 3, 4 });  // [3, 4]
const B = Tensor(f32, .{});        // scalar []

// B broadcasts to [3, 4]
const result = a.add(b);  // Shape: [3, 4]
```

### Row/Column Broadcast

```zig
const Matrix = Tensor(f32, .{ 3, 4 });
const RowVec = Tensor(f32, .{4});      // [4] → [1, 4] → [3, 4]
const ColVec = Tensor(f32, .{ 3, 1 }); // [3, 1] → [3, 4]

// Row broadcast
const r1 = matrix.add(row_vec);  // [3, 4] + [4] → [3, 4]

// Column broadcast
const r2 = matrix.mul(col_vec);  // [3, 4] * [3, 1] → [3, 4]
```

### Batched Broadcast

```zig
const Batch = Tensor(f32, .{ 8, 3, 4 });  // [batch, rows, cols]
const Single = Tensor(f32, .{ 3, 4 });    // [rows, cols]

// Single broadcasts across batch dimension
const result = batch.add(single);  // [8, 3, 4]
```

## Index Calculation

At runtime, broadcast indexing maps output indices to input indices:

```zig
fn broadcastIndex(
    comptime in_shape: anytype,
    comptime out_shape: anytype,
    out_idx: usize,
) usize {
    const out_coords = unflattenIndex(out_shape, out_idx);
    var in_idx: usize = 0;
    var stride: usize = 1;

    comptime var i = in_shape.len;
    inline while (i > 0) {
        i -= 1;
        const out_dim_idx = out_shape.len - in_shape.len + i;
        const coord = if (in_shape[i] == 1) 0 else out_coords[out_dim_idx];
        in_idx += coord * stride;
        stride *= in_shape[i];
    }

    return in_idx;
}
```

## Stride-Based Broadcasting

For efficient iteration, compute broadcast strides:

```zig
fn broadcastStrides(
    comptime in_shape: anytype,
    comptime out_shape: anytype,
) [out_shape.len]usize {
    var strides: [out_shape.len]usize = [_]usize{0} ** out_shape.len;

    const offset = out_shape.len - in_shape.len;
    var stride: usize = 1;

    comptime var i = in_shape.len;
    inline while (i > 0) {
        i -= 1;
        if (in_shape[i] != 1) {
            strides[offset + i] = stride;
        }
        // stride stays 0 for broadcast dims
        stride *= in_shape[i];
    }

    return strides;
}
```

Usage:

```zig
const a_strides = broadcastStrides(A.shape, Result.shape);
const b_strides = broadcastStrides(B.shape, Result.shape);

// Iterate result
for (0..Result.numel()) |i| {
    const coords = unflattenIndex(Result.shape, i);
    const a_idx = dotProduct(coords, a_strides);
    const b_idx = dotProduct(coords, b_strides);
    result[i] = a.data[a_idx] + b.data[b_idx];
}
```

## Invalid Broadcasts

```zig
const A = Tensor(f32, .{ 3, 4 });
const B = Tensor(f32, .{ 2, 4 });

// Compile error: 3 != 2 and neither is 1
const result = a.add(b);  // Error: Shapes not broadcastable
```

## Performance Considerations

### Contiguous vs Broadcast

```zig
// Fast: both contiguous
const A = Tensor(f32, .{ 1000, 1000 });
const B = Tensor(f32, .{ 1000, 1000 });

// Slower: B broadcasts, non-contiguous access
const A = Tensor(f32, .{ 1000, 1000 });
const B = Tensor(f32, .{1000});  // Row broadcast
```

### Pre-expand for Hot Paths

```zig
// If broadcasting same tensor repeatedly:
const bias = Tensor(f32, .{hidden_size});

// Consider explicit expansion for critical paths
const expanded_bias = bias.expand(.{ batch_size, hidden_size });
```

## Next Steps

- [Shape Algebra](./shape-algebra.md) - Shape computation details
- [Custom Operations](./custom-ops.md) - Implementing new ops
