# Shape Algebra

Tenzor performs shape computations at compile time.

## Shape Type

```zig
pub fn Shape(comptime dims: anytype) type {
    return struct {
        pub const ndim = dims.len;
        pub const dimensions = dims;

        pub fn numel() comptime_int {
            var n: comptime_int = 1;
            for (dims) |d| n *= d;
            return n;
        }
    };
}
```

## Operations

### Broadcasting

Compute broadcast result shape:

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
                @compileError("Not broadcastable");
            }

            result[result_ndim - 1 - i] = @max(ai, bi);
        }
    }

    return Shape(result);
}
```

### Matmul Shape

```zig
pub fn MatmulShape(comptime A: type, comptime B: type) type {
    comptime {
        const a = A.dimensions;
        const b = B.dimensions;

        // Inner dimensions must match
        if (a[a.len - 1] != b[b.len - 2]) {
            @compileError("Matmul inner dimension mismatch");
        }

        // Result: batch... × M × N
        const M = a[a.len - 2];
        const N = b[b.len - 1];

        // Handle batching
        if (a.len == 2 and b.len == 2) {
            return Shape(.{ M, N });
        }

        // Batched matmul
        const batch_dims = @max(a.len, b.len) - 2;
        var result: [batch_dims + 2]usize = undefined;

        // Broadcast batch dimensions
        for (0..batch_dims) |i| {
            const ai = if (i < a.len - 2) a[i] else 1;
            const bi = if (i < b.len - 2) b[i] else 1;
            result[i] = @max(ai, bi);
        }

        result[batch_dims] = M;
        result[batch_dims + 1] = N;

        return Shape(result);
    }
}
```

### Reduction Shape

```zig
pub fn ReduceShape(
    comptime Input: type,
    comptime axes: anytype,
    comptime keepdims: bool,
) type {
    const in_shape = Input.dimensions;

    comptime {
        if (axes.len == 0) {
            // Full reduction
            if (keepdims) {
                var ones: [in_shape.len]usize = undefined;
                for (&ones) |*d| d.* = 1;
                return Shape(ones);
            }
            return Shape(.{});
        }

        // Partial reduction
        var result: [in_shape.len - axes.len]usize = undefined;
        var j: usize = 0;

        for (0..in_shape.len) |i| {
            var is_reduced = false;
            for (axes) |ax| {
                if (ax == i) is_reduced = true;
            }
            if (!is_reduced) {
                result[j] = in_shape[i];
                j += 1;
            }
        }

        return Shape(result[0..j]);
    }
}
```

## Validation Functions

### Broadcast Compatibility

```zig
pub fn broadcastCompatible(comptime A: type, comptime B: type) bool {
    const a = A.dimensions;
    const b = B.dimensions;
    const max_ndim = @max(a.len, b.len);

    comptime {
        for (0..max_ndim) |i| {
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

### Same Shape

```zig
pub fn sameShape(comptime A: type, comptime B: type) bool {
    if (A.ndim != B.ndim) return false;

    comptime {
        for (0..A.ndim) |i| {
            if (A.dimensions[i] != B.dimensions[i]) {
                return false;
            }
        }
        return true;
    }
}
```

## Shape Utilities

### Flatten

```zig
pub fn FlattenShape(comptime Input: type) type {
    return Shape(.{Input.numel()});
}
```

### Reshape Validation

```zig
pub fn reshapeValid(comptime From: type, comptime To: type) bool {
    return From.numel() == To.numel();
}
```

### Transpose Shape

```zig
pub fn TransposeShape(comptime Input: type, comptime perm: anytype) type {
    const in_shape = Input.dimensions;
    var out_shape: [in_shape.len]usize = undefined;

    comptime {
        for (0..perm.len) |i| {
            out_shape[i] = in_shape[perm[i]];
        }
    }

    return Shape(out_shape);
}
```

## Examples

```zig
// Broadcasting
const A = Shape(.{ 3, 4 });
const B = Shape(.{4});
const C = BroadcastShape(A, B);  // Shape(.{ 3, 4 })

// Matmul
const X = Shape(.{ 2, 3, 4 });
const Y = Shape(.{ 2, 4, 5 });
const Z = MatmulShape(X, Y);  // Shape(.{ 2, 3, 5 })

// Reduction
const T = Shape(.{ 2, 3, 4 });
const R = ReduceShape(T, .{1}, false);  // Shape(.{ 2, 4 })
```

## Next Steps

- [Broadcasting](./broadcasting.md) - Broadcasting details
- [Comptime Magic](./comptime.md) - More comptime techniques
