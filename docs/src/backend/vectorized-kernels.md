# Vectorized Kernels

Tenzor's kernels are SIMD-optimized implementations of tensor operations.

## Kernel Structure

### Standard Pattern

```zig
pub fn kernel(input: []const f32, output: []f32) void {
    const vec_len = simd.suggestVectorLength(f32);
    var i: usize = 0;

    // Main SIMD loop
    while (i + vec_len <= input.len) : (i += vec_len) {
        const v = simd.load(f32, input[i..]);
        const result = process(v);
        simd.store(f32, result, output[i..]);
    }

    // Scalar remainder
    while (i < input.len) : (i += 1) {
        output[i] = processScalar(input[i]);
    }
}
```

## Elementwise Kernels

### Unary Operations

```zig
pub fn unaryOp(
    comptime op: OpTag,
    comptime T: type,
    input: []const T,
    output: []T,
) void {
    std.debug.assert(input.len == output.len);

    const vec_len = simd.suggestVectorLength(T);
    var i: usize = 0;

    while (i + vec_len <= input.len) : (i += vec_len) {
        const v = simd.load(T, input[i..]);
        const result = applyUnaryVec(op, T, v);
        simd.store(T, result, output[i..]);
    }

    while (i < input.len) : (i += 1) {
        output[i] = applyUnaryScalar(op, T, input[i]);
    }
}
```

### Binary Operations

```zig
pub fn binaryOp(
    comptime op: OpTag,
    comptime T: type,
    lhs: []const T,
    rhs: []const T,
    output: []T,
) void {
    std.debug.assert(lhs.len == rhs.len);
    std.debug.assert(lhs.len == output.len);

    const vec_len = simd.suggestVectorLength(T);
    var i: usize = 0;

    while (i + vec_len <= lhs.len) : (i += vec_len) {
        const v_lhs = simd.load(T, lhs[i..]);
        const v_rhs = simd.load(T, rhs[i..]);
        const result = applyBinaryVec(op, T, v_lhs, v_rhs);
        simd.store(T, result, output[i..]);
    }

    while (i < lhs.len) : (i += 1) {
        output[i] = applyBinaryScalar(op, T, lhs[i], rhs[i]);
    }
}
```

### Scalar Broadcast

```zig
pub fn binaryOpScalarRhs(
    comptime op: OpTag,
    comptime T: type,
    lhs: []const T,
    rhs_scalar: T,
    output: []T,
) void {
    const vec_len = simd.suggestVectorLength(T);
    const rhs_vec = simd.splat(T, rhs_scalar);

    var i: usize = 0;
    while (i + vec_len <= lhs.len) : (i += vec_len) {
        const v_lhs = simd.load(T, lhs[i..]);
        const result = applyBinaryVec(op, T, v_lhs, rhs_vec);
        simd.store(T, result, output[i..]);
    }

    while (i < lhs.len) : (i += 1) {
        output[i] = applyBinaryScalar(op, T, lhs[i], rhs_scalar);
    }
}
```

## Matmul Kernel

### Tiled Implementation

```zig
pub fn matmul(
    comptime T: type,
    comptime M: usize,
    comptime K: usize,
    comptime N: usize,
    a: *const [M * K]T,
    b: *const [K * N]T,
    c: *[M * N]T,
) void {
    const TILE = 32;  // Tile size for cache blocking

    // Initialize output
    @memset(c, 0);

    // Tiled multiplication
    var i: usize = 0;
    while (i < M) : (i += TILE) {
        var j: usize = 0;
        while (j < N) : (j += TILE) {
            var k: usize = 0;
            while (k < K) : (k += TILE) {
                matmulTile(T, M, K, N, a, b, c, i, j, k, TILE);
            }
        }
    }
}

fn matmulTile(...) void {
    // Compute C[i:i+tile, j:j+tile] += A[i:i+tile, k:k+tile] @ B[k:k+tile, j:j+tile]
    // With SIMD vectorization on inner loop
}
```

### SIMD Inner Loop

```zig
// For each row of tile
for (i_start..i_end) |i| {
    // For each column (vectorized)
    var j: usize = j_start;
    while (j + vec_len <= j_end) : (j += vec_len) {
        var acc = simd.load(T, c[i * N + j..]);

        for (k_start..k_end) |k| {
            const a_val: Vec(T) = @splat(a[i * K + k]);
            const b_vec = simd.load(T, b[k * N + j..]);
            acc = acc + a_val * b_vec;
        }

        simd.store(T, acc, c[i * N + j..]);
    }
}
```

## Reduction Kernels

### Sum Reduction

```zig
pub fn sum(comptime T: type, input: []const T) T {
    const vec_len = simd.suggestVectorLength(T);

    // Vector accumulator
    var acc: simd.Vec(T) = @splat(0);
    var i: usize = 0;

    while (i + vec_len <= input.len) : (i += vec_len) {
        const v = simd.load(T, input[i..]);
        acc = acc + v;
    }

    // Reduce vector to scalar
    var result = simd.reduceAdd(T, acc);

    // Add remainder
    while (i < input.len) : (i += 1) {
        result += input[i];
    }

    return result;
}
```

### Max Reduction

```zig
pub fn reduceMax(comptime T: type, input: []const T) T {
    if (input.len == 0) return -std.math.inf(T);

    const vec_len = simd.suggestVectorLength(T);
    var max_vec: simd.Vec(T) = @splat(-std.math.inf(T));
    var i: usize = 0;

    while (i + vec_len <= input.len) : (i += vec_len) {
        const v = simd.load(T, input[i..]);
        max_vec = @max(max_vec, v);
    }

    var result = simd.reduceMax(T, max_vec);

    while (i < input.len) : (i += 1) {
        result = @max(result, input[i]);
    }

    return result;
}
```

## Fused Kernels

### Elementwise Chain

```zig
pub fn fusedElementwiseChain(
    comptime ops: []const OpTag,
    input: []const f32,
    output: []f32,
) void {
    const vec_len = simd.suggestVectorLength(f32);
    var i: usize = 0;

    while (i + vec_len <= input.len) : (i += vec_len) {
        var v = simd.load(f32, input[i..]);

        // Apply all operations in sequence
        inline for (ops) |op| {
            v = applyUnaryVec(op, f32, v);
        }

        simd.store(f32, v, output[i..]);
    }

    // Scalar remainder
    while (i < input.len) : (i += 1) {
        var x = input[i];
        inline for (ops) |op| {
            x = applyUnaryScalar(op, f32, x);
        }
        output[i] = x;
    }
}
```

### Matmul with Epilogue

```zig
pub fn matmulWithEpilogue(
    // ... matmul params ...
    bias: ?[]const f32,
    comptime activation: ?OpTag,
) void {
    // Standard matmul...

    // Then apply epilogue in same loop
    if (bias) |b| {
        // Add bias
        result = result + simd.load(f32, b[j..]);
    }

    if (activation) |act| {
        result = applyUnaryVec(act, f32, result);
    }

    simd.store(f32, result, c[i * N + j..]);
}
```

## Performance Optimization

### Loop Unrolling

```zig
// Process 4 vectors per iteration
while (i + 4 * vec_len <= n) : (i += 4 * vec_len) {
    const v0 = simd.load(f32, input[i + 0*vec_len..]);
    const v1 = simd.load(f32, input[i + 1*vec_len..]);
    const v2 = simd.load(f32, input[i + 2*vec_len..]);
    const v3 = simd.load(f32, input[i + 3*vec_len..]);

    // Process all 4
    const r0 = process(v0);
    const r1 = process(v1);
    const r2 = process(v2);
    const r3 = process(v3);

    // Store all 4
    simd.store(f32, r0, output[i + 0*vec_len..]);
    simd.store(f32, r1, output[i + 1*vec_len..]);
    simd.store(f32, r2, output[i + 2*vec_len..]);
    simd.store(f32, r3, output[i + 3*vec_len..]);
}
```

## Next Steps

- [Execution](./execution.md) - How kernels are invoked
- [Fusion Engine](../fusion/overview.md) - Kernel fusion
