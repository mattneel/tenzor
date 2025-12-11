//! Elementwise CPU kernels with SIMD optimization.
//!
//! Provides efficient implementations for unary and binary operations
//! using SIMD vectorization with scalar fallback for remainders.

const std = @import("std");
const simd = @import("../simd.zig");
const ops = @import("../../../ops/expr.zig");

const OpTag = ops.OpTag;

/// Apply a unary operation to a contiguous slice.
pub fn unaryOp(
    comptime op: OpTag,
    comptime T: type,
    input: []const T,
    output: []T,
) void {
    std.debug.assert(input.len == output.len);

    const vec_len = simd.suggestVectorLength(T);

    var i: usize = 0;

    // SIMD loop
    while (i + vec_len <= input.len) : (i += vec_len) {
        const v = simd.load(T, input[i..]);
        const result = applyUnaryVec(op, T, v);
        simd.store(T, result, output[i..]);
    }

    // Scalar remainder
    while (i < input.len) : (i += 1) {
        output[i] = applyUnaryScalar(op, T, input[i]);
    }
}

/// Apply a binary operation to two contiguous slices.
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

    // SIMD loop
    while (i + vec_len <= lhs.len) : (i += vec_len) {
        const v_lhs = simd.load(T, lhs[i..]);
        const v_rhs = simd.load(T, rhs[i..]);
        const result = applyBinaryVec(op, T, v_lhs, v_rhs);
        simd.store(T, result, output[i..]);
    }

    // Scalar remainder
    while (i < lhs.len) : (i += 1) {
        output[i] = applyBinaryScalar(op, T, lhs[i], rhs[i]);
    }
}

/// Apply a binary operation with scalar broadcast on RHS.
pub fn binaryOpScalarRhs(
    comptime op: OpTag,
    comptime T: type,
    lhs: []const T,
    rhs_scalar: T,
    output: []T,
) void {
    std.debug.assert(lhs.len == output.len);

    const vec_len = simd.suggestVectorLength(T);
    const rhs_vec = simd.splat(T, rhs_scalar);

    var i: usize = 0;

    // SIMD loop
    while (i + vec_len <= lhs.len) : (i += vec_len) {
        const v_lhs = simd.load(T, lhs[i..]);
        const result = applyBinaryVec(op, T, v_lhs, rhs_vec);
        simd.store(T, result, output[i..]);
    }

    // Scalar remainder
    while (i < lhs.len) : (i += 1) {
        output[i] = applyBinaryScalar(op, T, lhs[i], rhs_scalar);
    }
}

/// Apply a binary operation with scalar broadcast on LHS.
pub fn binaryOpScalarLhs(
    comptime op: OpTag,
    comptime T: type,
    lhs_scalar: T,
    rhs: []const T,
    output: []T,
) void {
    std.debug.assert(rhs.len == output.len);

    const vec_len = simd.suggestVectorLength(T);
    const lhs_vec = simd.splat(T, lhs_scalar);

    var i: usize = 0;

    // SIMD loop
    while (i + vec_len <= rhs.len) : (i += vec_len) {
        const v_rhs = simd.load(T, rhs[i..]);
        const result = applyBinaryVec(op, T, lhs_vec, v_rhs);
        simd.store(T, result, output[i..]);
    }

    // Scalar remainder
    while (i < rhs.len) : (i += 1) {
        output[i] = applyBinaryScalar(op, T, lhs_scalar, rhs[i]);
    }
}

/// Apply unary operation to a single vector.
fn applyUnaryVec(comptime op: OpTag, comptime T: type, v: simd.Vec(T)) simd.Vec(T) {
    return switch (op) {
        .neg => simd.neg(T, v),
        .abs => simd.abs(T, v),
        .exp => simd.exp(T, v),
        .log => simd.log(T, v),
        .sqrt => simd.sqrt(T, v),
        .rsqrt => simd.rsqrt(T, v),
        .sin => simd.sin(T, v),
        .cos => simd.cos(T, v),
        .tanh => simd.tanh(T, v),
        .sigmoid => simd.sigmoid(T, v),
        .relu => simd.relu(T, v),
        .gelu => simd.gelu(T, v),
        .silu => simd.silu(T, v),
        .softplus => simd.softplus(T, v),
        .floor => simd.floor(T, v),
        .ceil => simd.ceil(T, v),
        .round => simd.round(T, v),
        else => @compileError("Unsupported unary operation: " ++ @tagName(op)),
    };
}

/// Apply unary operation to a scalar.
fn applyUnaryScalar(comptime op: OpTag, comptime T: type, x: T) T {
    return switch (op) {
        .neg => simd.scalar.neg(x),
        .abs => simd.scalar.abs(x),
        .exp => simd.scalar.exp(x),
        .log => simd.scalar.log(x),
        .sqrt => simd.scalar.sqrt(x),
        .rsqrt => simd.scalar.rsqrt(x),
        .sin => simd.scalar.sin(x),
        .cos => simd.scalar.cos(x),
        .tanh => simd.scalar.tanh(x),
        .sigmoid => simd.scalar.sigmoid(x),
        .relu => simd.scalar.relu(x),
        .gelu => simd.scalar.gelu(x),
        .silu => simd.scalar.silu(x),
        .softplus => simd.scalar.softplus(x),
        .floor => simd.scalar.floor(x),
        .ceil => simd.scalar.ceil(x),
        .round => simd.scalar.round(x),
        else => @compileError("Unsupported unary operation: " ++ @tagName(op)),
    };
}

/// Apply binary operation to two vectors.
fn applyBinaryVec(comptime op: OpTag, comptime T: type, a: simd.Vec(T), b: simd.Vec(T)) simd.Vec(T) {
    return switch (op) {
        .add => simd.add(T, a, b),
        .sub => simd.sub(T, a, b),
        .mul => simd.mul(T, a, b),
        .div => simd.div(T, a, b),
        .max => simd.max(T, a, b),
        .min => simd.min(T, a, b),
        .pow => blk: {
            // pow is element-wise, need to do scalar fallback
            const vec_len = simd.suggestVectorLength(T);
            const a_arr: [vec_len]T = a;
            const b_arr: [vec_len]T = b;
            var result: [vec_len]T = undefined;
            for (&result, a_arr, b_arr) |*r, av, bv| {
                r.* = simd.scalar.pow(av, bv);
            }
            break :blk result;
        },
        else => @compileError("Unsupported binary operation: " ++ @tagName(op)),
    };
}

/// Apply binary operation to scalars.
pub fn applyBinaryScalar(comptime op: OpTag, comptime T: type, a: T, b: T) T {
    return switch (op) {
        .add => simd.scalar.add(a, b),
        .sub => simd.scalar.sub(a, b),
        .mul => simd.scalar.mul(a, b),
        .div => simd.scalar.div(a, b),
        .pow => simd.scalar.pow(a, b),
        .max => simd.scalar.max(a, b),
        .min => simd.scalar.min(a, b),
        else => @compileError("Unsupported binary operation: " ++ @tagName(op)),
    };
}

// ============================================================================
// Broadcasting kernels
// ============================================================================

/// Apply binary operation with broadcasting.
/// Handles the general case where shapes may differ.
pub fn binaryOpBroadcast(
    comptime op: OpTag,
    comptime T: type,
    comptime lhs_ndim: usize,
    comptime rhs_ndim: usize,
    comptime out_ndim: usize,
    lhs: []const T,
    rhs: []const T,
    output: []T,
    lhs_shape: [lhs_ndim]usize,
    rhs_shape: [rhs_ndim]usize,
    out_shape: [out_ndim]usize,
    lhs_strides: [lhs_ndim]usize,
    rhs_strides: [rhs_ndim]usize,
) void {
    // Calculate total elements
    var total: usize = 1;
    for (out_shape) |d| total *= d;

    // For each output element
    for (0..total) |flat_idx| {
        // Convert flat index to multi-dimensional index
        var out_idx: [out_ndim]usize = undefined;
        var remaining = flat_idx;
        for (0..out_ndim) |i| {
            const dim_idx = out_ndim - 1 - i;
            out_idx[dim_idx] = remaining % out_shape[dim_idx];
            remaining /= out_shape[dim_idx];
        }

        // Compute broadcast indices for lhs and rhs
        const lhs_offset = computeBroadcastOffset(lhs_ndim, out_ndim, lhs_shape, lhs_strides, out_idx);
        const rhs_offset = computeBroadcastOffset(rhs_ndim, out_ndim, rhs_shape, rhs_strides, out_idx);

        output[flat_idx] = applyBinaryScalar(op, T, lhs[lhs_offset], rhs[rhs_offset]);
    }
}

/// Compute offset for broadcasted access.
fn computeBroadcastOffset(
    comptime src_ndim: usize,
    comptime out_ndim: usize,
    src_shape: [src_ndim]usize,
    src_strides: [src_ndim]usize,
    out_idx: [out_ndim]usize,
) usize {
    var offset: usize = 0;
    for (0..src_ndim) |i| {
        const out_dim_idx = out_ndim - src_ndim + i;
        // If dimension is 1, it's broadcast - use index 0
        const src_idx = if (src_shape[i] == 1) 0 else out_idx[out_dim_idx];
        offset += src_idx * src_strides[i];
    }
    return offset;
}

// ============================================================================
// Fused operations (for common patterns)
// ============================================================================

/// Fused multiply-add: output = a * b + c
pub fn fusedMulAdd(
    comptime T: type,
    a: []const T,
    b: []const T,
    c: []const T,
    output: []T,
) void {
    std.debug.assert(a.len == b.len);
    std.debug.assert(a.len == c.len);
    std.debug.assert(a.len == output.len);

    const vec_len = simd.suggestVectorLength(T);

    var i: usize = 0;

    // SIMD loop
    while (i + vec_len <= a.len) : (i += vec_len) {
        const va = simd.load(T, a[i..]);
        const vb = simd.load(T, b[i..]);
        const vc = simd.load(T, c[i..]);
        const result = simd.add(T, simd.mul(T, va, vb), vc);
        simd.store(T, result, output[i..]);
    }

    // Scalar remainder
    while (i < a.len) : (i += 1) {
        output[i] = a[i] * b[i] + c[i];
    }
}

/// Fused bias + relu: output = relu(input + bias)
pub fn fusedBiasRelu(
    comptime T: type,
    input: []const T,
    bias: T,
    output: []T,
) void {
    std.debug.assert(input.len == output.len);

    const vec_len = simd.suggestVectorLength(T);
    const bias_vec = simd.splat(T, bias);
    const zero_vec = simd.splat(T, 0);

    var i: usize = 0;

    // SIMD loop
    while (i + vec_len <= input.len) : (i += vec_len) {
        const v = simd.load(T, input[i..]);
        const biased = simd.add(T, v, bias_vec);
        const result = simd.max(T, biased, zero_vec);
        simd.store(T, result, output[i..]);
    }

    // Scalar remainder
    while (i < input.len) : (i += 1) {
        output[i] = @max(input[i] + bias, 0);
    }
}

// ============================================================================
// Tests
// ============================================================================

test "unaryOp relu" {
    const input = [_]f32{ -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 };
    var output: [8]f32 = undefined;

    unaryOp(.relu, f32, &input, &output);

    const expected = [_]f32{ 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 };
    for (output, expected) |out, exp| {
        try std.testing.expectApproxEqAbs(exp, out, 1e-6);
    }
}

test "unaryOp neg" {
    const input = [_]f32{ 1.0, -2.0, 3.0, -4.0 };
    var output: [4]f32 = undefined;

    unaryOp(.neg, f32, &input, &output);

    const expected = [_]f32{ -1.0, 2.0, -3.0, 4.0 };
    for (output, expected) |out, exp| {
        try std.testing.expectEqual(exp, out);
    }
}

test "unaryOp exp" {
    const input = [_]f32{ 0.0, 1.0, 2.0, -1.0 };
    var output: [4]f32 = undefined;

    unaryOp(.exp, f32, &input, &output);

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), output[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.71828), output[1], 1e-4);
}

test "binaryOp add" {
    const lhs = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const rhs = [_]f32{ 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0 };
    var output: [8]f32 = undefined;

    binaryOp(.add, f32, &lhs, &rhs, &output);

    for (output) |out| {
        try std.testing.expectEqual(@as(f32, 9.0), out);
    }
}

test "binaryOp mul" {
    const lhs = [_]f32{ 2.0, 3.0, 4.0, 5.0 };
    const rhs = [_]f32{ 3.0, 4.0, 5.0, 6.0 };
    var output: [4]f32 = undefined;

    binaryOp(.mul, f32, &lhs, &rhs, &output);

    const expected = [_]f32{ 6.0, 12.0, 20.0, 30.0 };
    for (output, expected) |out, exp| {
        try std.testing.expectEqual(exp, out);
    }
}

test "binaryOpScalarRhs add" {
    const lhs = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var output: [4]f32 = undefined;

    binaryOpScalarRhs(.add, f32, &lhs, 10.0, &output);

    const expected = [_]f32{ 11.0, 12.0, 13.0, 14.0 };
    for (output, expected) |out, exp| {
        try std.testing.expectEqual(exp, out);
    }
}

test "fusedMulAdd" {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ 2.0, 2.0, 2.0, 2.0 };
    const c = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    var output: [4]f32 = undefined;

    fusedMulAdd(f32, &a, &b, &c, &output);

    const expected = [_]f32{ 3.0, 5.0, 7.0, 9.0 }; // a*b + c
    for (output, expected) |out, exp| {
        try std.testing.expectEqual(exp, out);
    }
}

test "fusedBiasRelu" {
    const input = [_]f32{ -3.0, -1.0, 1.0, 3.0 };
    var output: [4]f32 = undefined;

    fusedBiasRelu(f32, &input, 2.0, &output);

    // input + 2, then relu: -1, 1, 3, 5 -> 0, 1, 3, 5
    const expected = [_]f32{ 0.0, 1.0, 3.0, 5.0 };
    for (output, expected) |out, exp| {
        try std.testing.expectEqual(exp, out);
    }
}

test "unaryOp sigmoid" {
    const input = [_]f32{ -100.0, 0.0, 100.0, 1.0 };
    var output: [4]f32 = undefined;

    unaryOp(.sigmoid, f32, &input, &output);

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), output[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), output[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), output[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7310586), output[3], 1e-5);
}

test "unaryOp with non-vectorizable size" {
    // Test with a size that doesn't evenly divide the vector length
    const input = [_]f32{ 1.0, 2.0, 3.0 };
    var output: [3]f32 = undefined;

    unaryOp(.neg, f32, &input, &output);

    const expected = [_]f32{ -1.0, -2.0, -3.0 };
    for (output, expected) |out, exp| {
        try std.testing.expectEqual(exp, out);
    }
}
