//! Reduction CPU kernels with SIMD optimization.
//!
//! Provides efficient implementations for reduction operations
//! (sum, mean, max, min, prod) with SIMD acceleration.

const std = @import("std");
const simd = @import("../simd.zig");
const ops = @import("../../../ops/expr.zig");

const OpTag = ops.OpTag;

/// Reduce an entire contiguous slice to a single value.
pub fn reduceAll(
    comptime op: OpTag,
    comptime T: type,
    input: []const T,
) T {
    if (input.len == 0) {
        return identityValue(op, T);
    }

    const vec_len = simd.suggestVectorLength(T);
    var i: usize = 0;

    // Initialize accumulator vector
    var acc_vec = simd.splat(T, identityValue(op, T));

    // SIMD loop
    while (i + vec_len <= input.len) : (i += vec_len) {
        const v = simd.load(T, input[i..]);
        acc_vec = applyReduceVec(op, T, acc_vec, v);
    }

    // Reduce vector to scalar
    var acc = reduceVecToScalar(op, T, acc_vec);

    // Scalar remainder
    while (i < input.len) : (i += 1) {
        acc = applyReduceScalar(op, T, acc, input[i]);
    }

    // Post-processing for mean
    if (op == .mean) {
        acc = acc / @as(T, @floatFromInt(input.len));
    }

    return acc;
}

/// Reduce along a single axis.
/// input_shape and output_shape describe the tensor dimensions.
pub fn reduceAxis(
    comptime op: OpTag,
    comptime T: type,
    comptime ndim: usize,
    input: []const T,
    output: []T,
    input_shape: [ndim]usize,
    axis: usize,
    keepdims: bool,
) void {
    _ = keepdims; // Shape already computed by caller

    // Calculate sizes
    var outer_size: usize = 1;
    for (0..axis) |i| {
        outer_size *= input_shape[i];
    }

    const reduce_size = input_shape[axis];

    var inner_size: usize = 1;
    for ((axis + 1)..ndim) |i| {
        inner_size *= input_shape[i];
    }

    // Perform reduction
    for (0..outer_size) |outer| {
        for (0..inner_size) |inner| {
            var acc = identityValue(op, T);

            for (0..reduce_size) |r| {
                const idx = outer * reduce_size * inner_size + r * inner_size + inner;
                acc = applyReduceScalar(op, T, acc, input[idx]);
            }

            // Post-processing for mean
            if (op == .mean) {
                acc = acc / @as(T, @floatFromInt(reduce_size));
            }

            output[outer * inner_size + inner] = acc;
        }
    }
}

/// Sum reduction of a contiguous slice.
pub fn sum(comptime T: type, input: []const T) T {
    return reduceAll(.sum, T, input);
}

/// Mean reduction of a contiguous slice.
pub fn mean(comptime T: type, input: []const T) T {
    return reduceAll(.mean, T, input);
}

/// Maximum reduction of a contiguous slice.
pub fn max(comptime T: type, input: []const T) T {
    return reduceAll(.reduce_max, T, input);
}

/// Minimum reduction of a contiguous slice.
pub fn min(comptime T: type, input: []const T) T {
    return reduceAll(.reduce_min, T, input);
}

/// Product reduction of a contiguous slice.
pub fn prod(comptime T: type, input: []const T) T {
    return reduceAll(.prod, T, input);
}

/// Variance of a contiguous slice.
pub fn variance(comptime T: type, input: []const T) T {
    if (input.len == 0) return 0;

    const m = mean(T, input);
    var acc: T = 0;

    const vec_len = simd.suggestVectorLength(T);
    const m_vec = simd.splat(T, m);
    var i: usize = 0;

    // SIMD loop
    while (i + vec_len <= input.len) : (i += vec_len) {
        const v = simd.load(T, input[i..]);
        const diff = simd.sub(T, v, m_vec);
        const sq = simd.mul(T, diff, diff);
        acc += simd.reduceAdd(T, sq);
    }

    // Scalar remainder
    while (i < input.len) : (i += 1) {
        const diff = input[i] - m;
        acc += diff * diff;
    }

    return acc / @as(T, @floatFromInt(input.len));
}

/// Standard deviation of a contiguous slice.
pub fn std_dev(comptime T: type, input: []const T) T {
    return @sqrt(variance(T, input));
}

/// Argmax - returns the index of the maximum value.
pub fn argmax(comptime T: type, input: []const T) usize {
    if (input.len == 0) return 0;

    var max_val = input[0];
    var max_idx: usize = 0;

    for (input[1..], 1..) |v, i| {
        if (v > max_val) {
            max_val = v;
            max_idx = i;
        }
    }

    return max_idx;
}

/// Argmin - returns the index of the minimum value.
pub fn argmin(comptime T: type, input: []const T) usize {
    if (input.len == 0) return 0;

    var min_val = input[0];
    var min_idx: usize = 0;

    for (input[1..], 1..) |v, i| {
        if (v < min_val) {
            min_val = v;
            min_idx = i;
        }
    }

    return min_idx;
}

// ============================================================================
// Helper functions
// ============================================================================

/// Identity value for a reduction operation.
fn identityValue(comptime op: OpTag, comptime T: type) T {
    return switch (op) {
        .sum, .mean => 0,
        .prod => 1,
        .reduce_max => -std.math.inf(T),
        .reduce_min => std.math.inf(T),
        else => @compileError("Unsupported reduction: " ++ @tagName(op)),
    };
}

/// Apply reduction operation to two vectors.
fn applyReduceVec(comptime op: OpTag, comptime T: type, acc: simd.Vec(T), v: simd.Vec(T)) simd.Vec(T) {
    return switch (op) {
        .sum, .mean => simd.add(T, acc, v),
        .prod => simd.mul(T, acc, v),
        .reduce_max => simd.max(T, acc, v),
        .reduce_min => simd.min(T, acc, v),
        else => @compileError("Unsupported reduction: " ++ @tagName(op)),
    };
}

/// Reduce a vector to a scalar.
fn reduceVecToScalar(comptime op: OpTag, comptime T: type, v: simd.Vec(T)) T {
    return switch (op) {
        .sum, .mean => simd.reduceAdd(T, v),
        .prod => simd.reduceMul(T, v),
        .reduce_max => simd.reduceMax(T, v),
        .reduce_min => simd.reduceMin(T, v),
        else => @compileError("Unsupported reduction: " ++ @tagName(op)),
    };
}

/// Apply reduction operation to two scalars.
fn applyReduceScalar(comptime op: OpTag, comptime T: type, acc: T, v: T) T {
    return switch (op) {
        .sum, .mean => acc + v,
        .prod => acc * v,
        .reduce_max => @max(acc, v),
        .reduce_min => @min(acc, v),
        else => @compileError("Unsupported reduction: " ++ @tagName(op)),
    };
}

// ============================================================================
// Tests
// ============================================================================

test "sum" {
    const input = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const result = sum(f32, &input);
    try std.testing.expectApproxEqAbs(@as(f32, 36), result, 1e-6);
}

test "sum with remainder" {
    const input = [_]f32{ 1, 2, 3, 4, 5 };
    const result = sum(f32, &input);
    try std.testing.expectApproxEqAbs(@as(f32, 15), result, 1e-6);
}

test "mean" {
    const input = [_]f32{ 2, 4, 6, 8 };
    const result = mean(f32, &input);
    try std.testing.expectApproxEqAbs(@as(f32, 5), result, 1e-6);
}

test "max" {
    const input = [_]f32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    const result = max(f32, &input);
    try std.testing.expectApproxEqAbs(@as(f32, 9), result, 1e-6);
}

test "min" {
    const input = [_]f32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    const result = min(f32, &input);
    try std.testing.expectApproxEqAbs(@as(f32, 1), result, 1e-6);
}

test "prod" {
    const input = [_]f32{ 1, 2, 3, 4 };
    const result = prod(f32, &input);
    try std.testing.expectApproxEqAbs(@as(f32, 24), result, 1e-6);
}

test "variance" {
    // Variance of [1, 2, 3, 4, 5]
    // Mean = 3
    // Variance = ((1-3)^2 + (2-3)^2 + (3-3)^2 + (4-3)^2 + (5-3)^2) / 5
    //          = (4 + 1 + 0 + 1 + 4) / 5 = 2
    const input = [_]f32{ 1, 2, 3, 4, 5 };
    const result = variance(f32, &input);
    try std.testing.expectApproxEqAbs(@as(f32, 2), result, 1e-6);
}

test "std_dev" {
    const input = [_]f32{ 1, 2, 3, 4, 5 };
    const result = std_dev(f32, &input);
    try std.testing.expectApproxEqAbs(@as(f32, 1.4142135), result, 1e-5);
}

test "argmax" {
    const input = [_]f32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    const result = argmax(f32, &input);
    try std.testing.expectEqual(@as(usize, 5), result);
}

test "argmin" {
    const input = [_]f32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    const result = argmin(f32, &input);
    try std.testing.expectEqual(@as(usize, 1), result);
}

test "reduceAxis sum" {
    // 2x3 matrix, reduce axis 1 (columns)
    // [[1, 2, 3], [4, 5, 6]] -> [6, 15]
    const input = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var output: [2]f32 = undefined;

    reduceAxis(.sum, f32, 2, &input, &output, .{ 2, 3 }, 1, false);

    try std.testing.expectApproxEqAbs(@as(f32, 6), output[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 15), output[1], 1e-6);
}

test "reduceAxis sum axis 0" {
    // 2x3 matrix, reduce axis 0 (rows)
    // [[1, 2, 3], [4, 5, 6]] -> [5, 7, 9]
    const input = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var output: [3]f32 = undefined;

    reduceAxis(.sum, f32, 2, &input, &output, .{ 2, 3 }, 0, false);

    try std.testing.expectApproxEqAbs(@as(f32, 5), output[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 7), output[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 9), output[2], 1e-6);
}

test "reduceAxis mean" {
    const input = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var output: [2]f32 = undefined;

    reduceAxis(.mean, f32, 2, &input, &output, .{ 2, 3 }, 1, false);

    try std.testing.expectApproxEqAbs(@as(f32, 2), output[0], 1e-6); // (1+2+3)/3
    try std.testing.expectApproxEqAbs(@as(f32, 5), output[1], 1e-6); // (4+5+6)/3
}

test "reduceAxis max" {
    const input = [_]f32{ 1, 5, 3, 4, 2, 6 };
    var output: [2]f32 = undefined;

    reduceAxis(.reduce_max, f32, 2, &input, &output, .{ 2, 3 }, 1, false);

    try std.testing.expectApproxEqAbs(@as(f32, 5), output[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 6), output[1], 1e-6);
}

test "empty input" {
    const input: []const f32 = &[_]f32{};
    try std.testing.expectEqual(@as(f32, 0), sum(f32, input));
    try std.testing.expectEqual(@as(f32, 1), prod(f32, input));
}
