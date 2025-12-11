//! Transpose CPU kernel.
//!
//! Permutes tensor axes by moving data to new positions.
//! Supports arbitrary permutations for N-dimensional tensors.

const std = @import("std");

/// Transpose a tensor with arbitrary permutation.
/// Moves data from input layout to output layout according to perm.
pub fn transpose(
    comptime T: type,
    comptime ndim: usize,
    input: []const T,
    output: []T,
    in_shape: [ndim]usize,
    perm: [ndim]usize,
) void {
    // Compute input strides (row-major)
    var in_strides: [ndim]usize = undefined;
    var stride: usize = 1;
    var i: usize = ndim;
    while (i > 0) {
        i -= 1;
        in_strides[i] = stride;
        stride *= in_shape[i];
    }

    // Compute output shape
    var out_shape: [ndim]usize = undefined;
    for (perm, 0..) |p, j| {
        out_shape[j] = in_shape[p];
    }

    // Compute output strides
    var out_strides: [ndim]usize = undefined;
    stride = 1;
    i = ndim;
    while (i > 0) {
        i -= 1;
        out_strides[i] = stride;
        stride *= out_shape[i];
    }

    // Total elements
    const numel = blk: {
        var n: usize = 1;
        for (in_shape) |d| n *= d;
        break :blk n;
    };

    // For each output position, compute the corresponding input position
    for (0..numel) |out_idx| {
        // Convert flat output index to multi-dimensional indices
        var out_coords: [ndim]usize = undefined;
        var remaining = out_idx;
        for (0..ndim) |j| {
            out_coords[j] = remaining / out_strides[j];
            remaining = remaining % out_strides[j];
        }

        // Map output coords to input coords using inverse permutation
        var in_coords: [ndim]usize = undefined;
        for (perm, 0..) |p, j| {
            in_coords[p] = out_coords[j];
        }

        // Compute input flat index
        var in_idx: usize = 0;
        for (0..ndim) |j| {
            in_idx += in_coords[j] * in_strides[j];
        }

        output[out_idx] = input[in_idx];
    }
}

/// Optimized 2D transpose (matrix transpose).
pub fn transpose2D(
    comptime T: type,
    input: []const T,
    output: []T,
    rows: usize,
    cols: usize,
) void {
    for (0..rows) |i| {
        for (0..cols) |j| {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}

/// Optimized 4D transpose for attention pattern.
/// Common permutation: [B, S, H, D] -> [B, H, S, D] (perm = [0, 2, 1, 3])
pub fn transpose4D_0213(
    comptime T: type,
    input: []const T,
    output: []T,
    b: usize,
    s: usize,
    h: usize,
    d: usize,
) void {
    // Input strides for [B, S, H, D]
    const in_h_stride: usize = d;
    const in_s_stride: usize = h * d;
    const in_b_stride: usize = s * h * d;

    // Output strides for [B, H, S, D]
    const out_s_stride: usize = d;
    const out_h_stride: usize = s * d;
    const out_b_stride: usize = h * s * d;

    for (0..b) |bi| {
        for (0..s) |si| {
            for (0..h) |hi| {
                const in_base = bi * in_b_stride + si * in_s_stride + hi * in_h_stride;
                const out_base = bi * out_b_stride + hi * out_h_stride + si * out_s_stride;

                // Copy the D elements
                @memcpy(output[out_base..][0..d], input[in_base..][0..d]);
            }
        }
    }
}

/// Optimized 4D transpose for attention output.
/// Common permutation: [B, H, S, D] -> [B, S, H, D] (perm = [0, 2, 1, 3])
pub fn transpose4D_0213_inverse(
    comptime T: type,
    input: []const T,
    output: []T,
    b: usize,
    h: usize,
    s: usize,
    d: usize,
) void {
    // Input is [B, H, S, D], output is [B, S, H, D]
    // This is same operation, just different interpretation
    transpose4D_0213(T, input, output, b, h, s, d);
}

// ============================================================================
// Tests
// ============================================================================

test "transpose 2D" {
    // 2x3 matrix -> 3x2
    const input = [_]f32{
        1, 2, 3,
        4, 5, 6,
    };
    var output: [6]f32 = undefined;

    transpose2D(f32, &input, &output, 2, 3);

    // Expected: [[1, 4], [2, 5], [3, 6]]
    try std.testing.expectEqual(@as(f32, 1), output[0]);
    try std.testing.expectEqual(@as(f32, 4), output[1]);
    try std.testing.expectEqual(@as(f32, 2), output[2]);
    try std.testing.expectEqual(@as(f32, 5), output[3]);
    try std.testing.expectEqual(@as(f32, 3), output[4]);
    try std.testing.expectEqual(@as(f32, 6), output[5]);
}

test "transpose general 3D" {
    // [2, 3, 4] with perm [0, 2, 1] -> [2, 4, 3]
    var input: [24]f32 = undefined;
    for (&input, 0..) |*v, idx| {
        v.* = @floatFromInt(idx);
    }
    var output: [24]f32 = undefined;

    transpose(f32, 3, &input, &output, .{ 2, 3, 4 }, .{ 0, 2, 1 });

    // Verify a few elements
    // input[0, 0, 0] = 0 -> output[0, 0, 0] = 0
    try std.testing.expectEqual(@as(f32, 0), output[0]);

    // input[0, 0, 1] = 1 -> output[0, 1, 0]
    // output shape is [2, 4, 3], so output[0, 1, 0] = 0*12 + 1*3 + 0 = 3
    try std.testing.expectEqual(@as(f32, 1), output[3]);

    // input[0, 1, 0] = 4 -> output[0, 0, 1]
    // output[0, 0, 1] = 0*12 + 0*3 + 1 = 1
    try std.testing.expectEqual(@as(f32, 4), output[1]);
}

test "transpose 4D attention pattern" {
    // [B=1, S=2, H=2, D=3] -> [B=1, H=2, S=2, D=3] with perm [0, 2, 1, 3]
    const input = [_]f32{
        // S=0, H=0
        1,  2,  3,
        // S=0, H=1
        4,  5,  6,
        // S=1, H=0
        7,  8,  9,
        // S=1, H=1
        10, 11, 12,
    };
    var output: [12]f32 = undefined;

    transpose4D_0213(f32, &input, &output, 1, 2, 2, 3);

    // Output layout: [B=1, H=2, S=2, D=3]
    // H=0: S=0 (1,2,3), S=1 (7,8,9)
    // H=1: S=0 (4,5,6), S=1 (10,11,12)
    try std.testing.expectEqual(@as(f32, 1), output[0]);
    try std.testing.expectEqual(@as(f32, 2), output[1]);
    try std.testing.expectEqual(@as(f32, 3), output[2]);
    try std.testing.expectEqual(@as(f32, 7), output[3]);
    try std.testing.expectEqual(@as(f32, 8), output[4]);
    try std.testing.expectEqual(@as(f32, 9), output[5]);
    try std.testing.expectEqual(@as(f32, 4), output[6]);
    try std.testing.expectEqual(@as(f32, 5), output[7]);
    try std.testing.expectEqual(@as(f32, 6), output[8]);
    try std.testing.expectEqual(@as(f32, 10), output[9]);
    try std.testing.expectEqual(@as(f32, 11), output[10]);
    try std.testing.expectEqual(@as(f32, 12), output[11]);
}

test "transpose identity permutation" {
    const input = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var output: [6]f32 = undefined;

    transpose(f32, 2, &input, &output, .{ 2, 3 }, .{ 0, 1 });

    // Should be unchanged
    for (input, output) |i, o| {
        try std.testing.expectEqual(i, o);
    }
}
