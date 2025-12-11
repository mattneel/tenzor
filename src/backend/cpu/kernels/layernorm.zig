//! Layer normalization CPU kernel.
//!
//! Computes: (x - mean) / sqrt(var + eps) * gamma + beta
//! Applied over the last `normalized_dims` dimensions.
//!
//! Standard epsilon: 1e-5 (same as PyTorch default)

const std = @import("std");
const simd = @import("../simd.zig");

/// Default epsilon for numerical stability.
pub const default_eps: f32 = 1e-5;

/// Layer normalization over the last dimension.
/// Input shape: [..., normalized_size]
/// Gamma/Beta shape: [normalized_size]
///
/// For each "instance" (all dims except last), computes:
/// 1. mean = mean(x)
/// 2. var = mean((x - mean)^2)
/// 3. x_norm = (x - mean) / sqrt(var + eps)
/// 4. output = x_norm * gamma + beta
pub fn layerNormLastDim(
    comptime T: type,
    input: []const T,
    gamma: []const T,
    beta: []const T,
    output: []T,
    comptime ndim: usize,
    shape: [ndim]usize,
    eps: T,
) void {
    const normalized_size = shape[ndim - 1];
    const num_instances = blk: {
        var n: usize = 1;
        for (0..ndim - 1) |i| {
            n *= shape[i];
        }
        break :blk n;
    };

    const vec_len = simd.suggestVectorLength(T);

    // Process each instance
    for (0..num_instances) |inst| {
        const start = inst * normalized_size;
        const end = start + normalized_size;
        const x = input[start..end];
        const y = output[start..end];

        // Step 1: Compute mean
        var sum: T = 0;
        var i: usize = 0;

        // SIMD sum
        var sum_vec = simd.splat(T, 0);
        while (i + vec_len <= normalized_size) : (i += vec_len) {
            const v = simd.load(T, x[i..]);
            sum_vec = simd.add(T, sum_vec, v);
        }
        sum = simd.reduceAdd(T, sum_vec);

        // Scalar remainder
        while (i < normalized_size) : (i += 1) {
            sum += x[i];
        }

        const mean = sum / @as(T, @floatFromInt(normalized_size));

        // Step 2: Compute variance = mean((x - mean)^2)
        var var_sum: T = 0;
        i = 0;

        const mean_vec = simd.splat(T, mean);
        var var_sum_vec = simd.splat(T, 0);
        while (i + vec_len <= normalized_size) : (i += vec_len) {
            const v = simd.load(T, x[i..]);
            const diff = simd.sub(T, v, mean_vec);
            const sq = simd.mul(T, diff, diff);
            var_sum_vec = simd.add(T, var_sum_vec, sq);
        }
        var_sum = simd.reduceAdd(T, var_sum_vec);

        while (i < normalized_size) : (i += 1) {
            const diff = x[i] - mean;
            var_sum += diff * diff;
        }

        const variance = var_sum / @as(T, @floatFromInt(normalized_size));

        // Step 3 & 4: Normalize and scale/shift
        const inv_std = 1.0 / @sqrt(variance + eps);
        const inv_std_vec = simd.splat(T, inv_std);

        i = 0;
        while (i + vec_len <= normalized_size) : (i += vec_len) {
            const v = simd.load(T, x[i..]);
            const g = simd.load(T, gamma[i..]);
            const b = simd.load(T, beta[i..]);

            // normalized = (x - mean) * inv_std
            const normalized = simd.mul(T, simd.sub(T, v, mean_vec), inv_std_vec);
            // output = normalized * gamma + beta
            const result = simd.add(T, simd.mul(T, normalized, g), b);
            simd.store(T, result, y[i..]);
        }

        // Scalar remainder
        while (i < normalized_size) : (i += 1) {
            const normalized = (x[i] - mean) * inv_std;
            y[i] = normalized * gamma[i] + beta[i];
        }
    }
}

/// Layer normalization over multiple trailing dimensions.
/// For example, if normalized_dims=2 and input is [B, H, W, C]:
/// - Normalizes over [W, C] for each [B, H] position
/// - Gamma/Beta shape: [W, C]
pub fn layerNorm(
    comptime T: type,
    input: []const T,
    gamma: []const T,
    beta: []const T,
    output: []T,
    comptime ndim: usize,
    shape: [ndim]usize,
    comptime normalized_dims: usize,
    eps: T,
) void {
    // For the common case of normalizing just the last dim
    if (normalized_dims == 1) {
        layerNormLastDim(T, input, gamma, beta, output, ndim, shape, eps);
        return;
    }

    // General case: compute normalized_size as product of last dims
    const normalized_size = blk: {
        var n: usize = 1;
        for (ndim - normalized_dims..ndim) |i| {
            n *= shape[i];
        }
        break :blk n;
    };

    const num_instances = blk: {
        var n: usize = 1;
        for (0..ndim - normalized_dims) |i| {
            n *= shape[i];
        }
        break :blk n;
    };

    const inv_normalized_size = 1.0 / @as(T, @floatFromInt(normalized_size));

    // Process each instance
    for (0..num_instances) |inst| {
        const start = inst * normalized_size;
        const end = start + normalized_size;
        const x = input[start..end];
        const y = output[start..end];

        // Compute mean
        var sum: T = 0;
        for (x) |v| sum += v;
        const mean = sum * inv_normalized_size;

        // Compute variance
        var var_sum: T = 0;
        for (x) |v| {
            const diff = v - mean;
            var_sum += diff * diff;
        }
        const variance = var_sum * inv_normalized_size;
        const inv_std = 1.0 / @sqrt(variance + eps);

        // Normalize and scale/shift
        for (y, x, gamma, beta) |*out, in, g, b| {
            const normalized = (in - mean) * inv_std;
            out.* = normalized * g + b;
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

test "layernorm basic 1D" {
    // Input: 4 elements, normalize all
    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const gamma = [_]f32{ 1.0, 1.0, 1.0, 1.0 }; // no scaling
    const beta = [_]f32{ 0.0, 0.0, 0.0, 0.0 }; // no shift
    var output: [4]f32 = undefined;

    layerNormLastDim(f32, &input, &gamma, &beta, &output, 1, .{4}, default_eps);

    // After normalization, mean should be ~0, std should be ~1
    var mean: f32 = 0;
    for (output) |v| mean += v;
    mean /= 4;
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), mean, 1e-5);

    var variance: f32 = 0;
    for (output) |v| {
        const diff = v - mean;
        variance += diff * diff;
    }
    variance /= 4;
    // Variance should be close to 1 (but slightly less due to using population variance)
    try std.testing.expect(variance > 0.9 and variance < 1.1);
}

test "layernorm with gamma/beta" {
    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const gamma = [_]f32{ 2.0, 2.0, 2.0, 2.0 }; // scale by 2
    const beta = [_]f32{ 1.0, 1.0, 1.0, 1.0 }; // shift by 1
    var output: [4]f32 = undefined;

    layerNormLastDim(f32, &input, &gamma, &beta, &output, 1, .{4}, default_eps);

    // Mean should be beta (1.0) after transformation
    var mean: f32 = 0;
    for (output) |v| mean += v;
    mean /= 4;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), mean, 1e-5);
}

test "layernorm 2D batch" {
    // 2 instances of 4 elements each
    const input = [_]f32{
        1.0, 2.0, 3.0, 4.0, // instance 0
        10.0, 20.0, 30.0, 40.0, // instance 1 (different scale)
    };
    const gamma = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    const beta = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    var output: [8]f32 = undefined;

    layerNormLastDim(f32, &input, &gamma, &beta, &output, 2, .{ 2, 4 }, default_eps);

    // Both instances should have mean ~0
    var mean0: f32 = 0;
    var mean1: f32 = 0;
    for (output[0..4]) |v| mean0 += v;
    for (output[4..8]) |v| mean1 += v;
    mean0 /= 4;
    mean1 /= 4;

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), mean0, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), mean1, 1e-5);

    // The normalized values should be the same (up to numerical precision)
    // because [1,2,3,4] and [10,20,30,40] have the same relative pattern
    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(output[i], output[4 + i], 1e-5);
    }
}

test "layernorm transformer pattern" {
    // [B=1, S=2, D=4] normalized over D
    const input = [_]f32{
        // Token 0
        1.0,  2.0, 3.0, 4.0,
        // Token 1
        -1.0, 0.0, 1.0, 2.0,
    };
    const gamma = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    const beta = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    var output: [8]f32 = undefined;

    layerNormLastDim(f32, &input, &gamma, &beta, &output, 3, .{ 1, 2, 4 }, default_eps);

    // Each token should have mean ~0
    var mean0: f32 = 0;
    var mean1: f32 = 0;
    for (output[0..4]) |v| mean0 += v;
    for (output[4..8]) |v| mean1 += v;
    mean0 /= 4;
    mean1 /= 4;

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), mean0, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), mean1, 1e-5);
}

test "layernorm numerical stability" {
    // Large values that would be numerically unstable without proper mean subtraction
    const input = [_]f32{ 1e6, 1e6 + 1, 1e6 + 2, 1e6 + 3 };
    const gamma = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    const beta = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    var output: [4]f32 = undefined;

    layerNormLastDim(f32, &input, &gamma, &beta, &output, 1, .{4}, default_eps);

    // Should produce valid finite numbers
    for (output) |v| {
        try std.testing.expect(std.math.isFinite(v));
    }

    // Mean should still be ~0
    var mean: f32 = 0;
    for (output) |v| mean += v;
    mean /= 4;
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), mean, 1e-4);
}
