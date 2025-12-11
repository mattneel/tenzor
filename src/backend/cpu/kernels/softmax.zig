//! Softmax CPU kernel with numerical stability.
//!
//! Computes softmax along the last axis with numerical stability:
//! softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
//!
//! Designed for attention patterns: [B, H, S, S] with axis=-1.
//!
//! Parallel versions available for multi-threaded execution.

const std = @import("std");
const simd = @import("../simd.zig");
const threading = @import("../threading.zig");

/// Softmax along the last axis.
/// Input/output shape: [..., axis_size]
/// Processes each "row" (last dimension) independently.
///
/// Optional mask can be provided for attention masking.
/// Mask values of -inf (or very negative) zero out those positions.
pub fn softmaxLastAxis(
    comptime T: type,
    input: []const T,
    output: []T,
    comptime ndim: usize,
    shape: [ndim]usize,
    mask: ?[]const T,
) void {
    const axis_size = shape[ndim - 1];
    const num_rows = blk: {
        var n: usize = 1;
        for (0..ndim - 1) |i| {
            n *= shape[i];
        }
        break :blk n;
    };

    // Process each row independently
    for (0..num_rows) |row| {
        const row_start = row * axis_size;
        const row_end = row_start + axis_size;
        const row_input = input[row_start..row_end];
        const row_output = output[row_start..row_end];
        const row_mask = if (mask) |m| m[row_start..row_end] else null;

        softmaxRow(T, row_input, row_output, row_mask);
    }
}

/// Softmax a single row (1D slice) with optional mask.
/// This is the numerically stable version:
/// 1. max_val = max(x + mask)  [mask is added, -inf for masked]
/// 2. shifted = x + mask - max_val
/// 3. exp_vals = exp(shifted)
/// 4. sum_val = sum(exp_vals)
/// 5. output = exp_vals / sum_val
fn softmaxRow(
    comptime T: type,
    input: []const T,
    output: []T,
    mask: ?[]const T,
) void {
    const len = input.len;
    if (len == 0) return;

    const vec_len = simd.suggestVectorLength(T);

    // Step 1: Find max value (with mask if present)
    var max_val: T = -std.math.inf(T);

    if (mask) |m| {
        var i: usize = 0;
        // SIMD max with mask
        var max_vec = simd.splat(T, -std.math.inf(T));
        while (i + vec_len <= len) : (i += vec_len) {
            const v = simd.load(T, input[i..]);
            const mv = simd.load(T, m[i..]);
            const masked = simd.add(T, v, mv);
            max_vec = simd.max(T, max_vec, masked);
        }
        max_val = simd.reduceMax(T, max_vec);
        // Scalar remainder
        while (i < len) : (i += 1) {
            max_val = @max(max_val, input[i] + m[i]);
        }
    } else {
        var i: usize = 0;
        var max_vec = simd.splat(T, -std.math.inf(T));
        while (i + vec_len <= len) : (i += vec_len) {
            const v = simd.load(T, input[i..]);
            max_vec = simd.max(T, max_vec, v);
        }
        max_val = simd.reduceMax(T, max_vec);
        while (i < len) : (i += 1) {
            max_val = @max(max_val, input[i]);
        }
    }

    // Step 2 & 3: Compute exp(x + mask - max) and accumulate sum
    var sum_val: T = 0;
    const max_vec_val = simd.splat(T, max_val);

    if (mask) |m| {
        var i: usize = 0;
        var sum_vec = simd.splat(T, 0);
        while (i + vec_len <= len) : (i += vec_len) {
            const v = simd.load(T, input[i..]);
            const mv = simd.load(T, m[i..]);
            const shifted = simd.sub(T, simd.add(T, v, mv), max_vec_val);
            const exp_v = simd.exp(T, shifted);
            simd.store(T, exp_v, output[i..]);
            sum_vec = simd.add(T, sum_vec, exp_v);
        }
        sum_val = simd.reduceAdd(T, sum_vec);
        // Scalar remainder
        while (i < len) : (i += 1) {
            const shifted = input[i] + m[i] - max_val;
            const exp_v = @exp(shifted);
            output[i] = exp_v;
            sum_val += exp_v;
        }
    } else {
        var i: usize = 0;
        var sum_vec = simd.splat(T, 0);
        while (i + vec_len <= len) : (i += vec_len) {
            const v = simd.load(T, input[i..]);
            const shifted = simd.sub(T, v, max_vec_val);
            const exp_v = simd.exp(T, shifted);
            simd.store(T, exp_v, output[i..]);
            sum_vec = simd.add(T, sum_vec, exp_v);
        }
        sum_val = simd.reduceAdd(T, sum_vec);
        while (i < len) : (i += 1) {
            const shifted = input[i] - max_val;
            const exp_v = @exp(shifted);
            output[i] = exp_v;
            sum_val += exp_v;
        }
    }

    // Step 4: Normalize by sum
    const inv_sum = 1.0 / sum_val;
    const inv_sum_vec = simd.splat(T, inv_sum);

    var i: usize = 0;
    while (i + vec_len <= len) : (i += vec_len) {
        const v = simd.load(T, output[i..]);
        const normalized = simd.mul(T, v, inv_sum_vec);
        simd.store(T, normalized, output[i..]);
    }
    while (i < len) : (i += 1) {
        output[i] *= inv_sum;
    }
}

/// Softmax along an arbitrary axis.
/// For attention we typically use axis=-1 (last), but this is the general case.
pub fn softmaxAxis(
    comptime T: type,
    input: []const T,
    output: []T,
    comptime ndim: usize,
    shape: [ndim]usize,
    comptime axis: usize,
    mask: ?[]const T,
) void {
    // For last axis, use optimized path
    if (axis == ndim - 1) {
        softmaxLastAxis(T, input, output, ndim, shape, mask);
        return;
    }

    // General case: compute outer_size, axis_size, inner_size
    var outer_size: usize = 1;
    for (0..axis) |i| {
        outer_size *= shape[i];
    }

    const axis_size = shape[axis];

    var inner_size: usize = 1;
    for ((axis + 1)..ndim) |i| {
        inner_size *= shape[i];
    }

    // For each (outer, inner) position, compute softmax over axis
    for (0..outer_size) |outer| {
        for (0..inner_size) |inner| {
            // Step 1: Find max
            var max_val: T = -std.math.inf(T);
            for (0..axis_size) |a| {
                const idx = outer * axis_size * inner_size + a * inner_size + inner;
                const val = if (mask) |m| input[idx] + m[idx] else input[idx];
                max_val = @max(max_val, val);
            }

            // Step 2 & 3: Compute exp and sum
            var sum_val: T = 0;
            for (0..axis_size) |a| {
                const idx = outer * axis_size * inner_size + a * inner_size + inner;
                const val = if (mask) |m| input[idx] + m[idx] else input[idx];
                const exp_v = @exp(val - max_val);
                output[idx] = exp_v;
                sum_val += exp_v;
            }

            // Step 4: Normalize
            const inv_sum = 1.0 / sum_val;
            for (0..axis_size) |a| {
                const idx = outer * axis_size * inner_size + a * inner_size + inner;
                output[idx] *= inv_sum;
            }
        }
    }
}

// ============================================================================
// Parallel Versions
// ============================================================================

/// Parallel softmax along the last axis.
/// Distributes rows across threads for multi-core execution.
pub fn softmaxLastAxisParallel(
    comptime T: type,
    input: []const T,
    output: []T,
    comptime ndim: usize,
    shape: [ndim]usize,
    mask: ?[]const T,
    pool: *threading.ThreadPool,
) void {
    const axis_size = shape[ndim - 1];
    const num_rows = blk: {
        var n: usize = 1;
        for (0..ndim - 1) |i| {
            n *= shape[i];
        }
        break :blk n;
    };

    const Context = struct {
        input: []const T,
        output: []T,
        axis_size: usize,
        mask: ?[]const T,
    };

    const ctx = Context{
        .input = input,
        .output = output,
        .axis_size = axis_size,
        .mask = mask,
    };

    pool.parallelForBatch(num_rows, ctx, struct {
        fn work(c: Context, start: usize, end: usize) void {
            for (start..end) |row| {
                const row_start = row * c.axis_size;
                const row_end = row_start + c.axis_size;
                const row_input = c.input[row_start..row_end];
                const row_output = c.output[row_start..row_end];
                const row_mask = if (c.mask) |m| m[row_start..row_end] else null;
                softmaxRow(T, row_input, row_output, row_mask);
            }
        }
    }.work);
}

/// Parallel softmax along an arbitrary axis.
pub fn softmaxAxisParallel(
    comptime T: type,
    input: []const T,
    output: []T,
    comptime ndim: usize,
    shape: [ndim]usize,
    comptime axis: usize,
    mask: ?[]const T,
    pool: *threading.ThreadPool,
) void {
    // For last axis, use optimized parallel path
    if (axis == ndim - 1) {
        softmaxLastAxisParallel(T, input, output, ndim, shape, mask, pool);
        return;
    }

    // General case: compute outer_size, axis_size, inner_size
    var outer_size: usize = 1;
    for (0..axis) |i| {
        outer_size *= shape[i];
    }

    const axis_size = shape[axis];

    var inner_size: usize = 1;
    for ((axis + 1)..ndim) |i| {
        inner_size *= shape[i];
    }

    const num_positions = outer_size * inner_size;

    const Context = struct {
        input: []const T,
        output: []T,
        mask: ?[]const T,
        axis_size: usize,
        inner_size: usize,
        outer_size: usize,
    };

    const ctx = Context{
        .input = input,
        .output = output,
        .mask = mask,
        .axis_size = axis_size,
        .inner_size = inner_size,
        .outer_size = outer_size,
    };

    pool.parallelForBatch(num_positions, ctx, struct {
        fn work(c: Context, start: usize, end: usize) void {
            for (start..end) |pos| {
                const outer = pos / c.inner_size;
                const inner = pos % c.inner_size;

                // Step 1: Find max
                var max_val: T = -std.math.inf(T);
                for (0..c.axis_size) |a| {
                    const idx = outer * c.axis_size * c.inner_size + a * c.inner_size + inner;
                    const val = if (c.mask) |m| c.input[idx] + m[idx] else c.input[idx];
                    max_val = @max(max_val, val);
                }

                // Step 2 & 3: Compute exp and sum
                var sum_val: T = 0;
                for (0..c.axis_size) |a| {
                    const idx = outer * c.axis_size * c.inner_size + a * c.inner_size + inner;
                    const val = if (c.mask) |m| c.input[idx] + m[idx] else c.input[idx];
                    const exp_v = @exp(val - max_val);
                    c.output[idx] = exp_v;
                    sum_val += exp_v;
                }

                // Step 4: Normalize
                const inv_sum = 1.0 / sum_val;
                for (0..c.axis_size) |a| {
                    const idx = outer * c.axis_size * c.inner_size + a * c.inner_size + inner;
                    c.output[idx] *= inv_sum;
                }
            }
        }
    }.work);
}

// ============================================================================
// Tests
// ============================================================================

test "softmax basic 1D" {
    const input = [_]f32{ 1.0, 2.0, 3.0 };
    var output: [3]f32 = undefined;

    softmaxLastAxis(f32, &input, &output, 1, .{3}, null);

    // exp(1), exp(2), exp(3) then normalize
    const e1 = @exp(@as(f32, 1.0 - 3.0));
    const e2 = @exp(@as(f32, 2.0 - 3.0));
    const e3 = @exp(@as(f32, 3.0 - 3.0));
    const sum = e1 + e2 + e3;

    try std.testing.expectApproxEqAbs(e1 / sum, output[0], 1e-5);
    try std.testing.expectApproxEqAbs(e2 / sum, output[1], 1e-5);
    try std.testing.expectApproxEqAbs(e3 / sum, output[2], 1e-5);

    // Should sum to 1
    const total = output[0] + output[1] + output[2];
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), total, 1e-5);
}

test "softmax numerical stability" {
    // Large values that would overflow without max subtraction
    const input = [_]f32{ 1000.0, 1001.0, 1002.0 };
    var output: [3]f32 = undefined;

    softmaxLastAxis(f32, &input, &output, 1, .{3}, null);

    // Should still produce valid probabilities
    try std.testing.expect(std.math.isFinite(output[0]));
    try std.testing.expect(std.math.isFinite(output[1]));
    try std.testing.expect(std.math.isFinite(output[2]));

    const total = output[0] + output[1] + output[2];
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), total, 1e-5);

    // Largest input should have highest probability
    try std.testing.expect(output[2] > output[1]);
    try std.testing.expect(output[1] > output[0]);
}

test "softmax 2D batch" {
    // 2 rows of 3 elements each
    const input = [_]f32{
        1.0, 2.0, 3.0, // row 0
        0.0, 0.0, 0.0, // row 1 (uniform)
    };
    var output: [6]f32 = undefined;

    softmaxLastAxis(f32, &input, &output, 2, .{ 2, 3 }, null);

    // Row 0: same as basic test
    const e1 = @exp(@as(f32, 1.0 - 3.0));
    const e2 = @exp(@as(f32, 2.0 - 3.0));
    const e3 = @exp(@as(f32, 3.0 - 3.0));
    const sum0 = e1 + e2 + e3;
    try std.testing.expectApproxEqAbs(e1 / sum0, output[0], 1e-5);
    try std.testing.expectApproxEqAbs(e2 / sum0, output[1], 1e-5);
    try std.testing.expectApproxEqAbs(e3 / sum0, output[2], 1e-5);

    // Row 1: uniform distribution
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / 3.0), output[3], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / 3.0), output[4], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / 3.0), output[5], 1e-5);
}

test "softmax with attention mask" {
    // Mask out position 0 (set to -inf equivalent)
    const input = [_]f32{ 1.0, 2.0, 3.0 };
    const mask = [_]f32{ -1e9, 0.0, 0.0 }; // Large negative = masked
    var output: [3]f32 = undefined;

    softmaxLastAxis(f32, &input, &output, 1, .{3}, &mask);

    // Position 0 should be ~0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), output[0], 1e-5);

    // Positions 1 and 2 should share the probability mass
    const e2 = @exp(@as(f32, 2.0 - 3.0));
    const e3 = @exp(@as(f32, 3.0 - 3.0));
    const sum = e2 + e3;
    try std.testing.expectApproxEqAbs(e2 / sum, output[1], 1e-5);
    try std.testing.expectApproxEqAbs(e3 / sum, output[2], 1e-5);
}

test "softmax 4D attention pattern" {
    // [B=1, H=2, S=2, S=2] - typical attention shape
    // Softmax over last dim (each 2x2 attention matrix, row-wise)
    const input = [_]f32{
        // Head 0
        1.0, 2.0, // row 0
        3.0, 4.0, // row 1
        // Head 1
        0.0, 0.0, // row 0
        1.0, -1.0, // row 1
    };
    var output: [8]f32 = undefined;

    softmaxLastAxis(f32, &input, &output, 4, .{ 1, 2, 2, 2 }, null);

    // Check each row sums to 1
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), output[0] + output[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), output[2] + output[3], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), output[4] + output[5], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), output[6] + output[7], 1e-5);

    // Head 1, row 0: uniform (both 0)
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), output[4], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), output[5], 1e-5);
}

test "softmax general axis" {
    // 2x3 matrix, softmax along axis 0
    const input = [_]f32{
        1.0, 2.0, 3.0, // row 0
        4.0, 5.0, 6.0, // row 1
    };
    var output: [6]f32 = undefined;

    softmaxAxis(f32, &input, &output, 2, .{ 2, 3 }, 0, null);

    // Column 0: softmax of [1, 4]
    const e1_c0 = @exp(@as(f32, 1.0 - 4.0));
    const e4_c0 = @exp(@as(f32, 4.0 - 4.0));
    const sum_c0 = e1_c0 + e4_c0;
    try std.testing.expectApproxEqAbs(e1_c0 / sum_c0, output[0], 1e-5);
    try std.testing.expectApproxEqAbs(e4_c0 / sum_c0, output[3], 1e-5);

    // Each column should sum to 1
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), output[0] + output[3], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), output[1] + output[4], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), output[2] + output[5], 1e-5);
}
