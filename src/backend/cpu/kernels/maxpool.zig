//! Max Pooling 2D CPU kernel with index tracking for backpropagation.
//!
//! Stores argmax indices during forward pass for efficient O(1) backward.
//! Layout: NHWC (batch, height, width, channels)
//!
//! Parallel versions available for multi-threaded execution.

const std = @import("std");
const threading = @import("../threading.zig");

/// Compute output dimension for pooling.
pub fn poolOutputSize(
    input_size: usize,
    pool_size: usize,
    stride: usize,
) usize {
    return (input_size - pool_size) / stride + 1;
}

/// MaxPool2D forward pass with index tracking.
///
/// Input layout:  [N, H, W, C]
/// Output layout: [N, H_out, W_out, C]
/// Indices:       [N, H_out, W_out, C] - flat index into input for each max
pub fn maxPool2dForward(
    comptime T: type,
    input: []const T,
    output: []T,
    indices: []usize,
    batch: usize,
    in_h: usize,
    in_w: usize,
    channels: usize,
    pool_h: usize,
    pool_w: usize,
    stride_h: usize,
    stride_w: usize,
) void {
    const out_h = poolOutputSize(in_h, pool_h, stride_h);
    const out_w = poolOutputSize(in_w, pool_w, stride_w);

    for (0..batch) |n| {
        for (0..out_h) |oh| {
            for (0..out_w) |ow| {
                const ih_start = oh * stride_h;
                const iw_start = ow * stride_w;

                for (0..channels) |c| {
                    // Find max in the pooling window
                    var max_val: T = -std.math.inf(T);
                    var max_idx: usize = 0;

                    for (0..pool_h) |ph| {
                        for (0..pool_w) |pw| {
                            const ih = ih_start + ph;
                            const iw = iw_start + pw;
                            const in_idx = ((n * in_h + ih) * in_w + iw) * channels + c;
                            const val = input[in_idx];

                            if (val > max_val) {
                                max_val = val;
                                max_idx = in_idx;
                            }
                        }
                    }

                    const out_idx = ((n * out_h + oh) * out_w + ow) * channels + c;
                    output[out_idx] = max_val;
                    indices[out_idx] = max_idx;
                }
            }
        }
    }
}

/// MaxPool2D backward pass using stored indices.
///
/// Scatters gradients from output back to the max positions in input.
/// This is sparse - most grad_input values remain zero.
pub fn maxPool2dBackward(
    comptime T: type,
    grad_output: []const T,
    indices: []const usize,
    grad_input: []T,
    batch: usize,
    in_h: usize,
    in_w: usize,
    channels: usize,
    pool_h: usize,
    pool_w: usize,
    stride_h: usize,
    stride_w: usize,
) void {
    const out_h = poolOutputSize(in_h, pool_h, stride_h);
    const out_w = poolOutputSize(in_w, pool_w, stride_w);

    // Initialize grad_input to zeros
    @memset(grad_input, 0);

    // Scatter gradients to max positions
    const out_size = batch * out_h * out_w * channels;
    for (0..out_size) |i| {
        const max_idx = indices[i];
        grad_input[max_idx] += grad_output[i];
    }
}

/// MaxPool2D backward without pre-stored indices (recomputes argmax).
/// Use this if you didn't store indices during forward.
pub fn maxPool2dBackwardRecompute(
    comptime T: type,
    input: []const T,
    grad_output: []const T,
    grad_input: []T,
    batch: usize,
    in_h: usize,
    in_w: usize,
    channels: usize,
    pool_h: usize,
    pool_w: usize,
    stride_h: usize,
    stride_w: usize,
) void {
    const out_h = poolOutputSize(in_h, pool_h, stride_h);
    const out_w = poolOutputSize(in_w, pool_w, stride_w);

    // Initialize grad_input to zeros
    @memset(grad_input, 0);

    for (0..batch) |n| {
        for (0..out_h) |oh| {
            for (0..out_w) |ow| {
                const ih_start = oh * stride_h;
                const iw_start = ow * stride_w;

                for (0..channels) |c| {
                    // Find argmax (recompute)
                    var max_val: T = -std.math.inf(T);
                    var max_idx: usize = 0;

                    for (0..pool_h) |ph| {
                        for (0..pool_w) |pw| {
                            const ih = ih_start + ph;
                            const iw = iw_start + pw;
                            const in_idx = ((n * in_h + ih) * in_w + iw) * channels + c;
                            const val = input[in_idx];

                            if (val > max_val) {
                                max_val = val;
                                max_idx = in_idx;
                            }
                        }
                    }

                    const out_idx = ((n * out_h + oh) * out_w + ow) * channels + c;
                    grad_input[max_idx] += grad_output[out_idx];
                }
            }
        }
    }
}

// ============================================================================
// Parallel versions
// ============================================================================

/// Parallel MaxPool2D forward pass.
pub fn maxPool2dForwardParallel(
    comptime T: type,
    pool: *threading.ThreadPool,
    input: []const T,
    output: []T,
    indices: []usize,
    batch: usize,
    in_h: usize,
    in_w: usize,
    channels: usize,
    pool_h: usize,
    pool_w: usize,
    stride_h: usize,
    stride_w: usize,
) void {
    const out_h = poolOutputSize(in_h, pool_h, stride_h);
    const out_w = poolOutputSize(in_w, pool_w, stride_w);
    const in_batch_size = in_h * in_w * channels;
    const out_batch_size = out_h * out_w * channels;

    const Context = struct {
        input: []const T,
        output: []T,
        indices: []usize,
        in_h: usize,
        in_w: usize,
        channels: usize,
        out_h: usize,
        out_w: usize,
        pool_h: usize,
        pool_w: usize,
        stride_h: usize,
        stride_w: usize,
        in_batch_size: usize,
        out_batch_size: usize,
    };

    const ctx = Context{
        .input = input,
        .output = output,
        .indices = indices,
        .in_h = in_h,
        .in_w = in_w,
        .channels = channels,
        .out_h = out_h,
        .out_w = out_w,
        .pool_h = pool_h,
        .pool_w = pool_w,
        .stride_h = stride_h,
        .stride_w = stride_w,
        .in_batch_size = in_batch_size,
        .out_batch_size = out_batch_size,
    };

    pool.parallelForBatch(batch, ctx, struct {
        fn work(c: Context, start: usize, end: usize) void {
            for (start..end) |n| {
                maxPool2dForwardSingle(
                    T,
                    c.input[n * c.in_batch_size ..][0..c.in_batch_size],
                    c.output[n * c.out_batch_size ..][0..c.out_batch_size],
                    c.indices[n * c.out_batch_size ..][0..c.out_batch_size],
                    c.in_h,
                    c.in_w,
                    c.channels,
                    c.out_h,
                    c.out_w,
                    c.pool_h,
                    c.pool_w,
                    c.stride_h,
                    c.stride_w,
                );
            }
        }
    }.work);
}

/// Single-batch maxpool forward (helper for parallel version).
fn maxPool2dForwardSingle(
    comptime T: type,
    input: []const T,
    output: []T,
    indices: []usize,
    _: usize, // in_h (unused but kept for API consistency)
    in_w: usize,
    channels: usize,
    out_h: usize,
    out_w: usize,
    pool_h: usize,
    pool_w: usize,
    stride_h: usize,
    stride_w: usize,
) void {
    for (0..out_h) |oh| {
        for (0..out_w) |ow| {
            const ih_start = oh * stride_h;
            const iw_start = ow * stride_w;

            for (0..channels) |c| {
                var max_val: T = -std.math.inf(T);
                var max_idx: usize = 0;

                for (0..pool_h) |ph| {
                    for (0..pool_w) |pw| {
                        const ih = ih_start + ph;
                        const iw = iw_start + pw;
                        const in_idx = (ih * in_w + iw) * channels + c;
                        const val = input[in_idx];

                        if (val > max_val) {
                            max_val = val;
                            max_idx = in_idx;
                        }
                    }
                }

                const out_idx = (oh * out_w + ow) * channels + c;
                output[out_idx] = max_val;
                indices[out_idx] = max_idx;
            }
        }
    }
}

/// Parallel MaxPool2D backward pass.
pub fn maxPool2dBackwardParallel(
    comptime T: type,
    pool: *threading.ThreadPool,
    grad_output: []const T,
    indices: []const usize,
    grad_input: []T,
    batch: usize,
    in_h: usize,
    in_w: usize,
    channels: usize,
    pool_h: usize,
    pool_w: usize,
    stride_h: usize,
    stride_w: usize,
) void {
    const out_h = poolOutputSize(in_h, pool_h, stride_h);
    const out_w = poolOutputSize(in_w, pool_w, stride_w);
    const in_batch_size = in_h * in_w * channels;
    const out_batch_size = out_h * out_w * channels;

    // Initialize grad_input to zeros
    @memset(grad_input, 0);

    const Context = struct {
        grad_output: []const T,
        indices: []const usize,
        grad_input: []T,
        in_batch_size: usize,
        out_batch_size: usize,
    };

    const ctx = Context{
        .grad_output = grad_output,
        .indices = indices,
        .grad_input = grad_input,
        .in_batch_size = in_batch_size,
        .out_batch_size = out_batch_size,
    };

    pool.parallelForBatch(batch, ctx, struct {
        fn work(c: Context, start: usize, end: usize) void {
            for (start..end) |n| {
                const out_offset = n * c.out_batch_size;
                const in_offset = n * c.in_batch_size;

                for (0..c.out_batch_size) |i| {
                    const local_max_idx = c.indices[out_offset + i];
                    // Convert to global index (add batch offset)
                    c.grad_input[in_offset + local_max_idx] += c.grad_output[out_offset + i];
                }
            }
        }
    }.work);
}

// ============================================================================
// Tests
// ============================================================================

test "maxpool output size calculation" {
    // 24x24 input, 2x2 pool, stride 2 -> 12x12
    try std.testing.expectEqual(@as(usize, 12), poolOutputSize(24, 2, 2));

    // 8x8 input, 2x2 pool, stride 2 -> 4x4
    try std.testing.expectEqual(@as(usize, 4), poolOutputSize(8, 2, 2));
}

test "maxpool2d forward basic" {
    // 1x4x4x1 input, 2x2 pool, stride 2 -> 1x2x2x1
    const input = [_]f32{
        1,  2,  3,  4,
        5,  6,  7,  8,
        9,  10, 11, 12,
        13, 14, 15, 16,
    };

    var output: [4]f32 = undefined;
    var indices: [4]usize = undefined;

    maxPool2dForward(
        f32,
        &input,
        &output,
        &indices,
        1, // batch
        4,
        4,
        1, // in_h, in_w, channels
        2,
        2, // pool_h, pool_w
        2,
        2, // stride_h, stride_w
    );

    // Max of each 2x2 block
    try std.testing.expectApproxEqAbs(@as(f32, 6), output[0], 1e-6); // max(1,2,5,6)
    try std.testing.expectApproxEqAbs(@as(f32, 8), output[1], 1e-6); // max(3,4,7,8)
    try std.testing.expectApproxEqAbs(@as(f32, 14), output[2], 1e-6); // max(9,10,13,14)
    try std.testing.expectApproxEqAbs(@as(f32, 16), output[3], 1e-6); // max(11,12,15,16)

    // Check indices (flat indices into input)
    try std.testing.expectEqual(@as(usize, 5), indices[0]); // position of 6
    try std.testing.expectEqual(@as(usize, 7), indices[1]); // position of 8
    try std.testing.expectEqual(@as(usize, 13), indices[2]); // position of 14
    try std.testing.expectEqual(@as(usize, 15), indices[3]); // position of 16
}

test "maxpool2d backward" {
    // 1x4x4x1 input, 2x2 pool, stride 2
    const input = [_]f32{
        1,  2,  3,  4,
        5,  6,  7,  8,
        9,  10, 11, 12,
        13, 14, 15, 16,
    };

    var output: [4]f32 = undefined;
    var indices: [4]usize = undefined;

    maxPool2dForward(f32, &input, &output, &indices, 1, 4, 4, 1, 2, 2, 2, 2);

    // Backward with grad_output = [1, 2, 3, 4]
    const grad_output = [_]f32{ 1, 2, 3, 4 };
    var grad_input: [16]f32 = undefined;

    maxPool2dBackward(f32, &grad_output, &indices, &grad_input, 1, 4, 4, 1, 2, 2, 2, 2);

    // Gradients should only be at max positions
    // Position 5 (value 6) gets grad 1
    // Position 7 (value 8) gets grad 2
    // Position 13 (value 14) gets grad 3
    // Position 15 (value 16) gets grad 4
    try std.testing.expectApproxEqAbs(@as(f32, 0), grad_input[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1), grad_input[5], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2), grad_input[7], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3), grad_input[13], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4), grad_input[15], 1e-6);
}

test "maxpool2d backward recompute matches indexed" {
    const input = [_]f32{
        1,  2,  3,  4,
        5,  6,  7,  8,
        9,  10, 11, 12,
        13, 14, 15, 16,
    };

    var output: [4]f32 = undefined;
    var indices: [4]usize = undefined;

    maxPool2dForward(f32, &input, &output, &indices, 1, 4, 4, 1, 2, 2, 2, 2);

    const grad_output = [_]f32{ 1, 2, 3, 4 };

    var grad_input_indexed: [16]f32 = undefined;
    var grad_input_recompute: [16]f32 = undefined;

    maxPool2dBackward(f32, &grad_output, &indices, &grad_input_indexed, 1, 4, 4, 1, 2, 2, 2, 2);
    maxPool2dBackwardRecompute(f32, &input, &grad_output, &grad_input_recompute, 1, 4, 4, 1, 2, 2, 2, 2);

    // Both methods should give same result
    for (grad_input_indexed, grad_input_recompute) |a, b| {
        try std.testing.expectApproxEqAbs(a, b, 1e-6);
    }
}

test "maxpool2d multi-channel" {
    // 1x2x2x2 input (2 channels), 2x2 pool, stride 2 -> 1x1x1x2
    const input = [_]f32{
        1, 10, // (0,0) ch0=1, ch1=10
        2, 20, // (0,1) ch0=2, ch1=20
        3, 30, // (1,0) ch0=3, ch1=30
        4, 40, // (1,1) ch0=4, ch1=40
    };

    var output: [2]f32 = undefined;
    var indices: [2]usize = undefined;

    maxPool2dForward(f32, &input, &output, &indices, 1, 2, 2, 2, 2, 2, 2, 2);

    // Channel 0: max(1,2,3,4) = 4
    // Channel 1: max(10,20,30,40) = 40
    try std.testing.expectApproxEqAbs(@as(f32, 4), output[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 40), output[1], 1e-6);
}
