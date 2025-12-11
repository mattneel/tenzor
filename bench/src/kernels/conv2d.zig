//! 2D Convolution benchmarks.
//!
//! Covers LeNet-5 shapes and larger ResNet-like convolutions.

const std = @import("std");
const ziterion = @import("ziterion").ziterion;
const tenzor = @import("tenzor");

const conv2d = tenzor.backend.cpu.kernels.conv2d;

/// Generate a conv2d forward benchmark.
fn makeConv2dBench(
    comptime batch: usize,
    comptime in_h: usize,
    comptime in_w: usize,
    comptime in_c: usize,
    comptime out_c: usize,
    comptime kernel: usize,
) fn (*ziterion.State) void {
    return struct {
        fn benchmark(state: *ziterion.State) void {
            const out_h = in_h - kernel + 1;
            const out_w = in_w - kernel + 1;

            var input: [batch * in_h * in_w * in_c]f32 = undefined;
            var weight: [out_c * kernel * kernel * in_c]f32 = undefined;
            var bias: [out_c]f32 = undefined;
            var output: [batch * out_h * out_w * out_c]f32 = undefined;

            // Initialize
            for (&input, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 17)) * 0.01;
            for (&weight, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 13)) * 0.01;
            for (&bias) |*v| v.* = 0.1;

            // FLOPs: batch * out_h * out_w * out_c * (2 * kernel^2 * in_c)
            const flops = batch * out_h * out_w * out_c * 2 * kernel * kernel * in_c;
            state.setItemsProcessed(flops);

            conv2d.conv2dForward(
                f32,
                &input,
                &weight,
                &bias,
                &output,
                batch,
                in_h,
                in_w,
                in_c,
                out_c,
                kernel,
                kernel,
                1,
                1,
                0,
                0,
            );
            ziterion.doNotOptimize(output);
        }
    }.benchmark;
}

pub const benchmarks = struct {
    // LeNet conv1: 28x28x1 -> 24x24x6, 5x5 kernel
    pub const conv2d_lenet1_b1 = makeConv2dBench(1, 28, 28, 1, 6, 5);
    pub const conv2d_lenet1_b64 = makeConv2dBench(64, 28, 28, 1, 6, 5);

    // LeNet conv2: 12x12x6 -> 8x8x16, 5x5 kernel
    pub const conv2d_lenet2_b1 = makeConv2dBench(1, 12, 12, 6, 16, 5);
    pub const conv2d_lenet2_b64 = makeConv2dBench(64, 12, 12, 6, 16, 5);

    // Larger conv (ResNet-like first layer)
    pub const conv2d_64x56x56_3x3 = makeConv2dBench(1, 56, 56, 64, 64, 3);
};
