//! Softmax benchmarks.
//!
//! Attention-relevant sizes for transformer workloads.

const std = @import("std");
const ziterion = @import("ziterion").ziterion;
const tenzor = @import("tenzor");

const softmax = tenzor.backend.cpu.kernels.softmax;

/// Generate softmax benchmark for 2D input [rows, cols].
fn makeSoftmaxBench(comptime rows: usize, comptime cols: usize) fn (*ziterion.State) void {
    return struct {
        fn benchmark(state: *ziterion.State) void {
            var input: [rows * cols]f32 = undefined;
            var output: [rows * cols]f32 = undefined;

            for (&input, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 100)) * 0.1 - 5.0;

            // Each row: find max (N), subtract (N), exp (N), sum (N), div (N) = 5N ops
            state.setItemsProcessed(rows * cols * 5);
            state.setBytesProcessed((rows * cols * 2) * @sizeOf(f32));

            softmax.softmaxLastAxis(f32, &input, &output, 2, .{ rows, cols }, null);
            ziterion.doNotOptimize(output);
        }
    }.benchmark;
}

pub const benchmarks = struct {
    // Attention matrices [seq_len, seq_len]
    pub const softmax_64x64 = makeSoftmaxBench(64, 64);
    pub const softmax_128x128 = makeSoftmaxBench(128, 128);
    pub const softmax_256x256 = makeSoftmaxBench(256, 256);
    pub const softmax_512x512 = makeSoftmaxBench(512, 512);

    // Batch x embedding [batch, vocab/embedding]
    pub const softmax_64x384 = makeSoftmaxBench(64, 384); // Arctic embedding
    pub const softmax_256x768 = makeSoftmaxBench(256, 768); // BERT-like
};
