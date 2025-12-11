//! Reduction operation benchmarks.
//!
//! Tests sum, max, mean reductions.

const std = @import("std");
const ziterion = @import("ziterion").ziterion;
const tenzor = @import("tenzor");

const reduce = tenzor.backend.cpu.kernels.reduce;

/// Generate reduction benchmark.
fn makeReduceBench(comptime size: usize, comptime op: enum { sum, max, mean }) fn (*ziterion.State) void {
    return struct {
        fn benchmark(state: *ziterion.State) void {
            var input: [size]f32 = undefined;

            for (&input, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 1000)) * 0.001;

            state.setItemsProcessed(size);
            state.setBytesProcessed(size * @sizeOf(f32));

            const result = switch (op) {
                .sum => reduce.sum(f32, &input),
                .max => reduce.max(f32, &input),
                .mean => reduce.mean(f32, &input),
            };
            ziterion.doNotOptimize(result);
        }
    }.benchmark;
}

/// Argmax benchmark.
fn makeArgmaxBench(comptime size: usize) fn (*ziterion.State) void {
    return struct {
        fn benchmark(state: *ziterion.State) void {
            var input: [size]f32 = undefined;

            for (&input, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 1000)) * 0.001;

            state.setItemsProcessed(size);
            state.setBytesProcessed(size * @sizeOf(f32));

            const result = reduce.argmax(f32, &input);
            ziterion.doNotOptimize(result);
        }
    }.benchmark;
}

pub const benchmarks = struct {
    // Small reductions
    pub const sum_64K = makeReduceBench(65536, .sum);
    pub const max_64K = makeReduceBench(65536, .max);
    pub const mean_64K = makeReduceBench(65536, .mean);

    // Large reductions
    pub const sum_1M = makeReduceBench(1_000_000, .sum);
    pub const max_1M = makeReduceBench(1_000_000, .max);
    pub const mean_1M = makeReduceBench(1_000_000, .mean);

    // Argmax
    pub const argmax_64K = makeArgmaxBench(65536);
    pub const argmax_1M = makeArgmaxBench(1_000_000);
};
