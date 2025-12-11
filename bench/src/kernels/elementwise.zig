//! Elementwise operation benchmarks.
//!
//! Tests memory bandwidth and SIMD efficiency.

const std = @import("std");
const ziterion = @import("ziterion").ziterion;
const tenzor = @import("tenzor");

const elementwise = tenzor.backend.cpu.kernels.elementwise;
const OpTag = tenzor.OpTag;

/// Generate unary op benchmark.
fn makeUnaryBench(comptime size: usize, comptime op: OpTag) fn (*ziterion.State) void {
    return struct {
        fn benchmark(state: *ziterion.State) void {
            var input: [size]f32 = undefined;
            var output: [size]f32 = undefined;

            for (&input, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 100)) * 0.01;

            state.setItemsProcessed(size);
            state.setBytesProcessed(size * 2 * @sizeOf(f32)); // read + write

            elementwise.unaryOp(op, f32, &input, &output);
            ziterion.doNotOptimize(output);
        }
    }.benchmark;
}

/// Generate binary op benchmark.
fn makeBinaryBench(comptime size: usize, comptime op: OpTag) fn (*ziterion.State) void {
    return struct {
        fn benchmark(state: *ziterion.State) void {
            var a: [size]f32 = undefined;
            var b: [size]f32 = undefined;
            var output: [size]f32 = undefined;

            for (&a, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 100)) * 0.01;
            for (&b, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i + 7) % 100)) * 0.01 + 0.01;

            state.setItemsProcessed(size);
            state.setBytesProcessed(size * 3 * @sizeOf(f32)); // 2 reads + 1 write

            elementwise.binaryOp(op, f32, &a, &b, &output);
            ziterion.doNotOptimize(output);
        }
    }.benchmark;
}

pub const benchmarks = struct {
    // Small (cache resident) - measure compute
    pub const relu_64K = makeUnaryBench(65536, .relu);
    pub const exp_64K = makeUnaryBench(65536, .exp);
    pub const tanh_64K = makeUnaryBench(65536, .tanh);

    // Large (memory bound) - measure bandwidth
    pub const relu_1M = makeUnaryBench(1_000_000, .relu);
    pub const exp_1M = makeUnaryBench(1_000_000, .exp);
    pub const tanh_1M = makeUnaryBench(1_000_000, .tanh);

    // Binary ops
    pub const add_64K = makeBinaryBench(65536, .add);
    pub const mul_64K = makeBinaryBench(65536, .mul);
    pub const add_1M = makeBinaryBench(1_000_000, .add);
    pub const mul_1M = makeBinaryBench(1_000_000, .mul);
};
