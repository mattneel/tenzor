//! Matrix multiplication benchmarks.
//!
//! Covers naive, tiled, and parallel variants at various sizes.

const std = @import("std");
const ziterion = @import("ziterion").ziterion;
const tenzor = @import("tenzor");

const matmul = tenzor.backend.cpu.kernels.matmul;
const threading = tenzor.backend.cpu.threading;

/// Generate a matmul benchmark for given dimensions.
fn makeMatmulBench(
    comptime M: usize,
    comptime K: usize,
    comptime N: usize,
    comptime variant: enum { naive, tiled },
) fn (*ziterion.State) void {
    return struct {
        fn benchmark(state: *ziterion.State) void {
            var a: [M * K]f32 = undefined;
            var b: [K * N]f32 = undefined;
            var c: [M * N]f32 = undefined;

            // Deterministic initialization
            for (&a, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 17)) * 0.1;
            for (&b, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 13)) * 0.1;

            // 2*M*K*N FLOPs per matmul (multiply-add)
            state.setItemsProcessed(2 * M * K * N);
            state.setBytesProcessed((M * K + K * N + M * N) * @sizeOf(f32));

            switch (variant) {
                .naive => matmul.matmulNaive(f32, &a, &b, &c, M, K, N),
                .tiled => matmul.matmulTiled(f32, &a, &b, &c, M, K, N),
            }
            ziterion.doNotOptimize(c);
        }
    }.benchmark;
}

/// Generate a parallel matmul benchmark.
fn makeParallelMatmulBench(
    comptime M: usize,
    comptime K: usize,
    comptime N: usize,
) fn (*ziterion.State) void {
    return struct {
        fn benchmark(state: *ziterion.State) void {
            var a: [M * K]f32 = undefined;
            var b: [K * N]f32 = undefined;
            var c: [M * N]f32 = undefined;

            for (&a, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 17)) * 0.1;
            for (&b, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 13)) * 0.1;

            state.setItemsProcessed(2 * M * K * N);
            state.setBytesProcessed((M * K + K * N + M * N) * @sizeOf(f32));

            const pool = threading.ThreadPool.create(state.scratchAllocator(), .{}) catch return;
            defer pool.destroy();

            matmul.matmulTiledParallel(f32, pool, &a, &b, &c, M, K, N);
            ziterion.doNotOptimize(c);
        }
    }.benchmark;
}

pub const benchmarks = struct {
    // Small matrices (L1 cache resident)
    pub const matmul_naive_64x64x64 = makeMatmulBench(64, 64, 64, .naive);
    pub const matmul_tiled_64x64x64 = makeMatmulBench(64, 64, 64, .tiled);

    // Medium matrices (L2/L3 cache)
    pub const matmul_naive_256x256x256 = makeMatmulBench(256, 256, 256, .naive);
    pub const matmul_tiled_256x256x256 = makeMatmulBench(256, 256, 256, .tiled);
    pub const matmul_parallel_256x256x256 = makeParallelMatmulBench(256, 256, 256);

    // Large matrices (memory bound)
    pub const matmul_tiled_512x512x512 = makeMatmulBench(512, 512, 512, .tiled);
    pub const matmul_parallel_512x512x512 = makeParallelMatmulBench(512, 512, 512);

    // Transformer-relevant shapes (embedding dim x seq_len)
    pub const matmul_tiled_384x512x384 = makeMatmulBench(384, 512, 384, .tiled);
    pub const matmul_parallel_384x512x384 = makeParallelMatmulBench(384, 512, 384);
};
