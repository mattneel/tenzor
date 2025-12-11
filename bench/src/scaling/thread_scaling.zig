//! Thread scaling benchmarks.
//!
//! Measures parallel speedup at different thread counts.

const std = @import("std");
const ziterion = @import("ziterion").ziterion;
const tenzor = @import("tenzor");

const matmul = tenzor.backend.cpu.kernels.matmul;
const threading = tenzor.backend.cpu.threading;

/// Generate thread scaling benchmark for matmul.
fn makeScalingBench(comptime thread_count: u32) fn (*ziterion.State) void {
    return struct {
        fn benchmark(state: *ziterion.State) void {
            const M = 512;
            const K = 512;
            const N = 512;

            var a: [M * K]f32 = undefined;
            var b: [K * N]f32 = undefined;
            var c: [M * N]f32 = undefined;

            for (&a, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 17)) * 0.1;
            for (&b, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 13)) * 0.1;

            state.setItemsProcessed(2 * M * K * N);

            const pool = threading.ThreadPool.create(state.scratchAllocator(), .{
                .thread_count = thread_count,
            }) catch return;
            defer pool.destroy();

            matmul.matmulTiledParallel(f32, pool, &a, &b, &c, M, K, N);
            ziterion.doNotOptimize(c);
        }
    }.benchmark;
}

pub const benchmarks = struct {
    pub const scaling_matmul_t1 = makeScalingBench(1);
    pub const scaling_matmul_t2 = makeScalingBench(2);
    pub const scaling_matmul_t4 = makeScalingBench(4);
    pub const scaling_matmul_t8 = makeScalingBench(8);
};
