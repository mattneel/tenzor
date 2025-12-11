//! Layer normalization benchmarks.
//!
//! Transformer-relevant sizes.

const std = @import("std");
const ziterion = @import("ziterion").ziterion;
const tenzor = @import("tenzor");

const layernorm = tenzor.backend.cpu.kernels.layernorm;

/// Generate layernorm benchmark for 2D input [instances, normalized_size].
fn makeLayerNormBench(comptime instances: usize, comptime norm_size: usize) fn (*ziterion.State) void {
    return struct {
        fn benchmark(state: *ziterion.State) void {
            var input: [instances * norm_size]f32 = undefined;
            var gamma: [norm_size]f32 = undefined;
            var beta: [norm_size]f32 = undefined;
            var output: [instances * norm_size]f32 = undefined;

            for (&input, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 100)) * 0.1;
            for (&gamma) |*v| v.* = 1.0;
            for (&beta) |*v| v.* = 0.0;

            // Per instance: mean (N), var (2N), normalize (3N) = ~6N ops
            state.setItemsProcessed(instances * norm_size * 6);
            state.setBytesProcessed((instances * norm_size * 2 + norm_size * 2) * @sizeOf(f32));

            layernorm.layerNormLastDim(
                f32,
                &input,
                &gamma,
                &beta,
                &output,
                2,
                .{ instances, norm_size },
                layernorm.default_eps,
            );
            ziterion.doNotOptimize(output);
        }
    }.benchmark;
}

pub const benchmarks = struct {
    // Arctic-like: [seq_len, embed_dim]
    pub const layernorm_128x384 = makeLayerNormBench(128, 384);
    pub const layernorm_512x384 = makeLayerNormBench(512, 384);

    // BERT-like
    pub const layernorm_128x768 = makeLayerNormBench(128, 768);
    pub const layernorm_512x768 = makeLayerNormBench(512, 768);

    // Small (single instance)
    pub const layernorm_1x384 = makeLayerNormBench(1, 384);
    pub const layernorm_1x768 = makeLayerNormBench(1, 768);
};
