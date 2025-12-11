//! LeNet-5 end-to-end model benchmarks.
//!
//! Measures full forward pass and training step throughput.

const std = @import("std");
const ziterion = @import("ziterion").ziterion;
const tenzor = @import("tenzor");

const LeNet = tenzor.model.lenet.LeNet;
const LeNetConfig = tenzor.model.lenet.LeNetConfig;

/// LeNet forward pass benchmark.
fn makeForwardBench(comptime batch_size: usize) fn (*ziterion.State) void {
    return struct {
        fn benchmark(state: *ziterion.State) void {
            const allocator = state.scratchAllocator();

            var model = LeNet.init(allocator, .{ .batch_size = batch_size }, null) catch return;
            defer model.deinit();

            // Initialize weights
            var prng = std.Random.DefaultPrng.init(42);
            model.weights.initKaiming(prng.random());

            // Create input
            var input: [batch_size * 28 * 28]f32 = undefined;
            for (&input, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 256)) / 255.0;

            state.setItemsProcessed(batch_size); // images/iteration

            _ = model.forward(&input, batch_size);
            ziterion.doNotOptimize(model.cache.fc3_out);
        }
    }.benchmark;
}

/// LeNet training step benchmark (forward + backward).
fn makeTrainBench(comptime batch_size: usize) fn (*ziterion.State) void {
    return struct {
        fn benchmark(state: *ziterion.State) void {
            const allocator = state.scratchAllocator();

            var model = LeNet.init(allocator, .{ .batch_size = batch_size }, null) catch return;
            defer model.deinit();

            var prng = std.Random.DefaultPrng.init(42);
            model.weights.initKaiming(prng.random());

            var input: [batch_size * 28 * 28]f32 = undefined;
            for (&input, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 256)) / 255.0;

            var labels: [batch_size]u32 = undefined;
            for (&labels, 0..) |*v, i| v.* = @truncate(i % 10);

            state.setItemsProcessed(batch_size);

            // Forward
            _ = model.forward(&input, batch_size);

            // Compute loss gradient
            model.computeLossGradient(&labels, batch_size);

            // Backward
            model.backward(&input, batch_size) catch return;

            ziterion.doNotOptimize(model.grads.conv1_weight);
        }
    }.benchmark;
}

pub const benchmarks = struct {
    // Single image inference
    pub const lenet_forward_b1 = makeForwardBench(1);

    // Batch inference
    pub const lenet_forward_b16 = makeForwardBench(16);
    pub const lenet_forward_b64 = makeForwardBench(64);

    // Training steps
    pub const lenet_train_b16 = makeTrainBench(16);
    pub const lenet_train_b64 = makeTrainBench(64);
};
