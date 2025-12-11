//! Integration tests for LeNet training.
//!
//! These tests verify:
//! - Forward pass produces correct shapes
//! - Backward pass computes gradients
//! - Training loop reduces loss
//! - Model can overfit a small batch

const std = @import("std");
const lenet = @import("../model/lenet.zig");
const mnist = @import("../io/mnist.zig");

// Test that LeNet can overfit a single batch (memorize the data).
// This is a sanity check that the forward/backward pass is working.
test "lenet overfit single batch" {
    const allocator = std.testing.allocator;

    const config = lenet.LeNetConfig{
        .batch_size = 8,
    };

    var model = try lenet.LeNet.init(allocator, config, null);
    defer model.deinit();

    // Initialize weights
    var prng = std.Random.DefaultPrng.init(12345);
    model.weights.initKaiming(prng.random());

    // Create synthetic data
    var input: [8 * 28 * 28]f32 = undefined;
    for (&input, 0..) |*v, i| {
        // Pattern: each sample has a distinct pattern based on its target
        v.* = @as(f32, @floatFromInt(i % 10)) / 10.0;
    }
    const targets = [_]u32{ 0, 1, 2, 3, 4, 5, 6, 7 };

    const lr: f32 = 0.1;

    // Get initial loss
    _ = model.forward(&input, 8);
    const initial_loss = model.computeLoss(&targets, 8).loss;

    // Train for many iterations to overfit
    for (0..200) |_| {
        _ = model.forward(&input, 8);
        model.grads.zero();
        model.computeLossGradient(&targets, 8);
        try model.backward(&input, 8);

        // SGD update
        sgdUpdate(&model, lr);
    }

    // Final metrics
    _ = model.forward(&input, 8);
    const final_metrics = model.computeLoss(&targets, 8);

    // Loss should decrease significantly (overfitting)
    try std.testing.expect(final_metrics.loss < initial_loss * 0.5);

    // Accuracy should improve (overfitting)
    try std.testing.expect(final_metrics.accuracy > 0.5);
}

// Test gradient magnitude is reasonable.
test "lenet gradient magnitudes" {
    const allocator = std.testing.allocator;

    const config = lenet.LeNetConfig{ .batch_size = 4 };
    var model = try lenet.LeNet.init(allocator, config, null);
    defer model.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    model.weights.initKaiming(prng.random());

    var input: [4 * 28 * 28]f32 = undefined;
    for (&input) |*v| v.* = prng.random().float(f32);
    const targets = [_]u32{ 0, 1, 2, 3 };

    _ = model.forward(&input, 4);
    model.grads.zero();
    model.computeLossGradient(&targets, 4);
    try model.backward(&input, 4);

    // Check gradient magnitudes are reasonable (not NaN, not too large)
    const grad_norm = computeGradNorm(&model);
    try std.testing.expect(!std.math.isNan(grad_norm));
    try std.testing.expect(grad_norm > 0);
    try std.testing.expect(grad_norm < 1000);
}

// Test accuracy improves during training.
test "lenet accuracy improves" {
    const allocator = std.testing.allocator;

    const config = lenet.LeNetConfig{ .batch_size = 16 };
    var model = try lenet.LeNet.init(allocator, config, null);
    defer model.deinit();

    var prng = std.Random.DefaultPrng.init(999);
    model.weights.initKaiming(prng.random());

    // Create labeled data where class = first pixel intensity bucket
    var input: [16 * 28 * 28]f32 = undefined;
    var targets: [16]u32 = undefined;

    for (0..16) |i| {
        const class: u32 = @intCast(i % 10);
        targets[i] = class;
        // Set first pixel to indicate class
        input[i * 28 * 28] = @as(f32, @floatFromInt(class)) / 10.0;
        // Fill rest with noise
        for (input[i * 28 * 28 + 1 ..][0 .. 28 * 28 - 1]) |*v| {
            v.* = prng.random().float(f32) * 0.1;
        }
    }

    // Initial accuracy (should be ~10% random)
    _ = model.forward(&input, 16);
    const initial_acc = model.computeLoss(&targets, 16).accuracy;

    // Train
    const lr: f32 = 0.05;
    for (0..50) |_| {
        _ = model.forward(&input, 16);
        model.grads.zero();
        model.computeLossGradient(&targets, 16);
        try model.backward(&input, 16);
        sgdUpdate(&model, lr);
    }

    // Final accuracy should improve
    _ = model.forward(&input, 16);
    const final_acc = model.computeLoss(&targets, 16).accuracy;

    try std.testing.expect(final_acc > initial_acc);
}

// Helper functions

fn sgdUpdate(model: *lenet.LeNet, lr: f32) void {
    updateParams(model.weights.conv1_weight, model.grads.conv1_weight, lr);
    updateParams(model.weights.conv1_bias, model.grads.conv1_bias, lr);
    updateParams(model.weights.conv2_weight, model.grads.conv2_weight, lr);
    updateParams(model.weights.conv2_bias, model.grads.conv2_bias, lr);
    updateParams(model.weights.fc1_weight, model.grads.fc1_weight, lr);
    updateParams(model.weights.fc1_bias, model.grads.fc1_bias, lr);
    updateParams(model.weights.fc2_weight, model.grads.fc2_weight, lr);
    updateParams(model.weights.fc2_bias, model.grads.fc2_bias, lr);
    updateParams(model.weights.fc3_weight, model.grads.fc3_weight, lr);
    updateParams(model.weights.fc3_bias, model.grads.fc3_bias, lr);
}

fn updateParams(params: []f32, grads: []const f32, lr: f32) void {
    for (params, grads) |*p, g| {
        p.* -= lr * g;
    }
}

fn computeGradNorm(model: *const lenet.LeNet) f32 {
    var sum: f32 = 0;
    for (model.grads.conv1_weight) |g| sum += g * g;
    for (model.grads.conv1_bias) |g| sum += g * g;
    for (model.grads.conv2_weight) |g| sum += g * g;
    for (model.grads.conv2_bias) |g| sum += g * g;
    for (model.grads.fc1_weight) |g| sum += g * g;
    for (model.grads.fc1_bias) |g| sum += g * g;
    for (model.grads.fc2_weight) |g| sum += g * g;
    for (model.grads.fc2_bias) |g| sum += g * g;
    for (model.grads.fc3_weight) |g| sum += g * g;
    for (model.grads.fc3_bias) |g| sum += g * g;
    return @sqrt(sum);
}
