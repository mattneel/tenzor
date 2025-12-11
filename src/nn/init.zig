//! Weight initialization utilities for neural networks.
//!
//! Proper initialization is critical for training deep networks.
//! These functions follow PyTorch conventions.

const std = @import("std");

/// Xavier (Glorot) uniform initialization.
///
/// Weights are drawn from U(-a, a) where a = sqrt(6 / (fan_in + fan_out)).
/// Good for tanh and sigmoid activations.
pub fn xavierUniform(weights: []f32, fan_in: usize, fan_out: usize, rng: std.Random) void {
    const a: f32 = @sqrt(6.0 / @as(f32, @floatFromInt(fan_in + fan_out)));
    for (weights) |*w| {
        w.* = rng.float(f32) * 2.0 * a - a;
    }
}

/// Xavier (Glorot) normal initialization.
///
/// Weights are drawn from N(0, std) where std = sqrt(2 / (fan_in + fan_out)).
pub fn xavierNormal(weights: []f32, fan_in: usize, fan_out: usize, rng: std.Random) void {
    const std_dev: f32 = @sqrt(2.0 / @as(f32, @floatFromInt(fan_in + fan_out)));
    for (weights) |*w| {
        w.* = randomNormal(rng) * std_dev;
    }
}

/// Kaiming (He) uniform initialization.
///
/// Weights are drawn from U(-a, a) where a = sqrt(6 / fan_in).
/// Good for ReLU activations.
pub fn kaimingUniform(weights: []f32, fan_in: usize, rng: std.Random) void {
    const a: f32 = @sqrt(6.0 / @as(f32, @floatFromInt(fan_in)));
    for (weights) |*w| {
        w.* = rng.float(f32) * 2.0 * a - a;
    }
}

/// Kaiming (He) normal initialization.
///
/// Weights are drawn from N(0, std) where std = sqrt(2 / fan_in).
pub fn kaimingNormal(weights: []f32, fan_in: usize, rng: std.Random) void {
    const std_dev: f32 = @sqrt(2.0 / @as(f32, @floatFromInt(fan_in)));
    for (weights) |*w| {
        w.* = randomNormal(rng) * std_dev;
    }
}

/// Uniform initialization in range [-bound, bound].
pub fn uniform(weights: []f32, bound: f32, rng: std.Random) void {
    for (weights) |*w| {
        w.* = rng.float(f32) * 2.0 * bound - bound;
    }
}

/// Normal initialization with given mean and std.
pub fn normal(weights: []f32, mean: f32, std_dev: f32, rng: std.Random) void {
    for (weights) |*w| {
        w.* = randomNormal(rng) * std_dev + mean;
    }
}

/// Initialize to zeros.
pub fn zeros(weights: []f32) void {
    @memset(weights, 0);
}

/// Initialize to ones.
pub fn ones(weights: []f32) void {
    for (weights) |*w| w.* = 1.0;
}

/// Initialize to a constant value.
pub fn constant(weights: []f32, value: f32) void {
    for (weights) |*w| w.* = value;
}

/// Calculate fan_in and fan_out for a convolutional layer.
///
/// Weight shape: [out_channels, kernel_h, kernel_w, in_channels]
pub fn calcConvFan(out_channels: usize, kernel_h: usize, kernel_w: usize, in_channels: usize) struct { fan_in: usize, fan_out: usize } {
    const receptive_field = kernel_h * kernel_w;
    return .{
        .fan_in = in_channels * receptive_field,
        .fan_out = out_channels * receptive_field,
    };
}

/// Calculate fan_in and fan_out for a linear layer.
///
/// Weight shape: [out_features, in_features]
pub fn calcLinearFan(out_features: usize, in_features: usize) struct { fan_in: usize, fan_out: usize } {
    return .{
        .fan_in = in_features,
        .fan_out = out_features,
    };
}

/// Generate a random number from standard normal distribution N(0, 1).
/// Uses Box-Muller transform.
fn randomNormal(rng: std.Random) f32 {
    const rand1 = rng.float(f32);
    const rand2 = rng.float(f32);

    // Box-Muller transform
    // Avoid log(0) by clamping
    const safe_rand1 = @max(rand1, 1e-10);
    const r = @sqrt(-2.0 * @log(safe_rand1));
    const theta = 2.0 * std.math.pi * rand2;

    return r * @cos(theta);
}

// ============================================================================
// Tests
// ============================================================================

test "xavier uniform bounds" {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    var weights: [100]f32 = undefined;
    xavierUniform(&weights, 64, 128, rng);

    const a: f32 = @sqrt(6.0 / @as(f32, 64 + 128));
    for (weights) |w| {
        try std.testing.expect(w >= -a and w <= a);
    }
}

test "kaiming uniform bounds" {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    var weights: [100]f32 = undefined;
    kaimingUniform(&weights, 64, rng);

    const a: f32 = @sqrt(6.0 / 64.0);
    for (weights) |w| {
        try std.testing.expect(w >= -a and w <= a);
    }
}

test "normal distribution statistics" {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    var weights: [10000]f32 = undefined;
    normal(&weights, 0, 1, rng);

    // Check mean is approximately 0
    var sum: f32 = 0;
    for (weights) |w| sum += w;
    const mean = sum / @as(f32, weights.len);
    try std.testing.expect(@abs(mean) < 0.1);

    // Check std is approximately 1
    var sq_sum: f32 = 0;
    for (weights) |w| sq_sum += (w - mean) * (w - mean);
    const variance = sq_sum / @as(f32, weights.len);
    try std.testing.expect(@abs(@sqrt(variance) - 1.0) < 0.1);
}

test "zeros initialization" {
    var weights = [_]f32{ 1.0, 2.0, 3.0 };
    zeros(&weights);

    for (weights) |w| {
        try std.testing.expectEqual(@as(f32, 0), w);
    }
}

test "conv fan calculation" {
    const fan = calcConvFan(16, 5, 5, 6);
    // fan_in = 6 * 5 * 5 = 150
    // fan_out = 16 * 5 * 5 = 400
    try std.testing.expectEqual(@as(usize, 150), fan.fan_in);
    try std.testing.expectEqual(@as(usize, 400), fan.fan_out);
}

test "linear fan calculation" {
    const fan = calcLinearFan(120, 256);
    try std.testing.expectEqual(@as(usize, 256), fan.fan_in);
    try std.testing.expectEqual(@as(usize, 120), fan.fan_out);
}
