//! LeNet-5 CNN for MNIST with manual gradient computation.
//!
//! Architecture:
//! Input:    28x28x1 (NHWC)
//! Conv1:    6 filters, 5x5, valid -> 24x24x6, ReLU
//! MaxPool:  2x2, stride 2 -> 12x12x6
//! Conv2:    16 filters, 5x5, valid -> 8x8x16, ReLU
//! MaxPool:  2x2, stride 2 -> 4x4x16
//! Flatten:  256
//! FC1:      256 -> 120, ReLU
//! FC2:      120 -> 84, ReLU
//! FC3:      84 -> 10
//!
//! All intermediate activations are stored for backward pass.

const std = @import("std");
const conv2d = @import("../backend/cpu/kernels/conv2d.zig");
const maxpool = @import("../backend/cpu/kernels/maxpool.zig");
const matmul = @import("../backend/cpu/kernels/matmul.zig");
const loss_mod = @import("../nn/loss.zig");
const init_mod = @import("../nn/init.zig");

pub const LeNetConfig = struct {
    batch_size: usize = 64,
    input_h: usize = 28,
    input_w: usize = 28,
    input_c: usize = 1,
    num_classes: usize = 10,
};

/// LeNet model weights.
pub const LeNetWeights = struct {
    // Conv1: [6, 5, 5, 1] = 150 params
    conv1_weight: []f32,
    conv1_bias: []f32, // [6]

    // Conv2: [16, 5, 5, 6] = 2400 params
    conv2_weight: []f32,
    conv2_bias: []f32, // [16]

    // FC1: [120, 256] = 30720 params
    fc1_weight: []f32,
    fc1_bias: []f32, // [120]

    // FC2: [84, 120] = 10080 params
    fc2_weight: []f32,
    fc2_bias: []f32, // [84]

    // FC3: [10, 84] = 840 params
    fc3_weight: []f32,
    fc3_bias: []f32, // [10]

    pub fn init(allocator: std.mem.Allocator) !LeNetWeights {
        return .{
            .conv1_weight = try allocator.alloc(f32, 6 * 5 * 5 * 1),
            .conv1_bias = try allocator.alloc(f32, 6),
            .conv2_weight = try allocator.alloc(f32, 16 * 5 * 5 * 6),
            .conv2_bias = try allocator.alloc(f32, 16),
            .fc1_weight = try allocator.alloc(f32, 120 * 256),
            .fc1_bias = try allocator.alloc(f32, 120),
            .fc2_weight = try allocator.alloc(f32, 84 * 120),
            .fc2_bias = try allocator.alloc(f32, 84),
            .fc3_weight = try allocator.alloc(f32, 10 * 84),
            .fc3_bias = try allocator.alloc(f32, 10),
        };
    }

    pub fn deinit(self: *LeNetWeights, allocator: std.mem.Allocator) void {
        allocator.free(self.conv1_weight);
        allocator.free(self.conv1_bias);
        allocator.free(self.conv2_weight);
        allocator.free(self.conv2_bias);
        allocator.free(self.fc1_weight);
        allocator.free(self.fc1_bias);
        allocator.free(self.fc2_weight);
        allocator.free(self.fc2_bias);
        allocator.free(self.fc3_weight);
        allocator.free(self.fc3_bias);
    }

    /// Initialize weights using Kaiming for ReLU layers.
    pub fn initKaiming(self: *LeNetWeights, rng: std.Random) void {
        // Conv1: fan_in = 1 * 5 * 5 = 25
        init_mod.kaimingUniform(self.conv1_weight, 25, rng);
        init_mod.zeros(self.conv1_bias);

        // Conv2: fan_in = 6 * 5 * 5 = 150
        init_mod.kaimingUniform(self.conv2_weight, 150, rng);
        init_mod.zeros(self.conv2_bias);

        // FC1: fan_in = 256
        init_mod.kaimingUniform(self.fc1_weight, 256, rng);
        init_mod.zeros(self.fc1_bias);

        // FC2: fan_in = 120
        init_mod.kaimingUniform(self.fc2_weight, 120, rng);
        init_mod.zeros(self.fc2_bias);

        // FC3: fan_in = 84
        init_mod.kaimingUniform(self.fc3_weight, 84, rng);
        init_mod.zeros(self.fc3_bias);
    }
};

/// Gradient storage (same shapes as weights).
pub const LeNetGrads = struct {
    conv1_weight: []f32,
    conv1_bias: []f32,
    conv2_weight: []f32,
    conv2_bias: []f32,
    fc1_weight: []f32,
    fc1_bias: []f32,
    fc2_weight: []f32,
    fc2_bias: []f32,
    fc3_weight: []f32,
    fc3_bias: []f32,

    pub fn init(allocator: std.mem.Allocator) !LeNetGrads {
        return .{
            .conv1_weight = try allocator.alloc(f32, 6 * 5 * 5 * 1),
            .conv1_bias = try allocator.alloc(f32, 6),
            .conv2_weight = try allocator.alloc(f32, 16 * 5 * 5 * 6),
            .conv2_bias = try allocator.alloc(f32, 16),
            .fc1_weight = try allocator.alloc(f32, 120 * 256),
            .fc1_bias = try allocator.alloc(f32, 120),
            .fc2_weight = try allocator.alloc(f32, 84 * 120),
            .fc2_bias = try allocator.alloc(f32, 84),
            .fc3_weight = try allocator.alloc(f32, 10 * 84),
            .fc3_bias = try allocator.alloc(f32, 10),
        };
    }

    pub fn deinit(self: *LeNetGrads, allocator: std.mem.Allocator) void {
        allocator.free(self.conv1_weight);
        allocator.free(self.conv1_bias);
        allocator.free(self.conv2_weight);
        allocator.free(self.conv2_bias);
        allocator.free(self.fc1_weight);
        allocator.free(self.fc1_bias);
        allocator.free(self.fc2_weight);
        allocator.free(self.fc2_bias);
        allocator.free(self.fc3_weight);
        allocator.free(self.fc3_bias);
    }

    pub fn zero(self: *LeNetGrads) void {
        @memset(self.conv1_weight, 0);
        @memset(self.conv1_bias, 0);
        @memset(self.conv2_weight, 0);
        @memset(self.conv2_bias, 0);
        @memset(self.fc1_weight, 0);
        @memset(self.fc1_bias, 0);
        @memset(self.fc2_weight, 0);
        @memset(self.fc2_bias, 0);
        @memset(self.fc3_weight, 0);
        @memset(self.fc3_bias, 0);
    }
};

/// Intermediate activations cache (needed for backward pass).
pub const LeNetCache = struct {
    batch_size: usize,

    // Forward pass intermediates
    conv1_out: []f32, // [N, 24, 24, 6]
    pool1_out: []f32, // [N, 12, 12, 6]
    pool1_indices: []usize, // [N, 12, 12, 6]
    conv2_out: []f32, // [N, 8, 8, 16]
    pool2_out: []f32, // [N, 4, 4, 16]
    pool2_indices: []usize, // [N, 4, 4, 16]
    fc1_pre: []f32, // [N, 120] before ReLU
    fc1_out: []f32, // [N, 120] after ReLU
    fc2_pre: []f32, // [N, 84]
    fc2_out: []f32, // [N, 84]
    fc3_out: []f32, // [N, 10] (logits)

    // Backward pass intermediates
    grad_fc3: []f32, // [N, 10]
    grad_fc2: []f32, // [N, 84]
    grad_fc1: []f32, // [N, 120]
    grad_pool2: []f32, // [N, 4, 4, 16]
    grad_conv2: []f32, // [N, 8, 8, 16]
    grad_pool1: []f32, // [N, 12, 12, 6]
    grad_conv1: []f32, // [N, 24, 24, 6]

    pub fn init(allocator: std.mem.Allocator, batch_size: usize) !LeNetCache {
        return .{
            .batch_size = batch_size,
            .conv1_out = try allocator.alloc(f32, batch_size * 24 * 24 * 6),
            .pool1_out = try allocator.alloc(f32, batch_size * 12 * 12 * 6),
            .pool1_indices = try allocator.alloc(usize, batch_size * 12 * 12 * 6),
            .conv2_out = try allocator.alloc(f32, batch_size * 8 * 8 * 16),
            .pool2_out = try allocator.alloc(f32, batch_size * 4 * 4 * 16),
            .pool2_indices = try allocator.alloc(usize, batch_size * 4 * 4 * 16),
            .fc1_pre = try allocator.alloc(f32, batch_size * 120),
            .fc1_out = try allocator.alloc(f32, batch_size * 120),
            .fc2_pre = try allocator.alloc(f32, batch_size * 84),
            .fc2_out = try allocator.alloc(f32, batch_size * 84),
            .fc3_out = try allocator.alloc(f32, batch_size * 10),
            .grad_fc3 = try allocator.alloc(f32, batch_size * 10),
            .grad_fc2 = try allocator.alloc(f32, batch_size * 84),
            .grad_fc1 = try allocator.alloc(f32, batch_size * 120),
            .grad_pool2 = try allocator.alloc(f32, batch_size * 4 * 4 * 16),
            .grad_conv2 = try allocator.alloc(f32, batch_size * 8 * 8 * 16),
            .grad_pool1 = try allocator.alloc(f32, batch_size * 12 * 12 * 6),
            .grad_conv1 = try allocator.alloc(f32, batch_size * 24 * 24 * 6),
        };
    }

    pub fn deinit(self: *LeNetCache, allocator: std.mem.Allocator) void {
        allocator.free(self.conv1_out);
        allocator.free(self.pool1_out);
        allocator.free(self.pool1_indices);
        allocator.free(self.conv2_out);
        allocator.free(self.pool2_out);
        allocator.free(self.pool2_indices);
        allocator.free(self.fc1_pre);
        allocator.free(self.fc1_out);
        allocator.free(self.fc2_pre);
        allocator.free(self.fc2_out);
        allocator.free(self.fc3_out);
        allocator.free(self.grad_fc3);
        allocator.free(self.grad_fc2);
        allocator.free(self.grad_fc1);
        allocator.free(self.grad_pool2);
        allocator.free(self.grad_conv2);
        allocator.free(self.grad_pool1);
        allocator.free(self.grad_conv1);
    }
};

/// LeNet model.
pub const LeNet = struct {
    allocator: std.mem.Allocator,
    config: LeNetConfig,
    weights: LeNetWeights,
    grads: LeNetGrads,
    cache: LeNetCache,

    pub fn init(allocator: std.mem.Allocator, config: LeNetConfig) !LeNet {
        return .{
            .allocator = allocator,
            .config = config,
            .weights = try LeNetWeights.init(allocator),
            .grads = try LeNetGrads.init(allocator),
            .cache = try LeNetCache.init(allocator, config.batch_size),
        };
    }

    pub fn deinit(self: *LeNet) void {
        self.weights.deinit(self.allocator);
        self.grads.deinit(self.allocator);
        self.cache.deinit(self.allocator);
    }

    /// Forward pass: compute predictions from input.
    /// Input: [batch_size, 28, 28, 1]
    /// Returns: logits [batch_size, 10]
    pub fn forward(self: *LeNet, input: []const f32, batch_size: usize) []const f32 {
        // Conv1: [N, 28, 28, 1] -> [N, 24, 24, 6]
        conv2d.conv2dForward(
            f32,
            input,
            self.weights.conv1_weight,
            self.weights.conv1_bias,
            self.cache.conv1_out,
            batch_size,
            28,
            28,
            1, // in_h, in_w, in_c
            6,
            5,
            5, // out_c, kh, kw
            1,
            1,
            0,
            0, // stride, pad
        );
        // ReLU in-place
        applyRelu(self.cache.conv1_out);

        // MaxPool1: [N, 24, 24, 6] -> [N, 12, 12, 6]
        maxpool.maxPool2dForward(
            f32,
            self.cache.conv1_out,
            self.cache.pool1_out,
            self.cache.pool1_indices,
            batch_size,
            24,
            24,
            6,
            2,
            2,
            2,
            2,
        );

        // Conv2: [N, 12, 12, 6] -> [N, 8, 8, 16]
        conv2d.conv2dForward(
            f32,
            self.cache.pool1_out,
            self.weights.conv2_weight,
            self.weights.conv2_bias,
            self.cache.conv2_out,
            batch_size,
            12,
            12,
            6,
            16,
            5,
            5,
            1,
            1,
            0,
            0,
        );
        applyRelu(self.cache.conv2_out);

        // MaxPool2: [N, 8, 8, 16] -> [N, 4, 4, 16]
        maxpool.maxPool2dForward(
            f32,
            self.cache.conv2_out,
            self.cache.pool2_out,
            self.cache.pool2_indices,
            batch_size,
            8,
            8,
            16,
            2,
            2,
            2,
            2,
        );

        // Flatten: [N, 4, 4, 16] is already flat in memory as [N, 256]

        // FC1: [N, 256] @ [256, 120]^T -> [N, 120]
        linearForward(
            self.cache.pool2_out,
            self.weights.fc1_weight,
            self.weights.fc1_bias,
            self.cache.fc1_pre,
            batch_size,
            256,
            120,
        );
        // ReLU with save of pre-activation
        for (self.cache.fc1_pre, self.cache.fc1_out) |pre, *out| {
            out.* = @max(pre, 0);
        }

        // FC2: [N, 120] @ [120, 84]^T -> [N, 84]
        linearForward(
            self.cache.fc1_out,
            self.weights.fc2_weight,
            self.weights.fc2_bias,
            self.cache.fc2_pre,
            batch_size,
            120,
            84,
        );
        for (self.cache.fc2_pre, self.cache.fc2_out) |pre, *out| {
            out.* = @max(pre, 0);
        }

        // FC3: [N, 84] @ [84, 10]^T -> [N, 10]
        linearForward(
            self.cache.fc2_out,
            self.weights.fc3_weight,
            self.weights.fc3_bias,
            self.cache.fc3_out,
            batch_size,
            84,
            10,
        );

        return self.cache.fc3_out;
    }

    /// Backward pass: compute gradients.
    /// Must be called after forward() and computeLossGradient().
    /// Uses self.cache.grad_fc3 which should be filled by computeLossGradient().
    pub fn backward(self: *LeNet, input: []const f32, batch_size: usize) void {
        // grad_fc3 is already filled by computeLossGradient()

        // === FC3 backward ===
        // grad_fc2_out = grad_fc3 @ fc3_weight
        // grad_fc3_weight += fc2_out^T @ grad_fc3
        // grad_fc3_bias += sum(grad_fc3, axis=0)
        linearBackward(
            self.cache.fc2_out,
            self.cache.grad_fc3,
            self.weights.fc3_weight,
            self.cache.grad_fc2,
            self.grads.fc3_weight,
            self.grads.fc3_bias,
            batch_size,
            84,
            10,
        );

        // ReLU backward for FC2
        applyReluBackward(self.cache.grad_fc2, self.cache.fc2_pre);

        // === FC2 backward ===
        linearBackward(
            self.cache.fc1_out,
            self.cache.grad_fc2,
            self.weights.fc2_weight,
            self.cache.grad_fc1,
            self.grads.fc2_weight,
            self.grads.fc2_bias,
            batch_size,
            120,
            84,
        );

        // ReLU backward for FC1
        applyReluBackward(self.cache.grad_fc1, self.cache.fc1_pre);

        // === FC1 backward ===
        // grad_pool2 = grad_fc1 @ fc1_weight
        linearBackward(
            self.cache.pool2_out,
            self.cache.grad_fc1,
            self.weights.fc1_weight,
            self.cache.grad_pool2,
            self.grads.fc1_weight,
            self.grads.fc1_bias,
            batch_size,
            256,
            120,
        );

        // === MaxPool2 backward ===
        maxpool.maxPool2dBackward(
            f32,
            self.cache.grad_pool2,
            self.cache.pool2_indices,
            self.cache.grad_conv2,
            batch_size,
            8,
            8,
            16,
            2,
            2,
            2,
            2,
        );

        // ReLU backward for Conv2
        applyReluBackward(self.cache.grad_conv2, self.cache.conv2_out);

        // === Conv2 backward ===
        conv2d.conv2dBackwardInput(
            f32,
            self.cache.grad_conv2,
            self.weights.conv2_weight,
            self.cache.grad_pool1,
            batch_size,
            12,
            12,
            6,
            16,
            5,
            5,
            1,
            1,
            0,
            0,
        );
        conv2d.conv2dBackwardWeight(
            f32,
            self.cache.pool1_out,
            self.cache.grad_conv2,
            self.grads.conv2_weight,
            self.grads.conv2_bias,
            batch_size,
            12,
            12,
            6,
            16,
            5,
            5,
            1,
            1,
            0,
            0,
        );

        // === MaxPool1 backward ===
        maxpool.maxPool2dBackward(
            f32,
            self.cache.grad_pool1,
            self.cache.pool1_indices,
            self.cache.grad_conv1,
            batch_size,
            24,
            24,
            6,
            2,
            2,
            2,
            2,
        );

        // ReLU backward for Conv1
        applyReluBackward(self.cache.grad_conv1, self.cache.conv1_out);

        // === Conv1 backward (only weight, not input) ===
        conv2d.conv2dBackwardWeight(
            f32,
            input,
            self.cache.grad_conv1,
            self.grads.conv1_weight,
            self.grads.conv1_bias,
            batch_size,
            28,
            28,
            1,
            6,
            5,
            5,
            1,
            1,
            0,
            0,
        );
    }

    /// Compute loss and accuracy.
    pub fn computeLoss(self: *LeNet, targets: []const u32, batch_size: usize) struct { loss: f32, accuracy: f32 } {
        const loss_val = loss_mod.crossEntropyLoss(f32, self.cache.fc3_out, targets, batch_size, 10);

        // Compute accuracy
        var correct: usize = 0;
        for (0..batch_size) |i| {
            const logits = self.cache.fc3_out[i * 10 ..][0..10];
            var max_idx: usize = 0;
            var max_val = logits[0];
            for (logits[1..], 1..) |v, j| {
                if (v > max_val) {
                    max_val = v;
                    max_idx = j;
                }
            }
            if (max_idx == targets[i]) correct += 1;
        }

        return .{
            .loss = loss_val,
            .accuracy = @as(f32, @floatFromInt(correct)) / @as(f32, @floatFromInt(batch_size)),
        };
    }

    /// Compute loss gradient (for backward pass).
    pub fn computeLossGradient(self: *LeNet, targets: []const u32, batch_size: usize) void {
        loss_mod.crossEntropyBackward(f32, self.cache.fc3_out, targets, self.cache.grad_fc3, batch_size, 10);
    }
};

// ============================================================================
// Helper functions
// ============================================================================

/// Apply ReLU in-place.
fn applyRelu(data: []f32) void {
    for (data) |*v| {
        v.* = @max(v.*, 0);
    }
}

/// Apply ReLU backward: grad = grad * (pre > 0).
fn applyReluBackward(grad: []f32, pre_activation: []const f32) void {
    for (grad, pre_activation) |*g, pre| {
        if (pre <= 0) g.* = 0;
    }
}

/// Linear layer forward: out = input @ weight^T + bias.
/// weight: [out_features, in_features]
fn linearForward(
    input: []const f32,
    weight: []const f32,
    bias: []const f32,
    output: []f32,
    batch_size: usize,
    in_features: usize,
    out_features: usize,
) void {
    // For each sample
    for (0..batch_size) |b| {
        const in_row = input[b * in_features ..][0..in_features];
        const out_row = output[b * out_features ..][0..out_features];

        // out = in @ W^T + bias
        for (0..out_features) |j| {
            var sum: f32 = bias[j];
            const w_row = weight[j * in_features ..][0..in_features];
            for (in_row, w_row) |in_val, w_val| {
                sum += in_val * w_val;
            }
            out_row[j] = sum;
        }
    }
}

/// Linear layer backward.
/// Computes:
///   grad_input = grad_output @ weight
///   grad_weight += input^T @ grad_output (accumulated)
///   grad_bias += sum(grad_output, axis=0) (accumulated)
fn linearBackward(
    input: []const f32,
    grad_output: []const f32,
    weight: []const f32,
    grad_input: []f32,
    grad_weight: []f32,
    grad_bias: []f32,
    batch_size: usize,
    in_features: usize,
    out_features: usize,
) void {
    // grad_input = grad_output @ weight
    for (0..batch_size) |b| {
        const grad_out_row = grad_output[b * out_features ..][0..out_features];
        const grad_in_row = grad_input[b * in_features ..][0..in_features];

        for (0..in_features) |i| {
            var sum: f32 = 0;
            for (0..out_features) |j| {
                sum += grad_out_row[j] * weight[j * in_features + i];
            }
            grad_in_row[i] = sum;
        }
    }

    // grad_weight += input^T @ grad_output
    // grad_bias += sum(grad_output)
    for (0..batch_size) |b| {
        const in_row = input[b * in_features ..][0..in_features];
        const grad_out_row = grad_output[b * out_features ..][0..out_features];

        for (0..out_features) |j| {
            grad_bias[j] += grad_out_row[j];
            for (0..in_features) |i| {
                grad_weight[j * in_features + i] += in_row[i] * grad_out_row[j];
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

test "lenet forward shape" {
    const allocator = std.testing.allocator;
    const config = LeNetConfig{ .batch_size = 2 };

    var model = try LeNet.init(allocator, config);
    defer model.deinit();

    // Random initialization
    var prng = std.Random.DefaultPrng.init(42);
    model.weights.initKaiming(prng.random());

    // Create input [2, 28, 28, 1]
    var input: [2 * 28 * 28]f32 = undefined;
    for (&input) |*v| v.* = prng.random().float(f32);

    // Forward pass
    const output = model.forward(&input, 2);

    // Check output shape
    try std.testing.expectEqual(@as(usize, 20), output.len); // 2 * 10
}

test "lenet backward runs" {
    const allocator = std.testing.allocator;
    const config = LeNetConfig{ .batch_size = 2 };

    var model = try LeNet.init(allocator, config);
    defer model.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    model.weights.initKaiming(prng.random());

    var input: [2 * 28 * 28]f32 = undefined;
    for (&input) |*v| v.* = prng.random().float(f32);

    const targets = [_]u32{ 3, 7 };

    // Forward
    _ = model.forward(&input, 2);

    // Zero grads
    model.grads.zero();

    // Compute loss gradient
    model.computeLossGradient(&targets, 2);

    // Backward
    model.backward(&input, 2);

    // Check that gradients are non-zero
    var has_nonzero: bool = false;
    for (model.grads.fc1_weight) |g| {
        if (g != 0) {
            has_nonzero = true;
            break;
        }
    }
    try std.testing.expect(has_nonzero);
}

test "lenet loss decreases" {
    const allocator = std.testing.allocator;
    const config = LeNetConfig{ .batch_size = 4 };

    var model = try LeNet.init(allocator, config);
    defer model.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    model.weights.initKaiming(prng.random());

    // Fixed input and targets
    var input: [4 * 28 * 28]f32 = undefined;
    for (&input) |*v| v.* = prng.random().float(f32);
    const targets = [_]u32{ 0, 1, 2, 3 };

    const lr: f32 = 0.01;

    // Initial loss
    _ = model.forward(&input, 4);
    const initial_metrics = model.computeLoss(&targets, 4);

    // Training loop
    for (0..10) |_| {
        _ = model.forward(&input, 4);
        model.grads.zero();
        model.computeLossGradient(&targets, 4);
        model.backward(&input, 4);

        // SGD update (inline)
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

    // Final loss
    _ = model.forward(&input, 4);
    const final_metrics = model.computeLoss(&targets, 4);

    // Loss should decrease
    try std.testing.expect(final_metrics.loss < initial_metrics.loss);
}

fn updateParams(params: []f32, grads: []const f32, lr: f32) void {
    for (params, grads) |*p, g| {
        p.* -= lr * g;
    }
}
