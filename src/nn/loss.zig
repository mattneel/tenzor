//! Loss functions for neural network training.
//!
//! Provides numerically stable forward and backward passes for common losses.

const std = @import("std");

/// Cross-entropy loss with log-softmax for numerical stability.
///
/// Forward: L = -log(softmax(logits)[target])
/// Backward: grad = softmax(logits) - one_hot(target)
///
/// The backward formula is beautifully simple because d/dx[-log(softmax(x))] = softmax(x) - 1 at target.
pub fn crossEntropyLoss(
    comptime T: type,
    logits: []const T,
    targets: []const u32,
    batch_size: usize,
    num_classes: usize,
) T {
    var total_loss: T = 0;

    for (0..batch_size) |i| {
        const sample_logits = logits[i * num_classes ..][0..num_classes];
        const target = targets[i];

        // Numerically stable log-softmax: log(softmax(x)) = x - max(x) - log(sum(exp(x - max(x))))
        // Find max for numerical stability
        var max_logit: T = sample_logits[0];
        for (sample_logits[1..]) |v| {
            max_logit = @max(max_logit, v);
        }

        // Compute log-sum-exp
        var sum_exp: T = 0;
        for (sample_logits) |v| {
            sum_exp += @exp(v - max_logit);
        }
        const log_sum_exp = @log(sum_exp);

        // Loss = -log_softmax[target] = -(logits[target] - max - log_sum_exp)
        const log_prob = sample_logits[target] - max_logit - log_sum_exp;
        total_loss -= log_prob;
    }

    return total_loss / @as(T, @floatFromInt(batch_size));
}

/// Backward pass for cross-entropy loss.
///
/// Computes grad_logits = softmax(logits) - one_hot(targets).
/// This is the gradient of -log(softmax(logits)[target]) w.r.t. logits.
pub fn crossEntropyBackward(
    comptime T: type,
    logits: []const T,
    targets: []const u32,
    grad_logits: []T,
    batch_size: usize,
    num_classes: usize,
) void {
    for (0..batch_size) |i| {
        const sample_logits = logits[i * num_classes ..][0..num_classes];
        const grad_sample = grad_logits[i * num_classes ..][0..num_classes];
        const target = targets[i];

        // Compute softmax
        var max_logit: T = sample_logits[0];
        for (sample_logits[1..]) |v| {
            max_logit = @max(max_logit, v);
        }

        var sum_exp: T = 0;
        for (sample_logits) |v| {
            sum_exp += @exp(v - max_logit);
        }

        // grad = softmax - one_hot
        // Scaled by 1/batch_size for mean loss
        const scale = 1.0 / @as(T, @floatFromInt(batch_size));
        for (sample_logits, 0..) |v, j| {
            const softmax_j = @exp(v - max_logit) / sum_exp;
            const one_hot_j: T = if (j == target) 1.0 else 0.0;
            grad_sample[j] = (softmax_j - one_hot_j) * scale;
        }
    }
}

/// Mean squared error loss.
///
/// Forward: L = mean((predictions - targets)^2)
pub fn mseLoss(
    comptime T: type,
    predictions: []const T,
    targets: []const T,
) T {
    var sum: T = 0;
    for (predictions, targets) |p, t| {
        const diff = p - t;
        sum += diff * diff;
    }
    return sum / @as(T, @floatFromInt(predictions.len));
}

/// Backward pass for MSE loss.
///
/// grad = 2 * (predictions - targets) / n
pub fn mseBackward(
    comptime T: type,
    predictions: []const T,
    targets: []const T,
    grad: []T,
) void {
    const n: T = @floatFromInt(predictions.len);
    for (predictions, targets, grad) |p, t, *g| {
        g.* = 2.0 * (p - t) / n;
    }
}

/// Softmax function (standalone, for inference).
pub fn softmax(
    comptime T: type,
    logits: []const T,
    output: []T,
    batch_size: usize,
    num_classes: usize,
) void {
    for (0..batch_size) |i| {
        const sample_logits = logits[i * num_classes ..][0..num_classes];
        const sample_output = output[i * num_classes ..][0..num_classes];

        // Find max for numerical stability
        var max_val: T = sample_logits[0];
        for (sample_logits[1..]) |v| {
            max_val = @max(max_val, v);
        }

        // Compute exp and sum
        var sum: T = 0;
        for (sample_logits, sample_output) |v, *o| {
            o.* = @exp(v - max_val);
            sum += o.*;
        }

        // Normalize
        for (sample_output) |*o| {
            o.* /= sum;
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

test "cross entropy loss basic" {
    // Logits: [2, 1, 0.1] -> softmax ≈ [0.659, 0.242, 0.099]
    // Target: 0 -> loss = -log(0.659) ≈ 0.417
    const logits = [_]f32{ 2.0, 1.0, 0.1 };
    const targets = [_]u32{0};

    const loss = crossEntropyLoss(f32, &logits, &targets, 1, 3);

    // Expected: -log(exp(2) / (exp(2) + exp(1) + exp(0.1)))
    const exp2: f32 = @exp(2.0);
    const exp1: f32 = @exp(1.0);
    const exp01: f32 = @exp(0.1);
    const expected = -@log(exp2 / (exp2 + exp1 + exp01));

    try std.testing.expectApproxEqAbs(expected, loss, 1e-5);
}

test "cross entropy backward" {
    const logits = [_]f32{ 2.0, 1.0, 0.1 };
    const targets = [_]u32{0};

    var grad: [3]f32 = undefined;
    crossEntropyBackward(f32, &logits, &targets, &grad, 1, 3);

    // grad = softmax - one_hot
    // softmax[0] ≈ 0.659, one_hot[0] = 1 -> grad[0] ≈ -0.341
    const exp2: f32 = @exp(2.0);
    const exp1: f32 = @exp(1.0);
    const exp01: f32 = @exp(0.1);
    const sum = exp2 + exp1 + exp01;

    const expected_grad0 = exp2 / sum - 1.0;
    const expected_grad1 = exp1 / sum;
    const expected_grad2 = exp01 / sum;

    try std.testing.expectApproxEqAbs(expected_grad0, grad[0], 1e-5);
    try std.testing.expectApproxEqAbs(expected_grad1, grad[1], 1e-5);
    try std.testing.expectApproxEqAbs(expected_grad2, grad[2], 1e-5);
}

test "cross entropy perfect prediction" {
    // If model predicts correctly with high confidence, loss should be low
    const logits = [_]f32{ 10.0, 0.0, 0.0 };
    const targets = [_]u32{0};

    const loss = crossEntropyLoss(f32, &logits, &targets, 1, 3);

    // softmax[0] ≈ 1.0, so loss ≈ 0
    try std.testing.expect(loss < 0.001);
}

test "cross entropy wrong prediction" {
    // If model predicts wrong class with high confidence, loss should be high
    const logits = [_]f32{ 0.0, 0.0, 10.0 };
    const targets = [_]u32{0};

    const loss = crossEntropyLoss(f32, &logits, &targets, 1, 3);

    // softmax[0] ≈ 0, so loss ≈ 10
    try std.testing.expect(loss > 9.0);
}

test "mse loss" {
    const predictions = [_]f32{ 1.0, 2.0, 3.0 };
    const targets = [_]f32{ 1.0, 2.0, 3.0 };

    const loss = mseLoss(f32, &predictions, &targets);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), loss, 1e-6);
}

test "mse loss nonzero" {
    const predictions = [_]f32{ 1.0, 2.0, 3.0 };
    const targets = [_]f32{ 2.0, 2.0, 2.0 };

    // MSE = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 = (1 + 0 + 1) / 3 = 2/3
    const loss = mseLoss(f32, &predictions, &targets);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0 / 3.0), loss, 1e-6);
}

test "softmax sums to one" {
    const logits = [_]f32{ 1.0, 2.0, 3.0 };
    var output: [3]f32 = undefined;

    softmax(f32, &logits, &output, 1, 3);

    var sum: f32 = 0;
    for (output) |v| sum += v;

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-6);
}

test "softmax largest logit wins" {
    const logits = [_]f32{ 1.0, 5.0, 2.0 };
    var output: [3]f32 = undefined;

    softmax(f32, &logits, &output, 1, 3);

    // Class 1 should have highest probability
    try std.testing.expect(output[1] > output[0]);
    try std.testing.expect(output[1] > output[2]);
}
