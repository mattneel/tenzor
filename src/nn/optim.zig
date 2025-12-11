//! Optimizers for neural network training.
//!
//! Provides SGD with momentum and other optimizers.

const std = @import("std");

/// Stochastic Gradient Descent with momentum.
///
/// Update rule:
///   v = momentum * v - learning_rate * grad
///   param = param + v
///
/// With weight decay (L2 regularization):
///   grad = grad + weight_decay * param
pub const SGD = struct {
    learning_rate: f32,
    momentum: f32,
    weight_decay: f32,
    allocator: std.mem.Allocator,

    /// Velocity buffers for momentum (one per parameter group)
    velocities: std.ArrayList([]f32),

    pub fn init(allocator: std.mem.Allocator, learning_rate: f32, momentum: f32, weight_decay: f32) SGD {
        return .{
            .learning_rate = learning_rate,
            .momentum = momentum,
            .weight_decay = weight_decay,
            .allocator = allocator,
            .velocities = .empty,
        };
    }

    pub fn deinit(self: *SGD) void {
        for (self.velocities.items) |v| {
            self.allocator.free(v);
        }
        self.velocities.deinit(self.allocator);
    }

    /// Register a parameter tensor and allocate its velocity buffer.
    /// Returns the parameter index for use in step().
    pub fn addParam(self: *SGD, param_size: usize) !usize {
        const velocity = try self.allocator.alloc(f32, param_size);
        @memset(velocity, 0);
        try self.velocities.append(self.allocator, velocity);
        return self.velocities.items.len - 1;
    }

    /// Update parameters using gradient.
    pub fn step(self: *SGD, param_idx: usize, params: []f32, grads: []const f32) void {
        const velocity = self.velocities.items[param_idx];
        const lr = self.learning_rate;
        const mom = self.momentum;
        const wd = self.weight_decay;

        for (params, grads, velocity) |*p, g, *v| {
            // Weight decay: add L2 penalty to gradient
            var grad = g;
            if (wd != 0) {
                grad += wd * p.*;
            }

            // Momentum update
            v.* = mom * v.* - lr * grad;
            p.* += v.*;
        }
    }

    /// Update without momentum (simpler, for testing).
    pub fn stepSimple(self: *SGD, params: []f32, grads: []const f32) void {
        const lr = self.learning_rate;
        const wd = self.weight_decay;

        for (params, grads) |*p, g| {
            var grad = g;
            if (wd != 0) {
                grad += wd * p.*;
            }
            p.* -= lr * grad;
        }
    }
};

/// Adam optimizer (Adaptive Moment Estimation).
///
/// Update rule:
///   m = beta1 * m + (1 - beta1) * grad
///   v = beta2 * v + (1 - beta2) * grad^2
///   m_hat = m / (1 - beta1^t)
///   v_hat = v / (1 - beta2^t)
///   param = param - lr * m_hat / (sqrt(v_hat) + eps)
pub const Adam = struct {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    allocator: std.mem.Allocator,

    /// First moment (mean) estimates
    m: std.ArrayList([]f32),
    /// Second moment (variance) estimates
    v: std.ArrayList([]f32),
    /// Timestep (for bias correction)
    t: usize,

    pub fn init(
        allocator: std.mem.Allocator,
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
    ) Adam {
        return .{
            .learning_rate = learning_rate,
            .beta1 = beta1,
            .beta2 = beta2,
            .eps = eps,
            .weight_decay = weight_decay,
            .allocator = allocator,
            .m = .empty,
            .v = .empty,
            .t = 0,
        };
    }

    pub fn initDefault(allocator: std.mem.Allocator, learning_rate: f32) Adam {
        return init(allocator, learning_rate, 0.9, 0.999, 1e-8, 0);
    }

    pub fn deinit(self: *Adam) void {
        for (self.m.items) |m| self.allocator.free(m);
        for (self.v.items) |v| self.allocator.free(v);
        self.m.deinit(self.allocator);
        self.v.deinit(self.allocator);
    }

    /// Register a parameter tensor.
    pub fn addParam(self: *Adam, param_size: usize) !usize {
        const m = try self.allocator.alloc(f32, param_size);
        @memset(m, 0);
        const v = try self.allocator.alloc(f32, param_size);
        @memset(v, 0);
        try self.m.append(self.allocator, m);
        try self.v.append(self.allocator, v);
        return self.m.items.len - 1;
    }

    /// Update parameters using gradient.
    pub fn step(self: *Adam, param_idx: usize, params: []f32, grads: []const f32) void {
        self.t += 1;

        const m = self.m.items[param_idx];
        const v = self.v.items[param_idx];
        const lr = self.learning_rate;
        const b1 = self.beta1;
        const b2 = self.beta2;
        const eps = self.eps;
        const wd = self.weight_decay;

        // Bias correction terms
        const t_f: f32 = @floatFromInt(self.t);
        const bc1 = 1.0 - std.math.pow(f32, b1, t_f);
        const bc2 = 1.0 - std.math.pow(f32, b2, t_f);

        for (params, grads, m, v) |*p, g, *m_i, *v_i| {
            var grad = g;
            if (wd != 0) {
                grad += wd * p.*;
            }

            // Update biased moments
            m_i.* = b1 * m_i.* + (1.0 - b1) * grad;
            v_i.* = b2 * v_i.* + (1.0 - b2) * grad * grad;

            // Bias-corrected moments
            const m_hat = m_i.* / bc1;
            const v_hat = v_i.* / bc2;

            // Update parameter
            p.* -= lr * m_hat / (@sqrt(v_hat) + eps);
        }
    }
};

/// Zero out a gradient buffer.
pub fn zeroGrad(grad: []f32) void {
    @memset(grad, 0);
}

// ============================================================================
// Tests
// ============================================================================

test "sgd basic update" {
    var sgd = SGD.init(std.testing.allocator, 0.1, 0, 0);
    defer sgd.deinit();

    var params = [_]f32{ 1.0, 2.0, 3.0 };
    const grads = [_]f32{ 1.0, 1.0, 1.0 };

    // Simple update: params -= lr * grads
    sgd.stepSimple(&params, &grads);

    try std.testing.expectApproxEqAbs(@as(f32, 0.9), params[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.9), params[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.9), params[2], 1e-6);
}

test "sgd with momentum" {
    var sgd = SGD.init(std.testing.allocator, 0.1, 0.9, 0);
    defer sgd.deinit();

    _ = try sgd.addParam(3);

    var params = [_]f32{ 1.0, 2.0, 3.0 };
    const grads = [_]f32{ 1.0, 1.0, 1.0 };

    // First step: v = -0.1 * grad = -0.1, params += v
    sgd.step(0, &params, &grads);

    try std.testing.expectApproxEqAbs(@as(f32, 0.9), params[0], 1e-6);

    // Second step: v = 0.9 * (-0.1) - 0.1 * 1 = -0.19
    // params = 0.9 - 0.19 = 0.71
    sgd.step(0, &params, &grads);

    try std.testing.expectApproxEqAbs(@as(f32, 0.71), params[0], 1e-6);
}

test "sgd with weight decay" {
    var sgd = SGD.init(std.testing.allocator, 0.1, 0, 0.01);
    defer sgd.deinit();

    var params = [_]f32{10.0};
    const grads = [_]f32{0.0};

    // With zero gradient, only weight decay applies
    // grad = 0 + 0.01 * 10 = 0.1
    // params = 10 - 0.1 * 0.1 = 9.99
    sgd.stepSimple(&params, &grads);

    try std.testing.expectApproxEqAbs(@as(f32, 9.99), params[0], 1e-6);
}

test "adam basic update" {
    var adam = Adam.initDefault(std.testing.allocator, 0.001);
    defer adam.deinit();

    _ = try adam.addParam(3);

    var params = [_]f32{ 1.0, 2.0, 3.0 };
    const grads = [_]f32{ 0.1, 0.1, 0.1 };

    // Several steps
    for (0..10) |_| {
        adam.step(0, &params, &grads);
    }

    // Params should have decreased
    try std.testing.expect(params[0] < 1.0);
    try std.testing.expect(params[1] < 2.0);
    try std.testing.expect(params[2] < 3.0);
}

test "zero grad" {
    var grad = [_]f32{ 1.0, 2.0, 3.0 };
    zeroGrad(&grad);

    for (grad) |v| {
        try std.testing.expectEqual(@as(f32, 0), v);
    }
}
