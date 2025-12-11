//! Learning rate schedulers.
//!
//! Adjusts learning rate during training using various schedules:
//! - Constant: fixed LR
//! - Step decay: reduce LR by factor every N epochs
//! - Cosine annealing: smooth cosine decay
//! - Linear warmup: gradual increase at start
//!
//! Example:
//!   var scheduler = Scheduler.cosine(0.01, 0.0001, 10000);
//!   for (0..10000) |step| {
//!       const lr = scheduler.getLR(step);
//!       optimizer.setLR(lr);
//!       // ... training step ...
//!   }

const std = @import("std");

/// Learning rate scheduler
pub const Scheduler = union(enum) {
    constant: ConstantScheduler,
    step: StepScheduler,
    cosine: CosineScheduler,
    warmup: WarmupScheduler,
    warmup_cosine: WarmupCosineScheduler,

    /// Create a constant LR scheduler
    pub fn initConstant(lr: f32) Scheduler {
        return .{ .constant = ConstantScheduler.init(lr) };
    }

    /// Create a step decay scheduler
    pub fn initStep(initial_lr: f32, gamma: f32, step_size: u64) Scheduler {
        return .{ .step = StepScheduler.init(initial_lr, gamma, step_size) };
    }

    /// Create a cosine annealing scheduler
    pub fn initCosine(initial_lr: f32, min_lr: f32, total_steps: u64) Scheduler {
        return .{ .cosine = CosineScheduler.init(initial_lr, min_lr, total_steps) };
    }

    /// Create a linear warmup scheduler
    pub fn initWarmup(target_lr: f32, warmup_steps: u64) Scheduler {
        return .{ .warmup = WarmupScheduler.init(target_lr, warmup_steps) };
    }

    /// Create a warmup + cosine decay scheduler
    pub fn initWarmupCosine(peak_lr: f32, min_lr: f32, warmup_steps: u64, total_steps: u64) Scheduler {
        return .{ .warmup_cosine = WarmupCosineScheduler.init(peak_lr, min_lr, warmup_steps, total_steps) };
    }

    /// Get learning rate for current step
    pub fn getLR(self: *const Scheduler, current_step: u64) f32 {
        return switch (self.*) {
            .constant => |s| s.getLR(),
            .step => |s| s.getLR(current_step),
            .cosine => |s| s.getLR(current_step),
            .warmup => |s| s.getLR(current_step),
            .warmup_cosine => |s| s.getLR(current_step),
        };
    }

    /// Get the initial/base learning rate
    pub fn getBaseLR(self: *const Scheduler) f32 {
        return switch (self.*) {
            .constant => |s| s.lr,
            .step => |s| s.initial_lr,
            .cosine => |s| s.initial_lr,
            .warmup => |s| s.target_lr,
            .warmup_cosine => |s| s.peak_lr,
        };
    }
};

/// Constant learning rate
pub const ConstantScheduler = struct {
    lr: f32,

    pub fn init(lr: f32) ConstantScheduler {
        return .{ .lr = lr };
    }

    pub fn getLR(self: *const ConstantScheduler) f32 {
        return self.lr;
    }
};

/// Step decay: reduce LR by factor every step_size steps
pub const StepScheduler = struct {
    initial_lr: f32,
    gamma: f32, // decay factor (e.g., 0.1 means LR *= 0.1)
    step_size: u64, // steps between decays

    pub fn init(initial_lr: f32, gamma: f32, step_size: u64) StepScheduler {
        return .{
            .initial_lr = initial_lr,
            .gamma = gamma,
            .step_size = step_size,
        };
    }

    pub fn getLR(self: *const StepScheduler, current_step: u64) f32 {
        const num_decays = current_step / self.step_size;
        return self.initial_lr * std.math.pow(f32, self.gamma, @floatFromInt(num_decays));
    }
};

/// Cosine annealing: smooth decay from initial_lr to min_lr
pub const CosineScheduler = struct {
    initial_lr: f32,
    min_lr: f32,
    total_steps: u64,

    pub fn init(initial_lr: f32, min_lr: f32, total_steps: u64) CosineScheduler {
        return .{
            .initial_lr = initial_lr,
            .min_lr = min_lr,
            .total_steps = total_steps,
        };
    }

    pub fn getLR(self: *const CosineScheduler, current_step: u64) f32 {
        if (current_step >= self.total_steps) {
            return self.min_lr;
        }

        const progress = @as(f32, @floatFromInt(current_step)) / @as(f32, @floatFromInt(self.total_steps));
        const cosine_decay = 0.5 * (1.0 + @cos(std.math.pi * progress));
        return self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay;
    }
};

/// Linear warmup: gradual increase from 0 to target_lr
pub const WarmupScheduler = struct {
    target_lr: f32,
    warmup_steps: u64,

    pub fn init(target_lr: f32, warmup_steps: u64) WarmupScheduler {
        return .{
            .target_lr = target_lr,
            .warmup_steps = warmup_steps,
        };
    }

    pub fn getLR(self: *const WarmupScheduler, current_step: u64) f32 {
        if (current_step >= self.warmup_steps) {
            return self.target_lr;
        }

        const progress = @as(f32, @floatFromInt(current_step)) / @as(f32, @floatFromInt(self.warmup_steps));
        return self.target_lr * progress;
    }
};

/// Warmup + cosine decay: linear warmup then cosine annealing
pub const WarmupCosineScheduler = struct {
    peak_lr: f32,
    min_lr: f32,
    warmup_steps: u64,
    total_steps: u64,

    pub fn init(peak_lr: f32, min_lr: f32, warmup_steps: u64, total_steps: u64) WarmupCosineScheduler {
        return .{
            .peak_lr = peak_lr,
            .min_lr = min_lr,
            .warmup_steps = warmup_steps,
            .total_steps = total_steps,
        };
    }

    pub fn getLR(self: *const WarmupCosineScheduler, current_step: u64) f32 {
        // Warmup phase
        if (current_step < self.warmup_steps) {
            const progress = @as(f32, @floatFromInt(current_step)) / @as(f32, @floatFromInt(self.warmup_steps));
            return self.peak_lr * progress;
        }

        // Cosine decay phase
        if (current_step >= self.total_steps) {
            return self.min_lr;
        }

        const decay_steps = self.total_steps - self.warmup_steps;
        const current_decay_step = current_step - self.warmup_steps;
        const progress = @as(f32, @floatFromInt(current_decay_step)) / @as(f32, @floatFromInt(decay_steps));
        const cosine_decay = 0.5 * (1.0 + @cos(std.math.pi * progress));
        return self.min_lr + (self.peak_lr - self.min_lr) * cosine_decay;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "constant scheduler" {
    const scheduler = Scheduler.initConstant(0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0.01), scheduler.getLR(0), 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.01), scheduler.getLR(1000), 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.01), scheduler.getLR(100000), 0.0001);
}

test "step scheduler" {
    const scheduler = Scheduler.initStep(0.1, 0.1, 100);

    // Initial LR
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), scheduler.getLR(0), 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), scheduler.getLR(99), 0.0001);

    // After first decay
    try std.testing.expectApproxEqAbs(@as(f32, 0.01), scheduler.getLR(100), 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.01), scheduler.getLR(199), 0.0001);

    // After second decay
    try std.testing.expectApproxEqAbs(@as(f32, 0.001), scheduler.getLR(200), 0.0001);
}

test "cosine scheduler" {
    const scheduler = Scheduler.initCosine(0.1, 0.001, 100);

    // Start: should be close to initial_lr
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), scheduler.getLR(0), 0.001);

    // Middle: somewhere between initial and min
    const mid_lr = scheduler.getLR(50);
    try std.testing.expect(mid_lr > 0.001 and mid_lr < 0.1);

    // End: should be close to min_lr
    try std.testing.expectApproxEqAbs(@as(f32, 0.001), scheduler.getLR(100), 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.001), scheduler.getLR(200), 0.001);
}

test "warmup scheduler" {
    const scheduler = Scheduler.initWarmup(0.1, 100);

    // Start: should be 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), scheduler.getLR(0), 0.001);

    // Middle: half of target
    try std.testing.expectApproxEqAbs(@as(f32, 0.05), scheduler.getLR(50), 0.001);

    // End of warmup: full target
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), scheduler.getLR(100), 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), scheduler.getLR(200), 0.001);
}

test "warmup + cosine scheduler" {
    const scheduler = Scheduler.initWarmupCosine(0.1, 0.001, 100, 1000);

    // Warmup phase
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), scheduler.getLR(0), 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.05), scheduler.getLR(50), 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), scheduler.getLR(100), 0.001);

    // Cosine decay phase
    const mid_lr = scheduler.getLR(550);
    try std.testing.expect(mid_lr > 0.001 and mid_lr < 0.1);

    // End
    try std.testing.expectApproxEqAbs(@as(f32, 0.001), scheduler.getLR(1000), 0.001);
}

test "get base LR" {
    const c = Scheduler.initConstant(0.05);
    try std.testing.expectApproxEqAbs(@as(f32, 0.05), c.getBaseLR(), 0.001);

    const s = Scheduler.initStep(0.1, 0.1, 100);
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), s.getBaseLR(), 0.001);
}
