//! Training callbacks: early stopping, checkpointing, etc.

const std = @import("std");
const metrics_mod = @import("metrics.zig");

/// Callback interface for training events
pub const Callback = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        onBatchEnd: ?*const fn (*anyopaque, *TrainerState) void = null,
        onEpochEnd: ?*const fn (*anyopaque, *TrainerState) bool = null, // return false to stop
        onTrainEnd: ?*const fn (*anyopaque, *TrainerState) void = null,
    };

    pub fn onBatchEnd(self: Callback, state: *TrainerState) void {
        if (self.vtable.onBatchEnd) |f| f(self.ptr, state);
    }

    pub fn onEpochEnd(self: Callback, state: *TrainerState) bool {
        if (self.vtable.onEpochEnd) |f| return f(self.ptr, state);
        return true; // continue by default
    }

    pub fn onTrainEnd(self: Callback, state: *TrainerState) void {
        if (self.vtable.onTrainEnd) |f| f(self.ptr, state);
    }
};

/// Current training state passed to callbacks
pub const TrainerState = struct {
    epoch: u32,
    total_epochs: u32,
    batch: u32,
    total_batches: u32,
    global_step: u64,
    train_loss: f32,
    train_acc: f32,
    val_loss: f32,
    val_acc: f32,
    best_val_loss: f32,
    best_val_acc: f32,
    best_epoch: u32,
    lr: f32,
    should_stop: bool,
};

/// Monitor metric for early stopping
pub const Monitor = enum {
    val_loss,
    val_acc,
    train_loss,
    train_acc,
};

/// Early stopping callback
pub const EarlyStopping = struct {
    patience: u32,
    min_delta: f32,
    monitor: Monitor,
    mode: Mode,
    best_value: f32,
    wait: u32,
    stopped_epoch: ?u32,

    pub const Mode = enum {
        min, // stop when metric stops decreasing
        max, // stop when metric stops increasing
    };

    pub fn init(patience: u32, min_delta: f32, monitor: Monitor) EarlyStopping {
        const mode: Mode = switch (monitor) {
            .val_loss, .train_loss => .min,
            .val_acc, .train_acc => .max,
        };
        return .{
            .patience = patience,
            .min_delta = min_delta,
            .monitor = monitor,
            .mode = mode,
            .best_value = if (mode == .min) std.math.inf(f32) else -std.math.inf(f32),
            .wait = 0,
            .stopped_epoch = null,
        };
    }

    /// Check if training should stop. Returns true if should stop.
    pub fn check(self: *EarlyStopping, state: *const TrainerState) bool {
        const current = self.getCurrentMetric(state);

        const improved = switch (self.mode) {
            .min => current < self.best_value - self.min_delta,
            .max => current > self.best_value + self.min_delta,
        };

        if (improved) {
            self.best_value = current;
            self.wait = 0;
            return false;
        }

        self.wait += 1;
        if (self.wait >= self.patience) {
            self.stopped_epoch = state.epoch;
            return true;
        }

        return false;
    }

    fn getCurrentMetric(self: *const EarlyStopping, state: *const TrainerState) f32 {
        return switch (self.monitor) {
            .val_loss => state.val_loss,
            .val_acc => state.val_acc,
            .train_loss => state.train_loss,
            .train_acc => state.train_acc,
        };
    }

    /// Get callback interface
    pub fn callback(self: *EarlyStopping) Callback {
        return .{
            .ptr = self,
            .vtable = &.{
                .onEpochEnd = onEpochEnd,
            },
        };
    }

    fn onEpochEnd(ptr: *anyopaque, state: *TrainerState) bool {
        const self: *EarlyStopping = @ptrCast(@alignCast(ptr));
        if (self.check(state)) {
            state.should_stop = true;
            return false;
        }
        return true;
    }
};

/// Best model checkpoint callback
pub const ModelCheckpoint = struct {
    save_path: []const u8,
    monitor: Monitor,
    mode: EarlyStopping.Mode,
    best_value: f32,
    save_fn: ?*const fn ([]const u8, *const TrainerState) anyerror!void,

    pub fn init(save_path: []const u8, monitor: Monitor) ModelCheckpoint {
        const mode: EarlyStopping.Mode = switch (monitor) {
            .val_loss, .train_loss => .min,
            .val_acc, .train_acc => .max,
        };
        return .{
            .save_path = save_path,
            .monitor = monitor,
            .mode = mode,
            .best_value = if (mode == .min) std.math.inf(f32) else -std.math.inf(f32),
            .save_fn = null,
        };
    }

    pub fn setSaveFn(self: *ModelCheckpoint, f: *const fn ([]const u8, *const TrainerState) anyerror!void) void {
        self.save_fn = f;
    }

    /// Check and potentially save checkpoint
    pub fn check(self: *ModelCheckpoint, state: *const TrainerState) !bool {
        const current = switch (self.monitor) {
            .val_loss => state.val_loss,
            .val_acc => state.val_acc,
            .train_loss => state.train_loss,
            .train_acc => state.train_acc,
        };

        const improved = switch (self.mode) {
            .min => current < self.best_value,
            .max => current > self.best_value,
        };

        if (improved) {
            self.best_value = current;
            if (self.save_fn) |f| {
                try f(self.save_path, state);
            }
            return true;
        }

        return false;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "early stopping - min mode" {
    var es = EarlyStopping.init(3, 0.001, .val_loss);

    var state = TrainerState{
        .epoch = 1,
        .total_epochs = 10,
        .batch = 0,
        .total_batches = 100,
        .global_step = 100,
        .train_loss = 0.5,
        .train_acc = 0.8,
        .val_loss = 0.4,
        .val_acc = 0.85,
        .best_val_loss = 0.4,
        .best_val_acc = 0.85,
        .best_epoch = 1,
        .lr = 0.01,
        .should_stop = false,
    };

    // First call - should improve
    try std.testing.expect(!es.check(&state));
    try std.testing.expectEqual(@as(u32, 0), es.wait);

    // No improvement
    state.val_loss = 0.4;
    try std.testing.expect(!es.check(&state));
    try std.testing.expectEqual(@as(u32, 1), es.wait);

    // Still no improvement
    state.val_loss = 0.41;
    try std.testing.expect(!es.check(&state));
    try std.testing.expectEqual(@as(u32, 2), es.wait);

    // Patience exhausted
    state.val_loss = 0.42;
    try std.testing.expect(es.check(&state));
}

test "early stopping - max mode" {
    var es = EarlyStopping.init(2, 0.01, .val_acc);

    var state = TrainerState{
        .epoch = 1,
        .total_epochs = 10,
        .batch = 0,
        .total_batches = 100,
        .global_step = 100,
        .train_loss = 0.5,
        .train_acc = 0.8,
        .val_loss = 0.4,
        .val_acc = 0.85,
        .best_val_loss = 0.4,
        .best_val_acc = 0.85,
        .best_epoch = 1,
        .lr = 0.01,
        .should_stop = false,
    };

    // Improvement
    try std.testing.expect(!es.check(&state));

    // Better
    state.val_acc = 0.90;
    try std.testing.expect(!es.check(&state));
    try std.testing.expectEqual(@as(u32, 0), es.wait);

    // No improvement
    state.val_acc = 0.89;
    try std.testing.expect(!es.check(&state));
    try std.testing.expectEqual(@as(u32, 1), es.wait);

    // Patience exhausted
    state.val_acc = 0.88;
    try std.testing.expect(es.check(&state));
}
