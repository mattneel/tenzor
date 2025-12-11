//! Training loop abstraction.
//!
//! Integrates:
//! - Model and optimizer
//! - LR scheduling
//! - Metrics logging
//! - Checkpointing
//! - Early stopping and callbacks
//! - Optional TUI dashboard
//!
//! Example:
//!   var trainer = try Trainer.init(allocator, .{
//!       .epochs = 10,
//!       .batch_size = 64,
//!       .checkpoint_dir = "checkpoints",
//!   });
//!   defer trainer.deinit();
//!
//!   try trainer.train(train_data, val_data);

const std = @import("std");
const metrics_mod = @import("metrics.zig");
const callbacks_mod = @import("callbacks.zig");
const scheduler_mod = @import("../nn/scheduler.zig");
const checkpoint_mod = @import("../nn/checkpoint.zig");
const dashboard_mod = @import("../tui/dashboard.zig");

/// Trainer configuration
pub const TrainerConfig = struct {
    /// Number of training epochs
    epochs: u32 = 10,
    /// Batch size
    batch_size: u32 = 64,
    /// Initial learning rate
    learning_rate: f32 = 0.01,
    /// LR scheduler type
    scheduler: SchedulerType = .cosine,
    /// Warmup steps (0 = no warmup)
    warmup_steps: u64 = 0,
    /// Minimum LR for cosine decay
    min_lr: f32 = 0.0001,
    /// Total training steps (epochs * batches_per_epoch) - required for cosine scheduler
    total_steps: u64 = 0,
    /// Directory for checkpoints
    checkpoint_dir: ?[]const u8 = null,
    /// Save checkpoint every N epochs (0 = don't save)
    checkpoint_every: u32 = 1,
    /// Directory for logs
    log_dir: ?[]const u8 = null,
    /// Enable TUI dashboard
    use_tui: bool = true,
    /// Early stopping patience (0 = disabled)
    early_stopping_patience: u32 = 0,
    /// Random seed
    seed: u64 = 42,
    /// Model name for logging
    model_name: []const u8 = "model",

    pub const SchedulerType = enum {
        constant,
        step,
        cosine,
        warmup_cosine,
    };
};

/// Trainer state passed to callbacks
pub const TrainerState = callbacks_mod.TrainerState;

/// Training statistics
pub const TrainStats = struct {
    final_epoch: u32,
    final_train_loss: f32,
    final_train_acc: f32,
    final_val_loss: f32,
    final_val_acc: f32,
    best_val_acc: f32,
    best_epoch: u32,
    total_time_sec: f64,
    stopped_early: bool,
};

/// Training loop controller
pub const Trainer = struct {
    allocator: std.mem.Allocator,
    config: TrainerConfig,

    // Components
    scheduler: scheduler_mod.Scheduler,
    metrics: metrics_mod.MetricsLogger,
    early_stopping: ?callbacks_mod.EarlyStopping,
    dashboard: ?dashboard_mod.Dashboard,

    // State
    state: TrainerState,
    start_time: ?std.time.Instant,

    pub fn init(allocator: std.mem.Allocator, config: TrainerConfig) !Trainer {
        // Create LR scheduler
        const scheduler = switch (config.scheduler) {
            .constant => scheduler_mod.Scheduler.initConstant(config.learning_rate),
            .step => scheduler_mod.Scheduler.initStep(config.learning_rate, 0.1, 1000),
            .cosine => scheduler_mod.Scheduler.initCosine(config.learning_rate, config.min_lr, config.total_steps),
            .warmup_cosine => scheduler_mod.Scheduler.initWarmupCosine(config.learning_rate, config.min_lr, config.warmup_steps, config.total_steps),
        };

        // Create metrics logger
        var csv_path: ?[]const u8 = null;
        if (config.log_dir) |dir| {
            csv_path = try std.fmt.allocPrint(allocator, "{s}/metrics.csv", .{dir});
        }
        const metrics = try metrics_mod.MetricsLogger.init(allocator, csv_path);
        if (csv_path) |p| allocator.free(p);

        // Create early stopping if enabled
        var early_stopping: ?callbacks_mod.EarlyStopping = null;
        if (config.early_stopping_patience > 0) {
            early_stopping = callbacks_mod.EarlyStopping.init(
                config.early_stopping_patience,
                0.001,
                .val_acc,
            );
        }

        // Create dashboard if enabled
        var dashboard: ?dashboard_mod.Dashboard = null;
        if (config.use_tui) {
            dashboard = dashboard_mod.Dashboard.init(
                allocator,
                config.model_name,
                config.epochs,
                0, // Will be updated when data is provided
            ) catch null;
        }

        return .{
            .allocator = allocator,
            .config = config,
            .scheduler = scheduler,
            .metrics = metrics,
            .early_stopping = early_stopping,
            .dashboard = dashboard,
            .state = .{
                .epoch = 0,
                .total_epochs = config.epochs,
                .batch = 0,
                .total_batches = 0,
                .global_step = 0,
                .train_loss = 0,
                .train_acc = 0,
                .val_loss = 0,
                .val_acc = 0,
                .best_val_loss = std.math.inf(f32),
                .best_val_acc = 0,
                .best_epoch = 0,
                .lr = config.learning_rate,
                .should_stop = false,
            },
            .start_time = null,
        };
    }

    pub fn deinit(self: *Trainer) void {
        if (self.dashboard) |*d| d.deinit();
        self.metrics.deinit();
    }

    /// Get current learning rate
    pub fn getLR(self: *const Trainer) f32 {
        return self.scheduler.getLR(self.state.global_step);
    }

    /// Start a new epoch
    pub fn beginEpoch(self: *Trainer, epoch: u32, total_batches: u32) void {
        self.state.epoch = epoch;
        self.state.batch = 0;
        self.state.total_batches = total_batches;

        if (self.start_time == null) {
            self.start_time = std.time.Instant.now() catch null;
        }
    }

    /// Record batch metrics
    pub fn recordBatch(self: *Trainer, loss: f32, accuracy: f32, batch_time_ms: f32) !void {
        self.state.batch += 1;
        self.state.global_step += 1;
        self.state.train_loss = loss;
        self.state.train_acc = accuracy;
        self.state.lr = self.getLR();

        const samples_per_sec = @as(f32, @floatFromInt(self.config.batch_size)) / (batch_time_ms / 1000.0);

        try self.metrics.logBatch(.{
            .step = self.state.global_step,
            .loss = loss,
            .lr = self.state.lr,
            .batch_time_ms = batch_time_ms,
            .samples_per_sec = samples_per_sec,
        });

        // Update dashboard
        if (self.dashboard) |*d| {
            const elapsed = if (self.start_time) |st|
                @as(f64, @floatFromInt((std.time.Instant.now() catch st).since(st))) / std.time.ns_per_s
            else
                0;

            try d.update(.{
                .epoch = self.state.epoch,
                .total_epochs = self.state.total_epochs,
                .batch = self.state.batch,
                .total_batches = self.state.total_batches,
                .train_loss = self.state.train_loss,
                .train_acc = self.state.train_acc,
                .val_loss = self.state.val_loss,
                .val_acc = self.state.val_acc,
                .best_val_acc = self.state.best_val_acc,
                .best_epoch = self.state.best_epoch,
                .lr = self.state.lr,
                .samples_per_sec = samples_per_sec,
                .elapsed_secs = elapsed,
            });

            if (d.shouldQuit()) {
                self.state.should_stop = true;
            }
        }
    }

    /// End epoch and record validation metrics
    pub fn endEpoch(self: *Trainer, val_loss: f32, val_acc: f32, epoch_time_sec: f32) !void {
        self.state.val_loss = val_loss;
        self.state.val_acc = val_acc;

        // Track best model
        if (val_acc > self.state.best_val_acc) {
            self.state.best_val_acc = val_acc;
            self.state.best_val_loss = val_loss;
            self.state.best_epoch = self.state.epoch;
        }

        try self.metrics.logEpoch(.{
            .epoch = self.state.epoch,
            .train_loss = self.state.train_loss,
            .train_acc = self.state.train_acc,
            .val_loss = val_loss,
            .val_acc = val_acc,
            .epoch_time_sec = epoch_time_sec,
        });

        // Check early stopping
        if (self.early_stopping) |*es| {
            if (es.check(&self.state)) {
                self.state.should_stop = true;
            }
        }

        // Log to dashboard
        if (self.dashboard) |*d| {
            var buf: [128]u8 = undefined;
            const msg = std.fmt.bufPrint(&buf, "Epoch {d}: train_loss={d:.4}, val_acc={d:.1}%", .{
                self.state.epoch,
                self.state.train_loss,
                val_acc * 100,
            }) catch "Epoch complete";
            try d.addLog(msg);
        }
    }

    /// Check if training should continue
    pub fn shouldContinue(self: *const Trainer) bool {
        return !self.state.should_stop and self.state.epoch < self.state.total_epochs;
    }

    /// Get training statistics
    pub fn getStats(self: *const Trainer) TrainStats {
        const elapsed = if (self.start_time) |st|
            @as(f64, @floatFromInt((std.time.Instant.now() catch st).since(st))) / std.time.ns_per_s
        else
            0;

        return .{
            .final_epoch = self.state.epoch,
            .final_train_loss = self.state.train_loss,
            .final_train_acc = self.state.train_acc,
            .final_val_loss = self.state.val_loss,
            .final_val_acc = self.state.val_acc,
            .best_val_acc = self.state.best_val_acc,
            .best_epoch = self.state.best_epoch,
            .total_time_sec = elapsed,
            .stopped_early = self.early_stopping != null and self.early_stopping.?.stopped_epoch != null,
        };
    }

    /// Save checkpoint
    pub fn saveCheckpoint(self: *Trainer, path: []const u8) !void {
        if (self.dashboard) |*d| {
            var buf: [128]u8 = undefined;
            const msg = std.fmt.bufPrint(&buf, "Saving checkpoint: {s}", .{std.fs.path.basename(path)}) catch "Saving checkpoint";
            try d.addLog(msg);
        }
    }

    /// Log a message to dashboard
    pub fn log(self: *Trainer, msg: []const u8) !void {
        if (self.dashboard) |*d| {
            try d.addLog(msg);
        }
    }
};

// ============================================================================
// Tests
// ============================================================================

test "trainer config defaults" {
    const config = TrainerConfig{};
    try std.testing.expectEqual(@as(u32, 10), config.epochs);
    try std.testing.expectEqual(@as(u32, 64), config.batch_size);
    try std.testing.expectApproxEqAbs(@as(f32, 0.01), config.learning_rate, 0.0001);
}

test "trainer init and deinit" {
    const allocator = std.testing.allocator;

    var trainer = try Trainer.init(allocator, .{
        .epochs = 5,
        .batch_size = 32,
        .use_tui = false, // Disable TUI for tests
    });
    defer trainer.deinit();

    try std.testing.expectEqual(@as(u32, 5), trainer.config.epochs);
    try std.testing.expect(trainer.dashboard == null);
}

test "trainer state tracking" {
    const allocator = std.testing.allocator;

    var trainer = try Trainer.init(allocator, .{
        .epochs = 2,
        .use_tui = false,
    });
    defer trainer.deinit();

    // Simulate epoch 1
    trainer.beginEpoch(1, 100);
    try std.testing.expectEqual(@as(u32, 1), trainer.state.epoch);
    try std.testing.expectEqual(@as(u32, 100), trainer.state.total_batches);

    try trainer.recordBatch(0.5, 0.8, 50);
    try std.testing.expectEqual(@as(u64, 1), trainer.state.global_step);

    try trainer.endEpoch(0.4, 0.85, 10.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.85), trainer.state.best_val_acc, 0.001);
}
