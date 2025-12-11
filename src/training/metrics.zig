//! Training metrics collection and logging.
//!
//! Collects per-batch and per-epoch metrics, supports CSV/JSON export.

const std = @import("std");

/// Per-batch training metrics
pub const BatchMetrics = struct {
    step: u64,
    loss: f32,
    lr: f32,
    batch_time_ms: f32,
    samples_per_sec: f32,
};

/// Per-epoch training metrics
pub const EpochMetrics = struct {
    epoch: u32,
    train_loss: f32,
    train_acc: f32,
    val_loss: f32,
    val_acc: f32,
    epoch_time_sec: f32,
};

/// Metrics history for tracking training progress
pub const MetricsHistory = struct {
    batch_metrics: std.ArrayList(BatchMetrics),
    epoch_metrics: std.ArrayList(EpochMetrics),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) MetricsHistory {
        return .{
            .batch_metrics = .empty,
            .epoch_metrics = .empty,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *MetricsHistory) void {
        self.batch_metrics.deinit(self.allocator);
        self.epoch_metrics.deinit(self.allocator);
    }

    pub fn addBatch(self: *MetricsHistory, metrics: BatchMetrics) !void {
        try self.batch_metrics.append(self.allocator, metrics);
    }

    pub fn addEpoch(self: *MetricsHistory, metrics: EpochMetrics) !void {
        try self.epoch_metrics.append(self.allocator, metrics);
    }

    /// Get the latest batch metrics
    pub fn lastBatch(self: *const MetricsHistory) ?BatchMetrics {
        if (self.batch_metrics.items.len == 0) return null;
        return self.batch_metrics.items[self.batch_metrics.items.len - 1];
    }

    /// Get the latest epoch metrics
    pub fn lastEpoch(self: *const MetricsHistory) ?EpochMetrics {
        if (self.epoch_metrics.items.len == 0) return null;
        return self.epoch_metrics.items[self.epoch_metrics.items.len - 1];
    }
};

/// Metrics logger that writes to files
pub const MetricsLogger = struct {
    csv_file: ?std.fs.File,
    history: MetricsHistory,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, csv_path: ?[]const u8) !MetricsLogger {
        var csv_file: ?std.fs.File = null;
        if (csv_path) |path| {
            csv_file = try std.fs.cwd().createFile(path, .{});
            // Write CSV header
            try csv_file.?.writeAll("epoch,train_loss,train_acc,val_loss,val_acc,epoch_time_sec\n");
        }

        return .{
            .csv_file = csv_file,
            .history = MetricsHistory.init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *MetricsLogger) void {
        if (self.csv_file) |f| f.close();
        self.history.deinit();
    }

    pub fn logBatch(self: *MetricsLogger, metrics: BatchMetrics) !void {
        try self.history.addBatch(metrics);
    }

    pub fn logEpoch(self: *MetricsLogger, metrics: EpochMetrics) !void {
        try self.history.addEpoch(metrics);

        // Write to CSV
        if (self.csv_file) |f| {
            var buf: [256]u8 = undefined;
            const line = std.fmt.bufPrint(&buf, "{d},{d:.6},{d:.4},{d:.6},{d:.4},{d:.2}\n", .{
                metrics.epoch,
                metrics.train_loss,
                metrics.train_acc,
                metrics.val_loss,
                metrics.val_acc,
                metrics.epoch_time_sec,
            }) catch return;
            try f.writeAll(line);
        }
    }

    pub fn flush(self: *MetricsLogger) !void {
        // Files are auto-flushed on close, but we can sync if needed
        _ = self;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "metrics history" {
    const allocator = std.testing.allocator;
    var history = MetricsHistory.init(allocator);
    defer history.deinit();

    try history.addBatch(.{
        .step = 1,
        .loss = 0.5,
        .lr = 0.01,
        .batch_time_ms = 50,
        .samples_per_sec = 1280,
    });

    try std.testing.expectEqual(@as(usize, 1), history.batch_metrics.items.len);

    const last = history.lastBatch().?;
    try std.testing.expectEqual(@as(u64, 1), last.step);
}

test "epoch metrics" {
    const allocator = std.testing.allocator;
    var history = MetricsHistory.init(allocator);
    defer history.deinit();

    try history.addEpoch(.{
        .epoch = 1,
        .train_loss = 0.5,
        .train_acc = 0.85,
        .val_loss = 0.4,
        .val_acc = 0.90,
        .epoch_time_sec = 30.5,
    });

    const last = history.lastEpoch().?;
    try std.testing.expectEqual(@as(u32, 1), last.epoch);
    try std.testing.expectApproxEqAbs(@as(f32, 0.90), last.val_acc, 0.001);
}
