//! Training dashboard TUI.
//!
//! Real-time training metrics display with progress bars, loss curves,
//! and scrolling log output.

const std = @import("std");
const terminal = @import("terminal.zig");
const widgets = @import("widgets.zig");
const Terminal = terminal.Terminal;
const Color = terminal.Color;
const Key = terminal.Key;

/// Training state passed to dashboard for rendering
pub const TrainerState = struct {
    epoch: u32,
    total_epochs: u32,
    batch: u32,
    total_batches: u32,
    train_loss: f32,
    train_acc: f32,
    val_loss: f32,
    val_acc: f32,
    best_val_acc: f32,
    best_epoch: u32,
    lr: f32,
    samples_per_sec: f32,
    elapsed_secs: f64,
};

/// Dashboard for training visualization
pub const Dashboard = struct {
    allocator: std.mem.Allocator,
    term: ?Terminal,
    title: []const u8,

    // Widgets
    epoch_progress: widgets.ProgressBar,
    batch_progress: widgets.ProgressBar,
    loss_chart: widgets.SparklineChart,
    acc_chart: widgets.SparklineChart,
    metrics: widgets.MetricsPanel,
    log: widgets.LogView,

    // State
    running: bool,
    last_render: std.time.Instant,

    const RENDER_INTERVAL_MS = 100; // 10 Hz refresh

    pub fn init(allocator: std.mem.Allocator, title: []const u8, total_epochs: u32, total_batches: u32) !Dashboard {
        var term = Terminal.init(allocator) catch |err| {
            if (err == error.NoTTY) {
                // Running without TTY, dashboard disabled
                return .{
                    .allocator = allocator,
                    .term = null,
                    .title = title,
                    .epoch_progress = widgets.ProgressBar.init("Epoch", total_epochs, 60),
                    .batch_progress = widgets.ProgressBar.init("Batch", total_batches, 60),
                    .loss_chart = widgets.SparklineChart.init(allocator, "Loss", 40, 6),
                    .acc_chart = widgets.SparklineChart.init(allocator, "Accuracy", 40, 6),
                    .metrics = widgets.MetricsPanel.init(allocator),
                    .log = widgets.LogView.init(allocator, 100),
                    .running = false,
                    .last_render = std.time.Instant.now() catch unreachable,
                };
            }
            return err;
        };
        errdefer term.deinit();

        try term.enterRawMode();
        try term.clear();

        return .{
            .allocator = allocator,
            .term = term,
            .title = title,
            .epoch_progress = widgets.ProgressBar.init("Epoch", total_epochs, 60),
            .batch_progress = widgets.ProgressBar.init("Batch", total_batches, 60),
            .loss_chart = widgets.SparklineChart.init(allocator, "Loss", 40, 6),
            .acc_chart = widgets.SparklineChart.init(allocator, "Accuracy", 40, 6),
            .metrics = widgets.MetricsPanel.init(allocator),
            .log = widgets.LogView.init(allocator, 100),
            .running = true,
            .last_render = std.time.Instant.now() catch unreachable,
        };
    }

    pub fn deinit(self: *Dashboard) void {
        self.loss_chart.deinit();
        self.acc_chart.deinit();
        self.metrics.deinit();
        self.log.deinit();
        if (self.term) |*t| {
            t.deinit();
        }
    }

    /// Check for quit key (q or Ctrl+C)
    pub fn shouldQuit(self: *Dashboard) bool {
        if (self.term) |*t| {
            if (t.pollInput()) |key| {
                return switch (key) {
                    .ctrl_c, .ctrl_q => true,
                    .char => |c| c == 'q' or c == 'Q',
                    else => false,
                };
            }
        }
        return false;
    }

    /// Update dashboard with current training state
    pub fn update(self: *Dashboard, state: TrainerState) !void {
        // Update progress bars
        self.epoch_progress.total = state.total_epochs;
        self.epoch_progress.update(state.epoch);
        self.batch_progress.total = state.total_batches;
        self.batch_progress.update(state.batch);

        // Add loss/acc to charts
        if (state.train_loss > 0) {
            try self.loss_chart.addPoint(state.train_loss);
        }
        if (state.train_acc > 0) {
            try self.acc_chart.addPoint(state.train_acc);
        }

        // Update metrics panel
        try self.metrics.set("train_loss", state.train_loss, .float4);
        try self.metrics.set("train_acc", state.train_acc, .percent);
        try self.metrics.set("val_loss", state.val_loss, .float4);
        try self.metrics.set("val_acc", state.val_acc, .percent);
        try self.metrics.set("lr", state.lr, .float4);
        try self.metrics.set("samples/s", state.samples_per_sec, .integer);

        // Throttle rendering
        const now = std.time.Instant.now() catch return;
        const elapsed_ms = now.since(self.last_render) / std.time.ns_per_ms;
        if (elapsed_ms >= RENDER_INTERVAL_MS) {
            try self.render(state);
            self.last_render = now;
        }
    }

    /// Log a message
    pub fn addLog(self: *Dashboard, msg: []const u8) !void {
        try self.log.addLine(msg);
    }

    /// Force a render
    pub fn render(self: *Dashboard, state: TrainerState) !void {
        const t = &(self.term orelse return);

        t.refreshSize();
        try t.clear();

        const width = t.width;
        var row: u16 = 1;

        // Header
        try t.moveTo(row, 1);
        try t.setBold();
        try t.setFg(.cyan);
        try t.write("tenzor train");
        try t.reset();
        try t.write(" - ");
        try t.write(self.title);

        // Quit hint
        try t.moveTo(row, width - 8);
        try t.setFg(.bright_black);
        try t.write("[q]uit");
        try t.reset();

        row += 1;
        try widgets.drawHLine(t, row, 1, width);

        // Progress bars
        row += 1;
        try self.epoch_progress.render(t, row, 2);

        // ETA calculation
        if (state.epoch > 0 and state.elapsed_secs > 0) {
            const total_batches_done = (state.epoch - 1) * state.total_batches + state.batch;
            const total_batches_remaining = (state.total_epochs * state.total_batches) - total_batches_done;
            const batches_per_sec = @as(f64, @floatFromInt(total_batches_done)) / state.elapsed_secs;
            if (batches_per_sec > 0) {
                const eta_secs = @as(u64, @intFromFloat(@as(f64, @floatFromInt(total_batches_remaining)) / batches_per_sec));
                const eta_min = eta_secs / 60;
                const eta_sec = eta_secs % 60;
                try t.moveTo(row, width - 16);
                try t.print("ETA: {d}m {d}s", .{ eta_min, eta_sec });
            }
        }

        row += 1;
        try self.batch_progress.render(t, row, 2);

        row += 1;
        try widgets.drawHLine(t, row, 1, width);

        // Charts side by side
        row += 1;
        const half_width = (width - 4) / 2;
        try self.loss_chart.render(t, row, 2);
        try self.acc_chart.render(t, row, half_width + 4);

        row += 7; // Chart height
        try widgets.drawHLine(t, row, 1, width);

        // Metrics
        row += 1;
        try self.metrics.render(t, row, 2, 20);

        // Best model info
        row += 1;
        try t.moveTo(row, 2);
        try t.setFg(.bright_black);
        try t.write("best: ");
        try t.reset();
        try t.setFg(.green);
        try t.print("{d:.1}%", .{state.best_val_acc * 100.0});
        try t.reset();
        try t.print(" (epoch {d})", .{state.best_epoch});

        row += 1;
        try widgets.drawHLine(t, row, 1, width);

        // Log section
        row += 1;
        try t.moveTo(row, 2);
        try t.setFg(.bright_black);
        try t.write("[Log]");
        try t.reset();

        row += 1;
        const log_height = t.height - row - 1;
        try self.log.render(t, row, 2, log_height);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "dashboard state struct" {
    const state = TrainerState{
        .epoch = 5,
        .total_epochs = 10,
        .batch = 234,
        .total_batches = 937,
        .train_loss = 0.234,
        .train_acc = 0.942,
        .val_loss = 0.198,
        .val_acc = 0.961,
        .best_val_acc = 0.963,
        .best_epoch = 3,
        .lr = 0.0087,
        .samples_per_sec = 1234,
        .elapsed_secs = 120.5,
    };

    try std.testing.expectEqual(@as(u32, 5), state.epoch);
    try std.testing.expectEqual(@as(f32, 0.234), state.train_loss);
}
