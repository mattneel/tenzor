//! TUI widgets: progress bars, charts, and text displays.

const std = @import("std");
const terminal = @import("terminal.zig");
const Terminal = terminal.Terminal;
const Color = terminal.Color;

/// Progress bar widget
pub const ProgressBar = struct {
    label: []const u8,
    current: u64,
    total: u64,
    width: u16,

    pub fn init(label: []const u8, total: u64, width: u16) ProgressBar {
        return .{
            .label = label,
            .current = 0,
            .total = total,
            .width = width,
        };
    }

    pub fn update(self: *ProgressBar, current: u64) void {
        self.current = @min(current, self.total);
    }

    pub fn render(self: *const ProgressBar, term: *Terminal, row: u16, col: u16) !void {
        try term.moveTo(row, col);

        // Label
        try term.write(self.label);
        try term.write(": ");

        // Bar characters
        const bar_width = self.width -| @as(u16, @intCast(self.label.len + 10)); // space for label + percentage
        const filled = if (self.total > 0)
            @as(u16, @intCast((self.current * bar_width) / self.total))
        else
            0;
        const empty = bar_width -| filled;

        try term.write("[");
        try term.setFg(.green);

        // Filled portion
        var i: u16 = 0;
        while (i < filled) : (i += 1) {
            try term.write("\xe2\x96\x88"); // Full block
        }

        try term.setFg(.bright_black);
        // Empty portion
        i = 0;
        while (i < empty) : (i += 1) {
            try term.write("\xe2\x96\x91"); // Light shade
        }

        try term.reset();
        try term.write("] ");

        // Percentage
        const pct = if (self.total > 0)
            (self.current * 100) / self.total
        else
            0;
        try term.print("{d:3}%", .{pct});
    }
};

/// Simple ASCII sparkline chart
pub const SparklineChart = struct {
    title: []const u8,
    data: std.ArrayList(f32),
    width: u16,
    height: u16,
    allocator: std.mem.Allocator,

    // Braille-like blocks for higher resolution
    const blocks = [_][]const u8{ " ", "\xe2\x96\x81", "\xe2\x96\x82", "\xe2\x96\x83", "\xe2\x96\x84", "\xe2\x96\x85", "\xe2\x96\x86", "\xe2\x96\x87", "\xe2\x96\x88" };

    pub fn init(allocator: std.mem.Allocator, title: []const u8, width: u16, height: u16) SparklineChart {
        return .{
            .title = title,
            .data = .empty,
            .width = width,
            .height = height,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *SparklineChart) void {
        self.data.deinit(self.allocator);
    }

    pub fn addPoint(self: *SparklineChart, value: f32) !void {
        try self.data.append(self.allocator, value);
        // Keep only width points
        if (self.data.items.len > self.width) {
            _ = self.data.orderedRemove(0);
        }
    }

    pub fn render(self: *const SparklineChart, term: *Terminal, row: u16, col: u16) !void {
        if (self.data.items.len == 0) return;

        // Find min/max for scaling
        var min: f32 = self.data.items[0];
        var max: f32 = self.data.items[0];
        for (self.data.items) |v| {
            min = @min(min, v);
            max = @max(max, v);
        }

        // Title
        try term.moveTo(row, col);
        try term.setBold();
        try term.write(self.title);
        try term.reset();

        // Y-axis labels
        try term.moveTo(row + 1, col);
        try term.print("{d:.2}", .{max});
        try term.moveTo(row + self.height, col);
        try term.print("{d:.2}", .{min});

        // Chart area
        const chart_col = col + 8; // Space for y-axis labels
        const range = if (max - min > 0.0001) max - min else 1.0;

        // Render each column
        for (self.data.items, 0..) |value, i| {
            if (i >= self.width - 8) break;

            // Normalize to 0-8 for block selection
            const normalized = (value - min) / range;
            const block_idx: usize = @intFromFloat(normalized * 8.0);
            const safe_idx = @min(block_idx, blocks.len - 1);

            try term.moveTo(row + self.height - 1, chart_col + @as(u16, @intCast(i)));
            try term.setFg(.cyan);
            try term.write(blocks[safe_idx]);
        }

        try term.reset();
    }
};

/// Metrics display panel
pub const MetricsPanel = struct {
    pub const Metric = struct {
        label: []const u8,
        value: f32,
        format: Format,

        pub const Format = enum {
            float2,
            float4,
            percent,
            integer,
        };
    };

    metrics: std.ArrayList(Metric),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) MetricsPanel {
        return .{
            .metrics = .empty,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *MetricsPanel) void {
        self.metrics.deinit(self.allocator);
    }

    pub fn set(self: *MetricsPanel, label: []const u8, value: f32, format: Metric.Format) !void {
        // Update existing or add new
        for (self.metrics.items) |*m| {
            if (std.mem.eql(u8, m.label, label)) {
                m.value = value;
                return;
            }
        }
        try self.metrics.append(self.allocator, .{ .label = label, .value = value, .format = format });
    }

    pub fn render(self: *const MetricsPanel, term: *Terminal, row: u16, col: u16, cols_per_metric: u16) !void {
        var current_col = col;
        for (self.metrics.items) |m| {
            try term.moveTo(row, current_col);
            try term.setFg(.bright_black);
            try term.write(m.label);
            try term.write(": ");
            try term.reset();

            switch (m.format) {
                .float2 => try term.print("{d:.2}", .{m.value}),
                .float4 => try term.print("{d:.4}", .{m.value}),
                .percent => try term.print("{d:.1}%", .{m.value * 100.0}),
                .integer => try term.print("{d}", .{@as(i64, @intFromFloat(m.value))}),
            }

            current_col += cols_per_metric;
        }
    }
};

/// Scrolling log view
pub const LogView = struct {
    lines: std.ArrayList([]const u8),
    max_lines: usize,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, max_lines: usize) LogView {
        return .{
            .lines = .empty,
            .max_lines = max_lines,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *LogView) void {
        for (self.lines.items) |line| {
            self.allocator.free(line);
        }
        self.lines.deinit(self.allocator);
    }

    pub fn addLine(self: *LogView, text: []const u8) !void {
        const duped = try self.allocator.dupe(u8, text);
        errdefer self.allocator.free(duped);

        if (self.lines.items.len >= self.max_lines) {
            const old = self.lines.orderedRemove(0);
            self.allocator.free(old);
        }

        try self.lines.append(self.allocator, duped);
    }

    pub fn render(self: *const LogView, term: *Terminal, start_row: u16, col: u16, height: u16) !void {
        const start_idx = if (self.lines.items.len > height)
            self.lines.items.len - height
        else
            0;

        var row = start_row;
        for (self.lines.items[start_idx..]) |line| {
            if (row >= start_row + height) break;
            try term.moveTo(row, col);
            try term.setFg(.bright_black);
            try term.write(line);
            row += 1;
        }
        try term.reset();
    }
};

/// Draw a horizontal line
pub fn drawHLine(term: *Terminal, row: u16, col: u16, width: u16) !void {
    try term.moveTo(row, col);
    var i: u16 = 0;
    while (i < width) : (i += 1) {
        try term.write("\xe2\x94\x80"); // Box drawing light horizontal
    }
}

/// Draw a box border
pub fn drawBox(term: *Terminal, row: u16, col: u16, width: u16, height: u16) !void {
    // Top border
    try term.moveTo(row, col);
    try term.write("\xe2\x94\x8c"); // Top-left corner
    var i: u16 = 0;
    while (i < width - 2) : (i += 1) {
        try term.write("\xe2\x94\x80"); // Horizontal
    }
    try term.write("\xe2\x94\x90"); // Top-right corner

    // Sides
    var r: u16 = 1;
    while (r < height - 1) : (r += 1) {
        try term.moveTo(row + r, col);
        try term.write("\xe2\x94\x82"); // Vertical
        try term.moveTo(row + r, col + width - 1);
        try term.write("\xe2\x94\x82"); // Vertical
    }

    // Bottom border
    try term.moveTo(row + height - 1, col);
    try term.write("\xe2\x94\x94"); // Bottom-left corner
    i = 0;
    while (i < width - 2) : (i += 1) {
        try term.write("\xe2\x94\x80"); // Horizontal
    }
    try term.write("\xe2\x94\x98"); // Bottom-right corner
}

// ============================================================================
// Tests
// ============================================================================

test "progress bar init" {
    var pb = ProgressBar.init("Test", 100, 50);
    try std.testing.expectEqual(@as(u64, 0), pb.current);
    try std.testing.expectEqual(@as(u64, 100), pb.total);
    pb.update(50);
    try std.testing.expectEqual(@as(u64, 50), pb.current);
}

test "sparkline chart" {
    const allocator = std.testing.allocator;
    var chart = SparklineChart.init(allocator, "Loss", 20, 5);
    defer chart.deinit();

    try chart.addPoint(1.0);
    try chart.addPoint(0.8);
    try chart.addPoint(0.6);
    try std.testing.expectEqual(@as(usize, 3), chart.data.items.len);
}

test "metrics panel" {
    const allocator = std.testing.allocator;
    var panel = MetricsPanel.init(allocator);
    defer panel.deinit();

    try panel.set("loss", 0.234, .float4);
    try panel.set("acc", 0.95, .percent);
    try std.testing.expectEqual(@as(usize, 2), panel.metrics.items.len);

    // Update existing
    try panel.set("loss", 0.123, .float4);
    try std.testing.expectEqual(@as(usize, 2), panel.metrics.items.len);
}
