//! Training checkpoint management using .tenzor format.
//!
//! Saves and loads:
//! - Model weights
//! - Optimizer state (velocities, Adam moments)
//! - Training metadata (epoch, step, best metrics)
//!
//! Example:
//!   // Save checkpoint
//!   try Checkpoint.save(allocator, "checkpoint.tenzor", weights, optimizer, metadata);
//!
//!   // Load checkpoint
//!   var cp = try Checkpoint.load(allocator, "checkpoint.tenzor");
//!   defer cp.deinit();
//!
//!   // Access weights via mmap
//!   const conv1_w = cp.getTensor("conv1.weight");

const std = @import("std");
const tenzor_format = @import("../io/tenzor_format.zig");

/// Checkpoint metadata stored in .tenzor JSON section
pub const CheckpointMetadata = struct {
    epoch: u32,
    global_step: u64,
    best_val_loss: f32,
    best_val_acc: f32,
    best_epoch: u32,
    learning_rate: f32,
    model_name: []const u8,
    timestamp: i64,

    pub fn toJson(self: CheckpointMetadata, allocator: std.mem.Allocator) ![]const u8 {
        return std.fmt.allocPrint(allocator,
            \\{{"epoch":{d},"global_step":{d},"best_val_loss":{d:.6},"best_val_acc":{d:.6},"best_epoch":{d},"learning_rate":{d:.8},"model_name":"{s}","timestamp":{d}}}
        , .{
            self.epoch,
            self.global_step,
            self.best_val_loss,
            self.best_val_acc,
            self.best_epoch,
            self.learning_rate,
            self.model_name,
            self.timestamp,
        });
    }
};

/// Loaded checkpoint with mmap'd tensors
pub const Checkpoint = struct {
    file: tenzor_format.TenzorFile,
    allocator: std.mem.Allocator,

    pub fn load(allocator: std.mem.Allocator, path: []const u8) !Checkpoint {
        const file = try tenzor_format.TenzorFile.open(allocator, path);
        return .{
            .file = file,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Checkpoint) void {
        self.file.close();
    }

    /// Get tensor data by name (zero-copy from mmap)
    pub fn getTensor(self: *Checkpoint, name: []const u8) ?[]const f32 {
        return self.file.getF32(name);
    }

    /// Get tensor entry by name
    pub fn getTensorEntry(self: *Checkpoint, name: []const u8) ?*const tenzor_format.TensorEntry {
        return self.file.getTensor(name);
    }

    /// Get raw tensor bytes by name
    pub fn getTensorBytes(self: *Checkpoint, name: []const u8) ?[]const u8 {
        const entry = self.file.getTensor(name) orelse return null;
        return self.file.getData(u8, entry);
    }

    /// List all tensor names in checkpoint
    pub fn tensorNames(self: *const Checkpoint, allocator: std.mem.Allocator) ![][]const u8 {
        // We can't recover original names from hashes, so return hashes as hex strings
        var names: std.ArrayList([]const u8) = .empty;
        errdefer {
            for (names.items) |n| allocator.free(n);
            names.deinit(allocator);
        }

        for (self.file.index) |entry| {
            const hex = try std.fmt.allocPrint(allocator, "0x{x:0>16}", .{entry.name_hash});
            try names.append(allocator, hex);
        }

        return names.toOwnedSlice(allocator);
    }

    /// Get number of tensors
    pub fn tensorCount(self: *const Checkpoint) usize {
        return self.file.index.len;
    }
};

/// Checkpoint writer for saving training state
pub const CheckpointWriter = struct {
    writer: tenzor_format.TenzorWriter,
    allocator: std.mem.Allocator,
    metadata_json: ?[]const u8,

    pub fn init(allocator: std.mem.Allocator, path: []const u8) !CheckpointWriter {
        return .{
            .writer = try tenzor_format.TenzorWriter.create(allocator, path),
            .allocator = allocator,
            .metadata_json = null,
        };
    }

    pub fn deinit(self: *CheckpointWriter) void {
        if (self.metadata_json) |json| {
            self.allocator.free(json);
        }
        self.writer.deinit();
    }

    /// Add a model weight tensor
    pub fn addWeight(self: *CheckpointWriter, name: []const u8, data: []const f32, shape: []const usize) !void {
        try self.writer.addF32(name, shape, data);
    }

    /// Add optimizer state tensor (e.g., "optim/conv1.weight/v" for velocity)
    pub fn addOptimizerState(self: *CheckpointWriter, param_name: []const u8, state_name: []const u8, data: []const f32, shape: []const usize) !void {
        var buf: [256]u8 = undefined;
        const full_name = std.fmt.bufPrint(&buf, "optim/{s}/{s}", .{ param_name, state_name }) catch return error.NameTooLong;
        try self.writer.addF32(full_name, shape, data);
    }

    /// Set checkpoint metadata
    pub fn setMetadata(self: *CheckpointWriter, metadata: CheckpointMetadata) !void {
        // Free previous metadata if any
        if (self.metadata_json) |old| {
            self.allocator.free(old);
        }
        const json = try metadata.toJson(self.allocator);
        self.metadata_json = json;
        self.writer.setMetadataJson(json);
    }

    /// Finalize and write checkpoint to disk
    pub fn finish(self: *CheckpointWriter) !void {
        try self.writer.finish();
    }
};

/// Save a complete training checkpoint
pub fn saveCheckpoint(
    allocator: std.mem.Allocator,
    path: []const u8,
    weights: anytype,
    optimizer_state: anytype,
    metadata: CheckpointMetadata,
) !void {
    var writer = try CheckpointWriter.init(allocator, path);
    defer writer.deinit();

    // Add model weights
    inline for (std.meta.fields(@TypeOf(weights))) |field| {
        const tensor = @field(weights, field.name);
        if (@hasField(@TypeOf(tensor), "data") and @hasField(@TypeOf(tensor), "shape")) {
            try writer.addWeight(field.name, tensor.data, &tensor.shape);
        }
    }

    // Add optimizer state
    if (@TypeOf(optimizer_state) != void) {
        inline for (std.meta.fields(@TypeOf(optimizer_state))) |field| {
            const state = @field(optimizer_state, field.name);
            if (@hasField(@TypeOf(state), "velocity")) {
                try writer.addOptimizerState(field.name, "v", state.velocity.data, &state.velocity.shape);
            }
            if (@hasField(@TypeOf(state), "m")) {
                try writer.addOptimizerState(field.name, "m", state.m.data, &state.m.shape);
            }
            if (@hasField(@TypeOf(state), "v_hat")) {
                try writer.addOptimizerState(field.name, "v_hat", state.v_hat.data, &state.v_hat.shape);
            }
        }
    }

    try writer.setMetadata(metadata);
    try writer.finish();
}

// ============================================================================
// Tests
// ============================================================================

test "checkpoint metadata to json" {
    const allocator = std.testing.allocator;

    const meta = CheckpointMetadata{
        .epoch = 5,
        .global_step = 4685,
        .best_val_loss = 0.198,
        .best_val_acc = 0.963,
        .best_epoch = 3,
        .learning_rate = 0.0087,
        .model_name = "lenet",
        .timestamp = 1700000000,
    };

    const json = try meta.toJson(allocator);
    defer allocator.free(json);

    try std.testing.expect(std.mem.indexOf(u8, json, "\"epoch\":5") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"model_name\":\"lenet\"") != null);
}

test "checkpoint writer creates file" {
    const allocator = std.testing.allocator;
    const test_path = "/tmp/test_checkpoint.tenzor";

    // Clean up from previous runs
    std.fs.cwd().deleteFile(test_path) catch {};

    {
        var writer = try CheckpointWriter.init(allocator, test_path);
        defer writer.deinit();

        // Add a simple tensor
        const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
        try writer.addWeight("test_weight", &data, &[_]usize{ 2, 2 });

        try writer.setMetadata(.{
            .epoch = 1,
            .global_step = 100,
            .best_val_loss = 0.5,
            .best_val_acc = 0.8,
            .best_epoch = 1,
            .learning_rate = 0.01,
            .model_name = "test",
            .timestamp = 0,
        });

        try writer.finish();
    }

    // Verify file was created
    const stat = try std.fs.cwd().statFile(test_path);
    try std.testing.expect(stat.size > 0);

    // Load and verify
    {
        var cp = try Checkpoint.load(allocator, test_path);
        defer cp.deinit();

        try std.testing.expectEqual(@as(usize, 1), cp.tensorCount());

        const tensor = cp.getTensor("test_weight");
        try std.testing.expect(tensor != null);
        try std.testing.expectEqual(@as(f32, 1.0), tensor.?[0]);
        try std.testing.expectEqual(@as(f32, 4.0), tensor.?[3]);
    }

    // Clean up
    std.fs.cwd().deleteFile(test_path) catch {};
}
