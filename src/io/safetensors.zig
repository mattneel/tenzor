//! SafeTensors format parser.
//!
//! SafeTensors is a simple binary format for storing tensors:
//! - 8 bytes: header size (little-endian u64)
//! - JSON header: metadata (tensor names, shapes, dtypes, offsets)
//! - Raw tensor data
//!
//! Reference: https://huggingface.co/docs/safetensors

const std = @import("std");

/// Supported data types in SafeTensors format.
pub const DType = enum {
    F16,
    BF16,
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    BOOL,

    pub fn fromString(s: []const u8) ?DType {
        const map = std.StaticStringMap(DType).initComptime(.{
            .{ "F16", .F16 },
            .{ "BF16", .BF16 },
            .{ "F32", .F32 },
            .{ "F64", .F64 },
            .{ "I8", .I8 },
            .{ "I16", .I16 },
            .{ "I32", .I32 },
            .{ "I64", .I64 },
            .{ "U8", .U8 },
            .{ "U16", .U16 },
            .{ "U32", .U32 },
            .{ "U64", .U64 },
            .{ "BOOL", .BOOL },
        });
        return map.get(s);
    }

    pub fn byteSize(self: DType) usize {
        return switch (self) {
            .F16, .BF16, .I16, .U16 => 2,
            .F32, .I32, .U32 => 4,
            .F64, .I64, .U64 => 8,
            .I8, .U8, .BOOL => 1,
        };
    }
};

/// Metadata for a single tensor in the file.
pub const TensorInfo = struct {
    name: []const u8,
    dtype: DType,
    shape: []const usize,
    data_start: usize,
    data_end: usize,

    pub fn byteSize(self: TensorInfo) usize {
        return self.data_end - self.data_start;
    }

    pub fn numel(self: TensorInfo) usize {
        if (self.shape.len == 0) return 1;
        var n: usize = 1;
        for (self.shape) |d| n *= d;
        return n;
    }
};

/// Parsed SafeTensors file.
pub const SafeTensors = struct {
    allocator: std.mem.Allocator,
    tensors: []TensorInfo,
    header_size: u64,
    data: []const u8, // Memory-mapped or loaded file data

    // Internal storage for parsed strings/arrays
    strings: std.ArrayList([]const u8),
    shapes: std.ArrayList([]const usize),

    pub fn deinit(self: *SafeTensors) void {
        // Free all allocated shapes
        for (self.shapes.items) |shape| {
            self.allocator.free(shape);
        }
        self.shapes.deinit(self.allocator);

        // Free all allocated strings
        for (self.strings.items) |s| {
            self.allocator.free(s);
        }
        self.strings.deinit(self.allocator);

        self.allocator.free(self.tensors);
    }

    /// Get tensor info by name.
    pub fn get(self: SafeTensors, name: []const u8) ?TensorInfo {
        for (self.tensors) |t| {
            if (std.mem.eql(u8, t.name, name)) {
                return t;
            }
        }
        return null;
    }

    /// Get raw tensor data as typed slice.
    pub fn getData(self: SafeTensors, comptime T: type, info: TensorInfo) []const T {
        const byte_data = self.data[self.header_size + 8 + info.data_start ..][0..info.byteSize()];
        const aligned: []align(@alignOf(T)) const u8 = @alignCast(byte_data);
        return std.mem.bytesAsSlice(T, aligned);
    }

    /// Get tensor data as f32 slice (most common case).
    pub fn getF32(self: SafeTensors, name: []const u8) ?[]const f32 {
        const info = self.get(name) orelse return null;
        if (info.dtype != .F32) return null;
        return self.getData(f32, info);
    }
};

/// Parse a SafeTensors file from raw bytes.
pub fn parse(allocator: std.mem.Allocator, data: []const u8) !SafeTensors {
    if (data.len < 8) {
        return error.InvalidFormat;
    }

    // Read header size (first 8 bytes, little-endian u64)
    const header_size = std.mem.readInt(u64, data[0..8], .little);

    if (data.len < 8 + header_size) {
        return error.InvalidFormat;
    }

    // Parse JSON header
    const json_bytes = data[8..][0..header_size];

    var strings: std.ArrayList([]const u8) = .empty;
    errdefer {
        for (strings.items) |s| allocator.free(s);
        strings.deinit(allocator);
    }

    var shapes: std.ArrayList([]const usize) = .empty;
    errdefer {
        for (shapes.items) |s| allocator.free(s);
        shapes.deinit(allocator);
    }

    var tensor_list: std.ArrayList(TensorInfo) = .empty;
    errdefer tensor_list.deinit(allocator);

    // Parse JSON
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_bytes, .{});
    defer parsed.deinit();

    const root = parsed.value;
    if (root != .object) {
        return error.InvalidFormat;
    }

    // Iterate over tensors
    var it = root.object.iterator();
    while (it.next()) |entry| {
        const tensor_name = entry.key_ptr.*;

        // Skip __metadata__ key
        if (std.mem.eql(u8, tensor_name, "__metadata__")) {
            continue;
        }

        const tensor_obj = entry.value_ptr.*;
        if (tensor_obj != .object) {
            return error.InvalidFormat;
        }

        // Parse dtype
        const dtype_val = tensor_obj.object.get("dtype") orelse return error.InvalidFormat;
        if (dtype_val != .string) return error.InvalidFormat;
        const dtype = DType.fromString(dtype_val.string) orelse return error.UnsupportedDType;

        // Parse shape
        const shape_val = tensor_obj.object.get("shape") orelse return error.InvalidFormat;
        if (shape_val != .array) return error.InvalidFormat;

        const shape = try allocator.alloc(usize, shape_val.array.items.len);
        errdefer allocator.free(shape);

        for (shape_val.array.items, 0..) |dim, i| {
            if (dim != .integer) return error.InvalidFormat;
            shape[i] = @intCast(dim.integer);
        }
        try shapes.append(allocator, shape);

        // Parse data_offsets
        const offsets_val = tensor_obj.object.get("data_offsets") orelse return error.InvalidFormat;
        if (offsets_val != .array or offsets_val.array.items.len != 2) return error.InvalidFormat;

        const start: usize = @intCast(offsets_val.array.items[0].integer);
        const end: usize = @intCast(offsets_val.array.items[1].integer);

        // Copy tensor name (it's owned by the JSON parser)
        const name_copy = try allocator.dupe(u8, tensor_name);
        try strings.append(allocator, name_copy);

        try tensor_list.append(allocator, .{
            .name = name_copy,
            .dtype = dtype,
            .shape = shape,
            .data_start = start,
            .data_end = end,
        });
    }

    return .{
        .allocator = allocator,
        .tensors = try tensor_list.toOwnedSlice(allocator),
        .header_size = header_size,
        .data = data,
        .strings = strings,
        .shapes = shapes,
    };
}

/// Load a SafeTensors file from disk.
pub fn load(allocator: std.mem.Allocator, path: []const u8) !struct { st: SafeTensors, data: []const u8 } {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    // Get file size and allocate buffer
    const stat = try file.stat();
    const data = try allocator.alloc(u8, stat.size);
    errdefer allocator.free(data);

    // Read entire file in a loop (Zig 0.16 File.read returns bytes per call)
    var total_read: usize = 0;
    while (total_read < stat.size) {
        const n = try file.read(data[total_read..]);
        if (n == 0) return error.UnexpectedEOF;
        total_read += n;
    }

    const st = try parse(allocator, data);

    return .{ .st = st, .data = data };
}

// ============================================================================
// Tests
// ============================================================================

test "parse minimal safetensors" {
    // Minimal valid safetensors: empty header with no tensors
    const header = "{}";
    const header_len: u64 = header.len;

    var data: [8 + header.len]u8 = undefined;
    std.mem.writeInt(u64, data[0..8], header_len, .little);
    @memcpy(data[8..], header);

    var st = try parse(std.testing.allocator, &data);
    defer st.deinit();

    try std.testing.expectEqual(@as(usize, 0), st.tensors.len);
}

test "parse safetensors with tensor" {
    // Header with one tensor (padded to 72 bytes for 4-byte alignment of tensor data)
    const header =
        \\{"test": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]}    }
    ;
    comptime std.debug.assert((8 + header.len) % 4 == 0); // Ensure alignment
    const header_len: u64 = header.len;

    // 8 bytes header size + header + 24 bytes tensor data
    // Align for f32 access since tensor data starts at offset 80 (divisible by 4)
    var data: [8 + header.len + 24]u8 align(@alignOf(f32)) = undefined;
    std.mem.writeInt(u64, data[0..8], header_len, .little);
    @memcpy(data[8..][0..header.len], header);

    // Fill tensor data with test values: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    const tensor_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    @memcpy(data[8 + header.len ..][0..24], std.mem.asBytes(&tensor_data));

    var st = try parse(std.testing.allocator, &data);
    defer st.deinit();

    try std.testing.expectEqual(@as(usize, 1), st.tensors.len);

    const info = st.get("test").?;
    try std.testing.expectEqualStrings("test", info.name);
    try std.testing.expectEqual(DType.F32, info.dtype);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 2, 3 }, info.shape);
    try std.testing.expectEqual(@as(usize, 6), info.numel());

    // Get the data
    const values = st.getData(f32, info);
    try std.testing.expectEqual(@as(usize, 6), values.len);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), values[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), values[5], 1e-6);
}

test "dtype byte sizes" {
    try std.testing.expectEqual(@as(usize, 4), DType.F32.byteSize());
    try std.testing.expectEqual(@as(usize, 2), DType.F16.byteSize());
    try std.testing.expectEqual(@as(usize, 8), DType.F64.byteSize());
    try std.testing.expectEqual(@as(usize, 1), DType.U8.byteSize());
}

test "dtype from string" {
    try std.testing.expectEqual(DType.F32, DType.fromString("F32").?);
    try std.testing.expectEqual(DType.BF16, DType.fromString("BF16").?);
    try std.testing.expect(DType.fromString("INVALID") == null);
}
