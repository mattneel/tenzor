//! Custom .tenzor binary format with mmap support.
//!
//! Format layout:
//!   [Header: 64 bytes]
//!   [Metadata: variable JSON, padded to 8-byte boundary]
//!   [Tensor Index: 48 bytes × N tensors]
//!   [Padding to 4KB page alignment]
//!   [Tensor Data: contiguous, page-aligned]
//!
//! Design goals:
//!   - Mmap-friendly: tensor data is page-aligned for zero-copy access
//!   - Fast lookup: tensor names are hashed for O(1) access
//!   - Self-contained: model + optimizer state + metadata in one file
//!   - Efficient: <1ms load time via mmap (no parsing, no copying)

const std = @import("std");
const posix = std.posix;

const PAGE_SIZE: usize = 4096;

/// Magic bytes: "TENZOR\x00\x00"
const MAGIC: [8]u8 = .{ 'T', 'E', 'N', 'Z', 'O', 'R', 0, 0 };
const VERSION: u32 = 1;

/// Supported data types.
pub const DType = enum(u8) {
    f32 = 0,
    f16 = 1,
    bf16 = 2,
    f64 = 3,
    i8 = 4,
    i16 = 5,
    i32 = 6,
    i64 = 7,
    u8 = 8,
    u16 = 9,
    u32 = 10,
    u64 = 11,

    pub fn byteSize(self: DType) usize {
        return switch (self) {
            .f16, .bf16, .i16, .u16 => 2,
            .f32, .i32, .u32 => 4,
            .f64, .i64, .u64 => 8,
            .i8, .u8 => 1,
        };
    }

    pub fn toString(self: DType) []const u8 {
        return switch (self) {
            .f32 => "f32",
            .f16 => "f16",
            .bf16 => "bf16",
            .f64 => "f64",
            .i8 => "i8",
            .i16 => "i16",
            .i32 => "i32",
            .i64 => "i64",
            .u8 => "u8",
            .u16 => "u16",
            .u32 => "u32",
            .u64 => "u64",
        };
    }
};

/// File header (64 bytes, fixed layout).
pub const Header = extern struct {
    magic: [8]u8 = MAGIC,
    version: u32 = VERSION,
    flags: u32 = 0,
    tensor_count: u32 = 0,
    _pad0: u32 = 0,
    index_offset: u64 = 0,
    data_offset: u64 = 0,
    metadata_size: u32 = 0,
    _reserved: [20]u8 = .{0} ** 20,

    comptime {
        std.debug.assert(@sizeOf(Header) == 64);
    }
};

/// Tensor index entry (48 bytes per tensor).
pub const TensorEntry = extern struct {
    name_hash: u64, // FNV-1a hash of tensor name
    dtype: DType,
    ndim: u8,
    _pad: [2]u8 = .{ 0, 0 },
    shape: [5]u32 = .{0} ** 5, // Up to 5 dimensions
    data_offset: u64, // Offset from start of data section
    data_size: u64, // Size in bytes

    comptime {
        std.debug.assert(@sizeOf(TensorEntry) == 48);
    }

    pub fn numel(self: TensorEntry) usize {
        if (self.ndim == 0) return 1;
        var n: usize = 1;
        for (self.shape[0..self.ndim]) |d| n *= d;
        return n;
    }

    pub fn getShape(self: TensorEntry) []const u32 {
        return self.shape[0..self.ndim];
    }
};

/// FNV-1a hash for tensor names.
pub fn hashName(name: []const u8) u64 {
    var hash: u64 = 0xcbf29ce484222325; // FNV offset basis
    for (name) |byte| {
        hash ^= byte;
        hash *%= 0x100000001b3; // FNV prime
    }
    return hash;
}

/// Memory-mapped .tenzor file for reading.
pub const TenzorFile = struct {
    mapping: []align(PAGE_SIZE) const u8,
    fd: posix.fd_t,
    header: *const Header,
    index: []const TensorEntry,
    metadata_json: []const u8,
    data_base: [*]const u8,

    // Name lookup cache (built lazily)
    allocator: ?std.mem.Allocator,
    name_map: ?std.AutoHashMap(u64, usize),

    pub const OpenError = error{
        InvalidMagic,
        UnsupportedVersion,
        InvalidFormat,
        MmapFailed,
    } || std.fs.File.OpenError || posix.MMapError;

    /// Open and mmap a .tenzor file.
    pub fn open(allocator: std.mem.Allocator, path: []const u8) OpenError!TenzorFile {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        const fd = file.handle;
        const stat = file.stat() catch return error.InvalidFormat;
        const file_size = stat.size;

        if (file_size < @sizeOf(Header)) {
            return error.InvalidFormat;
        }

        // Memory map the entire file
        const mapping = posix.mmap(
            null,
            file_size,
            posix.PROT.READ,
            .{ .TYPE = .PRIVATE },
            fd,
            0,
        ) catch return error.MmapFailed;

        const aligned_mapping: []align(PAGE_SIZE) const u8 = @alignCast(mapping);

        // Parse header
        const header: *const Header = @ptrCast(aligned_mapping.ptr);

        if (!std.mem.eql(u8, &header.magic, &MAGIC)) {
            posix.munmap(aligned_mapping);
            return error.InvalidMagic;
        }

        if (header.version != VERSION) {
            posix.munmap(aligned_mapping);
            return error.UnsupportedVersion;
        }

        // Get metadata JSON slice
        const metadata_start = @sizeOf(Header);
        const metadata_json = aligned_mapping[metadata_start..][0..header.metadata_size];

        // Get tensor index
        const index_ptr: [*]const TensorEntry = @ptrCast(@alignCast(aligned_mapping.ptr + header.index_offset));
        const index = index_ptr[0..header.tensor_count];

        // Data base pointer
        const data_base = aligned_mapping.ptr + header.data_offset;

        return .{
            .mapping = aligned_mapping,
            .fd = fd,
            .header = header,
            .index = index,
            .metadata_json = metadata_json,
            .data_base = data_base,
            .allocator = allocator,
            .name_map = null,
        };
    }

    /// Close the file and unmap memory.
    pub fn close(self: *TenzorFile) void {
        if (self.name_map) |*m| m.deinit();
        posix.munmap(self.mapping);
    }

    /// Get tensor entry by name hash (O(n) scan, or O(1) if name map built).
    pub fn getTensorByHash(self: *TenzorFile, hash: u64) ?*const TensorEntry {
        // Use name map if available
        if (self.name_map) |m| {
            if (m.get(hash)) |idx| {
                return &self.index[idx];
            }
            return null;
        }

        // Linear scan
        for (self.index) |*entry| {
            if (entry.name_hash == hash) return entry;
        }
        return null;
    }

    /// Get tensor entry by name.
    pub fn getTensor(self: *TenzorFile, name: []const u8) ?*const TensorEntry {
        return self.getTensorByHash(hashName(name));
    }

    /// Get tensor data as typed slice (zero-copy from mmap).
    pub fn getData(self: *const TenzorFile, comptime T: type, entry: *const TensorEntry) []const T {
        const byte_ptr = self.data_base + entry.data_offset;
        const byte_slice = byte_ptr[0..entry.data_size];
        const aligned: []align(@alignOf(T)) const u8 = @alignCast(byte_slice);
        return std.mem.bytesAsSlice(T, aligned);
    }

    /// Get tensor data as f32 slice.
    pub fn getF32(self: *TenzorFile, name: []const u8) ?[]const f32 {
        const entry = self.getTensor(name) orelse return null;
        if (entry.dtype != .f32) return null;
        return self.getData(f32, entry);
    }

    /// Build name lookup map for O(1) access (optional optimization).
    pub fn buildNameMap(self: *TenzorFile) !void {
        if (self.name_map != null) return;
        if (self.allocator == null) return error.NoAllocator;

        var map = std.AutoHashMap(u64, usize).init(self.allocator.?);
        errdefer map.deinit();

        for (self.index, 0..) |entry, i| {
            try map.put(entry.name_hash, i);
        }

        self.name_map = map;
    }

    /// Parse metadata JSON.
    pub fn parseMetadata(self: *const TenzorFile, allocator: std.mem.Allocator) !std.json.Parsed(std.json.Value) {
        return std.json.parseFromSlice(std.json.Value, allocator, self.metadata_json, .{});
    }

    /// Get string from metadata.
    pub fn getMetadataString(self: *const TenzorFile, allocator: std.mem.Allocator, key: []const u8) !?[]const u8 {
        var parsed = try self.parseMetadata(allocator);
        defer parsed.deinit();

        if (parsed.value != .object) return null;
        const val = parsed.value.object.get(key) orelse return null;
        if (val != .string) return null;
        return try allocator.dupe(u8, val.string);
    }
};

/// Writer for creating .tenzor files.
pub const TenzorWriter = struct {
    allocator: std.mem.Allocator,
    file: std.fs.File,
    tensors: std.ArrayList(TensorInfo),
    data_buffer: std.ArrayList(u8),
    metadata_json: []const u8,

    const TensorInfo = struct {
        name: []const u8,
        hash: u64,
        dtype: DType,
        shape: []const usize,
        data: []const u8,
    };

    pub fn create(allocator: std.mem.Allocator, path: []const u8) !TenzorWriter {
        const file = try std.fs.cwd().createFile(path, .{});
        return .{
            .allocator = allocator,
            .file = file,
            .tensors = .empty,
            .data_buffer = .empty,
            .metadata_json = "{}",
        };
    }

    pub fn deinit(self: *TenzorWriter) void {
        // Free tensor info copies
        for (self.tensors.items) |t| {
            self.allocator.free(t.name);
            self.allocator.free(t.shape);
        }
        self.tensors.deinit(self.allocator);
        self.data_buffer.deinit(self.allocator);
        self.file.close();
    }

    /// Add a tensor to the file.
    pub fn addTensor(
        self: *TenzorWriter,
        name: []const u8,
        dtype: DType,
        shape: []const usize,
        data: []const u8,
    ) !void {
        // Verify data size matches shape
        var expected_numel: usize = 1;
        for (shape) |d| expected_numel *= d;
        const expected_bytes = expected_numel * dtype.byteSize();
        if (data.len != expected_bytes) {
            return error.DataSizeMismatch;
        }

        // Copy name and shape
        const name_copy = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(name_copy);

        const shape_copy = try self.allocator.dupe(usize, shape);
        errdefer self.allocator.free(shape_copy);

        try self.tensors.append(self.allocator, .{
            .name = name_copy,
            .hash = hashName(name),
            .dtype = dtype,
            .shape = shape_copy,
            .data = data,
        });
    }

    /// Add f32 tensor (convenience).
    pub fn addF32(self: *TenzorWriter, name: []const u8, shape: []const usize, data: []const f32) !void {
        try self.addTensor(name, .f32, shape, std.mem.sliceAsBytes(data));
    }

    /// Set metadata JSON string.
    pub fn setMetadataJson(self: *TenzorWriter, json: []const u8) void {
        self.metadata_json = json;
    }

    /// Set metadata from a struct (will be serialized to JSON).
    pub fn setMetadata(self: *TenzorWriter, comptime T: type, value: T) !void {
        var list: std.ArrayList(u8) = .empty;
        try std.json.stringify(value, .{}, list.writer(self.allocator));
        self.metadata_json = try list.toOwnedSlice(self.allocator);
    }

    /// Finalize and write the file.
    pub fn finish(self: *TenzorWriter) !void {
        const tensor_count: u32 = @intCast(self.tensors.items.len);

        // Calculate offsets
        const header_size = @sizeOf(Header);
        const metadata_padded = alignUp(self.metadata_json.len, 8);
        const index_offset = header_size + metadata_padded;
        const index_size = tensor_count * @sizeOf(TensorEntry);
        const data_offset_unaligned = index_offset + index_size;
        const data_offset = alignUp(data_offset_unaligned, PAGE_SIZE);

        // Build header
        var header = Header{
            .tensor_count = tensor_count,
            .index_offset = @intCast(index_offset),
            .data_offset = @intCast(data_offset),
            .metadata_size = @intCast(self.metadata_json.len),
        };

        // Write header
        try self.file.writeAll(std.mem.asBytes(&header));

        // Write metadata (padded to 8 bytes)
        try self.file.writeAll(self.metadata_json);
        const metadata_padding = metadata_padded - self.metadata_json.len;
        if (metadata_padding > 0) {
            const zeros: [8]u8 = .{0} ** 8;
            try self.file.writeAll(zeros[0..metadata_padding]);
        }

        // Write tensor index and collect data
        var current_data_offset: u64 = 0;
        for (self.tensors.items) |t| {
            var entry = TensorEntry{
                .name_hash = t.hash,
                .dtype = t.dtype,
                .ndim = @intCast(t.shape.len),
                .data_offset = current_data_offset,
                .data_size = t.data.len,
            };

            // Copy shape
            for (t.shape, 0..) |d, i| {
                if (i >= 5) break;
                entry.shape[i] = @intCast(d);
            }

            try self.file.writeAll(std.mem.asBytes(&entry));
            current_data_offset += t.data.len;
        }

        // Pad to page boundary
        const current_pos = index_offset + index_size;
        const padding_needed = data_offset - current_pos;
        if (padding_needed > 0) {
            const zeros = try self.allocator.alloc(u8, padding_needed);
            defer self.allocator.free(zeros);
            @memset(zeros, 0);
            try self.file.writeAll(zeros);
        }

        // Write tensor data
        for (self.tensors.items) |t| {
            try self.file.writeAll(t.data);
        }
    }
};

fn alignUp(value: usize, alignment: usize) usize {
    return (value + alignment - 1) & ~(alignment - 1);
}

// ============================================================================
// Conversion from SafeTensors
// ============================================================================

const safetensors = @import("safetensors.zig");

/// Convert a SafeTensors file to .tenzor format.
pub fn convertFromSafetensors(
    allocator: std.mem.Allocator,
    input_path: []const u8,
    output_path: []const u8,
) !void {
    // Load safetensors
    const result = try safetensors.load(allocator, input_path);
    defer allocator.free(result.data);
    var st = result.st;
    defer st.deinit();

    // Create tenzor writer
    var writer = try TenzorWriter.create(allocator, output_path);
    defer writer.deinit();

    // Convert each tensor
    for (st.tensors) |tensor| {
        const dtype: DType = switch (tensor.dtype) {
            .F32 => .f32,
            .F16 => .f16,
            .BF16 => .bf16,
            .F64 => .f64,
            .I8 => .i8,
            .I16 => .i16,
            .I32 => .i32,
            .I64 => .i64,
            .U8 => .u8,
            .U16 => .u16,
            .U32 => .u32,
            .U64 => .u64,
            .BOOL => .u8,
        };

        // Get raw data
        const data_start = st.header_size + 8 + tensor.data_start;
        const data_end = st.header_size + 8 + tensor.data_end;
        const raw_data = st.data[data_start..data_end];

        try writer.addTensor(tensor.name, dtype, tensor.shape, raw_data);
    }

    try writer.finish();
}

// ============================================================================
// Tests
// ============================================================================

test "header size" {
    try std.testing.expectEqual(@as(usize, 64), @sizeOf(Header));
}

test "tensor entry size" {
    try std.testing.expectEqual(@as(usize, 48), @sizeOf(TensorEntry));
}

test "hash name" {
    const h1 = hashName("conv1.weight");
    const h2 = hashName("conv1.weight");
    const h3 = hashName("conv1.bias");

    try std.testing.expectEqual(h1, h2);
    try std.testing.expect(h1 != h3);
}

test "dtype byte size" {
    try std.testing.expectEqual(@as(usize, 4), DType.f32.byteSize());
    try std.testing.expectEqual(@as(usize, 2), DType.f16.byteSize());
    try std.testing.expectEqual(@as(usize, 8), DType.f64.byteSize());
}

test "write and read tenzor file" {
    const allocator = std.testing.allocator;

    // Create a test file
    const path = "/tmp/test_tenzor.tenzor";

    // Test data
    const data1 = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const data2 = [_]f32{ 0.1, 0.2, 0.3 };

    // Write
    {
        var writer = try TenzorWriter.create(allocator, path);
        defer writer.deinit();

        try writer.addF32("tensor1", &.{ 2, 3 }, &data1);
        try writer.addF32("tensor2", &.{3}, &data2);
        try writer.finish();
    }

    // Read
    {
        var reader = try TenzorFile.open(allocator, path);
        defer reader.close();

        // Check tensor count
        try std.testing.expectEqual(@as(u32, 2), reader.header.tensor_count);

        // Check tensor1
        const entry1 = reader.getTensor("tensor1").?;
        try std.testing.expectEqual(@as(u8, 2), entry1.ndim);
        try std.testing.expectEqual(@as(u32, 2), entry1.shape[0]);
        try std.testing.expectEqual(@as(u32, 3), entry1.shape[1]);

        const read_data1 = reader.getData(f32, entry1);
        try std.testing.expectEqual(@as(usize, 6), read_data1.len);
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), read_data1[0], 1e-6);
        try std.testing.expectApproxEqAbs(@as(f32, 6.0), read_data1[5], 1e-6);

        // Check tensor2
        const entry2 = reader.getTensor("tensor2").?;
        const read_data2 = reader.getData(f32, entry2);
        try std.testing.expectEqual(@as(usize, 3), read_data2.len);
        try std.testing.expectApproxEqAbs(@as(f32, 0.1), read_data2[0], 1e-6);
    }

    // Clean up
    std.fs.cwd().deleteFile(path) catch {};
}

test "tensor entry numel" {
    var entry = TensorEntry{
        .name_hash = 0,
        .dtype = .f32,
        .ndim = 3,
        .shape = .{ 2, 3, 4, 0, 0 },
        .data_offset = 0,
        .data_size = 96,
    };

    try std.testing.expectEqual(@as(usize, 24), entry.numel());
}
