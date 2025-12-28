//! Minimal Protocol Buffers decoder for ONNX.
//!
//! ONNX uses protobuf3 for serialization. This module implements just enough
//! of the protobuf wire format to parse ONNX models without external dependencies.
//!
//! Wire format reference: https://protobuf.dev/programming-guides/encoding/

const std = @import("std");

/// Wire types in protobuf encoding.
pub const WireType = enum(u3) {
    varint = 0, // int32, int64, uint32, uint64, sint32, sint64, bool, enum
    fixed64 = 1, // fixed64, sfixed64, double
    length_delimited = 2, // string, bytes, embedded messages, packed repeated
    start_group = 3, // deprecated
    end_group = 4, // deprecated
    fixed32 = 5, // fixed32, sfixed32, float
};

/// Decoded field header containing field number and wire type.
pub const FieldHeader = struct {
    field_number: u32,
    wire_type: WireType,
};

/// Protobuf decoder that reads from a byte slice.
pub const Decoder = struct {
    data: []const u8,
    pos: usize,

    pub fn init(data: []const u8) Decoder {
        return .{ .data = data, .pos = 0 };
    }

    /// Returns true if there's more data to read.
    pub fn hasMore(self: *const Decoder) bool {
        return self.pos < self.data.len;
    }

    /// Returns remaining bytes.
    pub fn remaining(self: *const Decoder) []const u8 {
        return self.data[self.pos..];
    }

    /// Reads a single byte.
    pub fn readByte(self: *Decoder) !u8 {
        if (self.pos >= self.data.len) return error.EndOfStream;
        const byte = self.data[self.pos];
        self.pos += 1;
        return byte;
    }

    /// Reads exactly `n` bytes.
    pub fn readBytes(self: *Decoder, n: usize) ![]const u8 {
        if (self.pos + n > self.data.len) return error.EndOfStream;
        const bytes = self.data[self.pos..][0..n];
        self.pos += n;
        return bytes;
    }

    /// Skips `n` bytes.
    pub fn skip(self: *Decoder, n: usize) !void {
        if (self.pos + n > self.data.len) return error.EndOfStream;
        self.pos += n;
    }

    // ========================================================================
    // Varint decoding (LEB128)
    // ========================================================================

    /// Reads a varint-encoded unsigned 64-bit integer.
    /// Varints use 7 bits per byte with MSB as continuation flag.
    pub fn readVarint(self: *Decoder) !u64 {
        var result: u64 = 0;
        var shift: u6 = 0;

        while (true) {
            const byte = try self.readByte();
            const value: u64 = @intCast(byte & 0x7F);
            result |= value << shift;

            // MSB = 0 means this is the last byte
            if (byte & 0x80 == 0) break;

            shift += 7;
            if (shift >= 64) return error.VarintTooLong;
        }

        return result;
    }

    /// Reads a varint as u32.
    pub fn readVarintU32(self: *Decoder) !u32 {
        const value = try self.readVarint();
        if (value > std.math.maxInt(u32)) return error.VarintOverflow;
        return @intCast(value);
    }

    /// Reads a varint as i32 (zigzag decoded for sint32).
    pub fn readVarintI32(self: *Decoder) !i32 {
        const value = try self.readVarintU32();
        return @bitCast(value);
    }

    /// Reads a varint as i64.
    pub fn readVarintI64(self: *Decoder) !i64 {
        const value = try self.readVarint();
        return @bitCast(value);
    }

    /// Reads a zigzag-encoded signed integer (sint32/sint64).
    pub fn readSint64(self: *Decoder) !i64 {
        const n = try self.readVarint();
        // Zigzag decode: (n >> 1) ^ -(n & 1)
        return @bitCast((n >> 1) ^ (0 -% (n & 1)));
    }

    /// Reads a zigzag-encoded sint32.
    pub fn readSint32(self: *Decoder) !i32 {
        const value = try self.readSint64();
        if (value < std.math.minInt(i32) or value > std.math.maxInt(i32)) {
            return error.VarintOverflow;
        }
        return @intCast(value);
    }

    // ========================================================================
    // Fixed-width types
    // ========================================================================

    /// Reads a fixed 32-bit value (little-endian).
    pub fn readFixed32(self: *Decoder) !u32 {
        const bytes = try self.readBytes(4);
        return std.mem.readInt(u32, bytes[0..4], .little);
    }

    /// Reads a fixed 64-bit value (little-endian).
    pub fn readFixed64(self: *Decoder) !u64 {
        const bytes = try self.readBytes(8);
        return std.mem.readInt(u64, bytes[0..8], .little);
    }

    /// Reads a 32-bit float.
    pub fn readFloat(self: *Decoder) !f32 {
        const bits = try self.readFixed32();
        return @bitCast(bits);
    }

    /// Reads a 64-bit double.
    pub fn readDouble(self: *Decoder) !f64 {
        const bits = try self.readFixed64();
        return @bitCast(bits);
    }

    // ========================================================================
    // Length-delimited types
    // ========================================================================

    /// Reads a length-prefixed byte slice (for strings, bytes, embedded messages).
    pub fn readLengthDelimited(self: *Decoder) ![]const u8 {
        const len = try self.readVarint();
        if (len > std.math.maxInt(usize)) return error.LengthTooLarge;
        return try self.readBytes(@intCast(len));
    }

    /// Reads a UTF-8 string.
    pub fn readString(self: *Decoder) ![]const u8 {
        return try self.readLengthDelimited();
    }

    // ========================================================================
    // Field parsing
    // ========================================================================

    /// Reads a field header (field number + wire type).
    pub fn readFieldHeader(self: *Decoder) !FieldHeader {
        const tag = try self.readVarint();
        const wire_type: u3 = @intCast(tag & 0x07);
        const field_number: u32 = @intCast(tag >> 3);

        if (field_number == 0) return error.InvalidFieldNumber;

        return .{
            .field_number = field_number,
            .wire_type = @enumFromInt(wire_type),
        };
    }

    /// Skips a field value based on wire type.
    pub fn skipField(self: *Decoder, wire_type: WireType) !void {
        switch (wire_type) {
            .varint => _ = try self.readVarint(),
            .fixed64 => try self.skip(8),
            .length_delimited => {
                const len = try self.readVarint();
                try self.skip(@intCast(len));
            },
            .fixed32 => try self.skip(4),
            .start_group, .end_group => return error.GroupsNotSupported,
        }
    }

    // ========================================================================
    // Packed repeated fields
    // ========================================================================

    /// Iterator for packed repeated varints.
    pub fn PackedVarintIterator(comptime T: type) type {
        return struct {
            decoder: Decoder,

            pub fn next(self: *@This()) !?T {
                if (!self.decoder.hasMore()) return null;
                const value = try self.decoder.readVarint();
                return @intCast(value);
            }
        };
    }

    /// Creates an iterator for packed repeated varints.
    pub fn packedVarints(self: *Decoder, comptime T: type) !PackedVarintIterator(T) {
        const data = try self.readLengthDelimited();
        return .{ .decoder = Decoder.init(data) };
    }

    /// Iterator for packed repeated fixed32 values.
    pub fn PackedFixed32Iterator(comptime T: type) type {
        return struct {
            data: []const u8,
            pos: usize,

            pub fn next(self: *@This()) ?T {
                if (self.pos + 4 > self.data.len) return null;
                const bytes = self.data[self.pos..][0..4];
                self.pos += 4;
                const bits = std.mem.readInt(u32, bytes, .little);
                return @bitCast(bits);
            }
        };
    }

    /// Creates an iterator for packed repeated fixed32/float values.
    pub fn packedFixed32(self: *Decoder, comptime T: type) !PackedFixed32Iterator(T) {
        const data = try self.readLengthDelimited();
        return .{ .data = data, .pos = 0 };
    }

    /// Iterator for packed repeated fixed64 values.
    pub fn PackedFixed64Iterator(comptime T: type) type {
        return struct {
            data: []const u8,
            pos: usize,

            pub fn next(self: *@This()) ?T {
                if (self.pos + 8 > self.data.len) return null;
                const bytes = self.data[self.pos..][0..8];
                self.pos += 8;
                const bits = std.mem.readInt(u64, bytes, .little);
                return @bitCast(bits);
            }
        };
    }

    /// Creates an iterator for packed repeated fixed64/double values.
    pub fn packedFixed64(self: *Decoder, comptime T: type) !PackedFixed64Iterator(T) {
        const data = try self.readLengthDelimited();
        return .{ .data = data, .pos = 0 };
    }

    // ========================================================================
    // Utility: collect packed into slice
    // ========================================================================

    /// Reads packed repeated varints into an allocated slice.
    pub fn readPackedVarints(self: *Decoder, comptime T: type, allocator: std.mem.Allocator) ![]T {
        const data = try self.readLengthDelimited();
        var sub = Decoder.init(data);

        // Count elements first
        var count: usize = 0;
        while (sub.hasMore()) {
            _ = try sub.readVarint();
            count += 1;
        }

        // Allocate and fill
        const result = try allocator.alloc(T, count);
        sub = Decoder.init(data);
        for (result) |*item| {
            const value = try sub.readVarint();
            item.* = @intCast(value);
        }

        return result;
    }

    /// Reads packed repeated floats into an allocated slice.
    pub fn readPackedFloats(self: *Decoder, allocator: std.mem.Allocator) ![]f32 {
        const data = try self.readLengthDelimited();
        if (data.len % 4 != 0) return error.InvalidPackedData;

        const count = data.len / 4;
        const result = try allocator.alloc(f32, count);

        for (result, 0..) |*item, i| {
            const bytes = data[i * 4 ..][0..4];
            const bits = std.mem.readInt(u32, bytes, .little);
            item.* = @bitCast(bits);
        }

        return result;
    }

    /// Reads packed repeated doubles into an allocated slice.
    pub fn readPackedDoubles(self: *Decoder, allocator: std.mem.Allocator) ![]f64 {
        const data = try self.readLengthDelimited();
        if (data.len % 8 != 0) return error.InvalidPackedData;

        const count = data.len / 8;
        const result = try allocator.alloc(f64, count);

        for (result, 0..) |*item, i| {
            const bytes = data[i * 8 ..][0..8];
            const bits = std.mem.readInt(u64, bytes, .little);
            item.* = @bitCast(bits);
        }

        return result;
    }

    /// Reads packed repeated int32s into an allocated slice.
    pub fn readPackedInt32s(self: *Decoder, allocator: std.mem.Allocator) ![]i32 {
        const data = try self.readLengthDelimited();
        if (data.len % 4 != 0) return error.InvalidPackedData;

        const count = data.len / 4;
        const result = try allocator.alloc(i32, count);

        for (result, 0..) |*item, i| {
            const bytes = data[i * 4 ..][0..4];
            item.* = std.mem.readInt(i32, bytes, .little);
        }

        return result;
    }

    /// Reads packed repeated int64s into an allocated slice.
    pub fn readPackedInt64s(self: *Decoder, allocator: std.mem.Allocator) ![]i64 {
        const data = try self.readLengthDelimited();
        var sub = Decoder.init(data);

        // Count elements first
        var count: usize = 0;
        while (sub.hasMore()) {
            _ = try sub.readVarint();
            count += 1;
        }

        // Allocate and fill
        const result = try allocator.alloc(i64, count);
        sub = Decoder.init(data);
        for (result) |*item| {
            item.* = try sub.readVarintI64();
        }

        return result;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "varint decoding" {
    // Single byte
    {
        var dec = Decoder.init(&[_]u8{0x01});
        try std.testing.expectEqual(@as(u64, 1), try dec.readVarint());
    }

    // 127 (max single byte)
    {
        var dec = Decoder.init(&[_]u8{0x7F});
        try std.testing.expectEqual(@as(u64, 127), try dec.readVarint());
    }

    // 128 (two bytes)
    {
        var dec = Decoder.init(&[_]u8{ 0x80, 0x01 });
        try std.testing.expectEqual(@as(u64, 128), try dec.readVarint());
    }

    // 300 = 0b100101100 = 0xAC 0x02
    {
        var dec = Decoder.init(&[_]u8{ 0xAC, 0x02 });
        try std.testing.expectEqual(@as(u64, 300), try dec.readVarint());
    }

    // Large value: 150 = 0x96 0x01
    {
        var dec = Decoder.init(&[_]u8{ 0x96, 0x01 });
        try std.testing.expectEqual(@as(u64, 150), try dec.readVarint());
    }
}

test "zigzag decoding" {
    // sint32/sint64 use zigzag encoding
    // 0 -> 0, -1 -> 1, 1 -> 2, -2 -> 3, 2 -> 4, ...
    {
        var dec = Decoder.init(&[_]u8{0x00});
        try std.testing.expectEqual(@as(i64, 0), try dec.readSint64());
    }
    {
        var dec = Decoder.init(&[_]u8{0x01});
        try std.testing.expectEqual(@as(i64, -1), try dec.readSint64());
    }
    {
        var dec = Decoder.init(&[_]u8{0x02});
        try std.testing.expectEqual(@as(i64, 1), try dec.readSint64());
    }
    {
        var dec = Decoder.init(&[_]u8{0x03});
        try std.testing.expectEqual(@as(i64, -2), try dec.readSint64());
    }
}

test "fixed32 decoding" {
    // Little-endian 0x12345678
    var dec = Decoder.init(&[_]u8{ 0x78, 0x56, 0x34, 0x12 });
    try std.testing.expectEqual(@as(u32, 0x12345678), try dec.readFixed32());
}

test "float decoding" {
    // 1.0f = 0x3F800000
    var dec = Decoder.init(&[_]u8{ 0x00, 0x00, 0x80, 0x3F });
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), try dec.readFloat(), 1e-6);
}

test "field header decoding" {
    // Field 1, wire type 0 (varint): tag = (1 << 3) | 0 = 0x08
    {
        var dec = Decoder.init(&[_]u8{0x08});
        const header = try dec.readFieldHeader();
        try std.testing.expectEqual(@as(u32, 1), header.field_number);
        try std.testing.expectEqual(WireType.varint, header.wire_type);
    }

    // Field 2, wire type 2 (length-delimited): tag = (2 << 3) | 2 = 0x12
    {
        var dec = Decoder.init(&[_]u8{0x12});
        const header = try dec.readFieldHeader();
        try std.testing.expectEqual(@as(u32, 2), header.field_number);
        try std.testing.expectEqual(WireType.length_delimited, header.wire_type);
    }

    // Field 150, wire type 0: tag = (150 << 3) | 0 = 1200 = 0xB0 0x09
    {
        var dec = Decoder.init(&[_]u8{ 0xB0, 0x09 });
        const header = try dec.readFieldHeader();
        try std.testing.expectEqual(@as(u32, 150), header.field_number);
        try std.testing.expectEqual(WireType.varint, header.wire_type);
    }
}

test "length-delimited decoding" {
    // Length 5, then "hello"
    var dec = Decoder.init(&[_]u8{ 0x05, 'h', 'e', 'l', 'l', 'o' });
    const str = try dec.readLengthDelimited();
    try std.testing.expectEqualStrings("hello", str);
}

test "skip field" {
    // Skip varint field
    {
        var dec = Decoder.init(&[_]u8{ 0x96, 0x01, 0xFF }); // varint 150, then 0xFF
        try dec.skipField(.varint);
        try std.testing.expectEqual(@as(u8, 0xFF), try dec.readByte());
    }

    // Skip length-delimited field
    {
        var dec = Decoder.init(&[_]u8{ 0x03, 'a', 'b', 'c', 0xFF }); // len 3, "abc", then 0xFF
        try dec.skipField(.length_delimited);
        try std.testing.expectEqual(@as(u8, 0xFF), try dec.readByte());
    }

    // Skip fixed32
    {
        var dec = Decoder.init(&[_]u8{ 0x01, 0x02, 0x03, 0x04, 0xFF });
        try dec.skipField(.fixed32);
        try std.testing.expectEqual(@as(u8, 0xFF), try dec.readByte());
    }
}

test "packed repeated varints" {
    // Packed repeated int32: length 3, values [1, 150, 2]
    // 150 = 0x96 0x01 (2 bytes varint)
    var dec = Decoder.init(&[_]u8{ 0x04, 0x01, 0x96, 0x01, 0x02 });
    const values = try dec.readPackedVarints(i32, std.testing.allocator);
    defer std.testing.allocator.free(values);

    try std.testing.expectEqual(@as(usize, 3), values.len);
    try std.testing.expectEqual(@as(i32, 1), values[0]);
    try std.testing.expectEqual(@as(i32, 150), values[1]);
    try std.testing.expectEqual(@as(i32, 2), values[2]);
}

test "packed repeated floats" {
    // Two floats: 1.0 and 2.0
    // 1.0f = 0x3F800000, 2.0f = 0x40000000
    var dec = Decoder.init(&[_]u8{
        0x08, // length 8
        0x00, 0x00, 0x80, 0x3F, // 1.0
        0x00, 0x00, 0x00, 0x40, // 2.0
    });
    const values = try dec.readPackedFloats(std.testing.allocator);
    defer std.testing.allocator.free(values);

    try std.testing.expectEqual(@as(usize, 2), values.len);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), values[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), values[1], 1e-6);
}

test "multiple fields" {
    // Message with field 1 (varint) = 150, field 2 (string) = "testing"
    const data = [_]u8{
        0x08, 0x96, 0x01, // field 1, varint 150
        0x12, 0x07, 't', 'e', 's', 't', 'i', 'n', 'g', // field 2, string "testing"
    };
    var dec = Decoder.init(&data);

    // Read field 1
    const h1 = try dec.readFieldHeader();
    try std.testing.expectEqual(@as(u32, 1), h1.field_number);
    try std.testing.expectEqual(WireType.varint, h1.wire_type);
    const v1 = try dec.readVarint();
    try std.testing.expectEqual(@as(u64, 150), v1);

    // Read field 2
    const h2 = try dec.readFieldHeader();
    try std.testing.expectEqual(@as(u32, 2), h2.field_number);
    try std.testing.expectEqual(WireType.length_delimited, h2.wire_type);
    const v2 = try dec.readString();
    try std.testing.expectEqualStrings("testing", v2);

    try std.testing.expect(!dec.hasMore());
}
