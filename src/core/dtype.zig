//! Data type definitions for tensor elements.
//!
//! Provides a DType enum representing all supported numeric types,
//! along with comptime utilities for type introspection and conversion.

const std = @import("std");

/// Supported data types for tensor elements.
pub const DType = enum {
    // Floating point types
    f16,
    bf16,
    f32,
    f64,

    // Signed integer types
    i8,
    i16,
    i32,
    i64,

    // Unsigned integer types
    u8,
    u16,
    u32,
    u64,

    /// Returns the size in bytes of this data type.
    pub fn sizeOf(comptime self: DType) comptime_int {
        return @sizeOf(self.ZigType());
    }

    /// Returns the alignment in bytes of this data type.
    pub fn alignOf(comptime self: DType) comptime_int {
        return @alignOf(self.ZigType());
    }

    /// Converts the DType enum to the corresponding Zig type.
    pub fn ZigType(comptime self: DType) type {
        return switch (self) {
            .f16 => f16,
            .bf16 => @Vector(1, u16), // bf16 represented as u16 bits, need special handling
            .f32 => f32,
            .f64 => f64,
            .i8 => i8,
            .i16 => i16,
            .i32 => i32,
            .i64 => i64,
            .u8 => u8,
            .u16 => u16,
            .u32 => u32,
            .u64 => u64,
        };
    }

    /// Returns true if this is a floating-point type.
    pub fn isFloat(comptime self: DType) bool {
        return switch (self) {
            .f16, .bf16, .f32, .f64 => true,
            else => false,
        };
    }

    /// Returns true if this is a signed integer type.
    pub fn isSigned(comptime self: DType) bool {
        return switch (self) {
            .i8, .i16, .i32, .i64 => true,
            else => false,
        };
    }

    /// Returns true if this is an unsigned integer type.
    pub fn isUnsigned(comptime self: DType) bool {
        return switch (self) {
            .u8, .u16, .u32, .u64 => true,
            else => false,
        };
    }

    /// Returns true if this is any integer type (signed or unsigned).
    pub fn isInteger(comptime self: DType) bool {
        return self.isSigned() or self.isUnsigned();
    }

    /// Returns the number of bits in this data type.
    pub fn bits(comptime self: DType) comptime_int {
        return self.sizeOf() * 8;
    }

    /// Gets the DType corresponding to a Zig type.
    pub fn fromZigType(comptime T: type) ?DType {
        return switch (T) {
            f16 => .f16,
            f32 => .f32,
            f64 => .f64,
            i8 => .i8,
            i16 => .i16,
            i32 => .i32,
            i64 => .i64,
            u8 => .u8,
            u16 => .u16,
            u32 => .u32,
            u64 => .u64,
            else => null,
        };
    }

    /// Returns a human-readable name for this dtype.
    pub fn name(comptime self: DType) []const u8 {
        return @tagName(self);
    }
};

/// Check if a Zig type is a supported tensor element type.
pub fn isSupportedType(comptime T: type) bool {
    return DType.fromZigType(T) != null;
}

/// Get the appropriate epsilon value for floating-point comparisons.
pub fn epsilon(comptime T: type) T {
    return switch (T) {
        f16 => 9.77e-4, // 2^-10
        f32 => 1.19209290e-7, // 2^-23
        f64 => 2.2204460492503131e-16, // 2^-52
        else => @compileError("epsilon only defined for floating-point types"),
    };
}

/// Machine precision (unit roundoff) for floating-point types.
pub fn machinePrecision(comptime T: type) T {
    return switch (T) {
        f16 => 4.88e-4, // 2^-11
        f32 => 5.96046448e-8, // 2^-24
        f64 => 1.1102230246251565e-16, // 2^-53
        else => @compileError("machinePrecision only defined for floating-point types"),
    };
}

// ============================================================================
// Tests
// ============================================================================

test "DType size and alignment" {
    try std.testing.expectEqual(@as(comptime_int, 2), DType.f16.sizeOf());
    try std.testing.expectEqual(@as(comptime_int, 4), DType.f32.sizeOf());
    try std.testing.expectEqual(@as(comptime_int, 8), DType.f64.sizeOf());
    try std.testing.expectEqual(@as(comptime_int, 1), DType.i8.sizeOf());
    try std.testing.expectEqual(@as(comptime_int, 2), DType.i16.sizeOf());
    try std.testing.expectEqual(@as(comptime_int, 4), DType.i32.sizeOf());
    try std.testing.expectEqual(@as(comptime_int, 8), DType.i64.sizeOf());

    // Alignment should match size for primitive types
    try std.testing.expectEqual(@as(comptime_int, 4), DType.f32.alignOf());
    try std.testing.expectEqual(@as(comptime_int, 8), DType.f64.alignOf());
}

test "DType ZigType conversion" {
    try std.testing.expectEqual(f32, DType.f32.ZigType());
    try std.testing.expectEqual(f64, DType.f64.ZigType());
    try std.testing.expectEqual(i32, DType.i32.ZigType());
    try std.testing.expectEqual(u8, DType.u8.ZigType());
}

test "DType category predicates" {
    // Float types
    try std.testing.expect(DType.f16.isFloat());
    try std.testing.expect(DType.f32.isFloat());
    try std.testing.expect(DType.f64.isFloat());
    try std.testing.expect(!DType.i32.isFloat());

    // Signed integers
    try std.testing.expect(DType.i8.isSigned());
    try std.testing.expect(DType.i32.isSigned());
    try std.testing.expect(!DType.u32.isSigned());
    try std.testing.expect(!DType.f32.isSigned());

    // Unsigned integers
    try std.testing.expect(DType.u8.isUnsigned());
    try std.testing.expect(DType.u64.isUnsigned());
    try std.testing.expect(!DType.i32.isUnsigned());

    // Integer predicate
    try std.testing.expect(DType.i32.isInteger());
    try std.testing.expect(DType.u32.isInteger());
    try std.testing.expect(!DType.f32.isInteger());
}

test "DType bits" {
    try std.testing.expectEqual(@as(comptime_int, 16), DType.f16.bits());
    try std.testing.expectEqual(@as(comptime_int, 32), DType.f32.bits());
    try std.testing.expectEqual(@as(comptime_int, 64), DType.f64.bits());
    try std.testing.expectEqual(@as(comptime_int, 8), DType.i8.bits());
}

test "DType fromZigType" {
    try std.testing.expectEqual(DType.f32, DType.fromZigType(f32).?);
    try std.testing.expectEqual(DType.i64, DType.fromZigType(i64).?);
    try std.testing.expectEqual(@as(?DType, null), DType.fromZigType(bool));
    try std.testing.expectEqual(@as(?DType, null), DType.fromZigType([]u8));
}

test "isSupportedType" {
    try std.testing.expect(isSupportedType(f32));
    try std.testing.expect(isSupportedType(i64));
    try std.testing.expect(!isSupportedType(bool));
    try std.testing.expect(!isSupportedType(void));
}

test "epsilon values" {
    // Epsilon should be small but nonzero
    try std.testing.expect(epsilon(f32) > 0);
    try std.testing.expect(epsilon(f32) < 1e-5);
    try std.testing.expect(epsilon(f64) > 0);
    try std.testing.expect(epsilon(f64) < 1e-14);
}
