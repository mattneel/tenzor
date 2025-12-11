//! Core Tensor type with compile-time known shape.
//!
//! The Tensor type is parameterized by element type and shape at compile time,
//! enabling full compile-time validation of operations and zero-cost abstractions.

const std = @import("std");
const dtype_mod = @import("dtype.zig");
const shape_mod = @import("shape.zig");
const strides_mod = @import("strides.zig");
const expr_mod = @import("../ops/expr.zig");

const DType = dtype_mod.DType;
const Shape = shape_mod.Shape;
const MAX_RANK = shape_mod.MAX_RANK;
const NodeKind = expr_mod.NodeKind;

/// Tensor type constructor.
///
/// Creates a tensor type with compile-time known element type and shape.
/// All shape validation happens at compile time.
///
/// Example:
/// ```
/// const Mat = Tensor(f32, .{64, 128});
/// var mat = try Mat.init(allocator);
/// defer mat.deinit();
/// ```
pub fn Tensor(comptime T: type, comptime shape_dims: anytype) type {
    // Validate element type
    if (DType.fromZigType(T) == null) {
        @compileError("Unsupported element type: " ++ @typeName(T) ++
            ". Supported types: f16, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64");
    }

    const ShapeType = Shape(shape_dims);

    return struct {
        /// Node kind for expression graph traversal
        pub const kind: NodeKind = .tensor;
        /// The element type
        pub const ElementType = T;
        /// The shape type
        pub const ShapeT = ShapeType;
        /// Number of dimensions
        pub const ndim: usize = ShapeType.ndim;
        /// Shape dimensions as array
        pub const shape: [ndim]usize = ShapeType.dimensions;
        /// Total number of elements
        pub const numel: usize = ShapeType.numel();
        /// Data type enum value
        pub const dtype: DType = DType.fromZigType(T).?;

        /// Pointer to the underlying data
        data: [*]T,
        /// Allocator used for this tensor
        allocator: std.mem.Allocator,
        /// Whether this tensor owns its data (for views)
        owns_data: bool,
        /// Strides for each dimension
        strides: [ndim]usize,

        const Self = @This();

        /// Initialize a new tensor with uninitialized data.
        pub fn init(allocator: std.mem.Allocator) !Self {
            if (numel == 0) {
                return Self{
                    .data = undefined,
                    .allocator = allocator,
                    .owns_data = true,
                    .strides = ShapeType.strides(),
                };
            }

            const data = try allocator.alloc(T, numel);
            return Self{
                .data = data.ptr,
                .allocator = allocator,
                .owns_data = true,
                .strides = ShapeType.strides(),
            };
        }

        /// Initialize a tensor filled with zeros.
        pub fn zeros(allocator: std.mem.Allocator) !Self {
            var self = try init(allocator);
            self.fill(0);
            return self;
        }

        /// Initialize a tensor filled with ones.
        pub fn ones(allocator: std.mem.Allocator) !Self {
            var self = try init(allocator);
            self.fill(1);
            return self;
        }

        /// Initialize a tensor filled with a constant value.
        pub fn full(allocator: std.mem.Allocator, value: T) !Self {
            var self = try init(allocator);
            self.fill(value);
            return self;
        }

        /// Initialize a tensor from existing data (copies the data).
        pub fn fromSlice(allocator: std.mem.Allocator, data: []const T) !Self {
            if (data.len != numel) {
                return error.SizeMismatch;
            }

            var self = try init(allocator);
            @memcpy(self.slice(), data);
            return self;
        }

        /// Initialize from a multi-dimensional array (comptime).
        pub fn fromArray(allocator: std.mem.Allocator, comptime arr: anytype) !Self {
            const flat = comptime flattenArray(arr);
            if (flat.len != numel) {
                @compileError("Array size does not match tensor shape");
            }

            var self = try init(allocator);
            @memcpy(self.slice(), &flat);
            return self;
        }

        /// Release the tensor's memory.
        pub fn deinit(self: *Self) void {
            if (self.owns_data and numel > 0) {
                self.allocator.free(self.slice());
            }
            self.* = undefined;
        }

        /// Fill the tensor with a constant value.
        pub fn fill(self: *Self, value: T) void {
            @memset(self.slice(), value);
        }

        /// Get the data as a slice.
        pub fn slice(self: *const Self) []T {
            if (numel == 0) return &[_]T{};
            return self.data[0..numel];
        }

        /// Get the data as a const slice.
        pub fn constSlice(self: *const Self) []const T {
            return self.slice();
        }

        /// Get element at flat index.
        pub fn getFlatIndex(self: *const Self, idx: usize) T {
            std.debug.assert(idx < numel);
            return self.data[idx];
        }

        /// Set element at flat index.
        pub fn setFlatIndex(self: *Self, idx: usize, value: T) void {
            std.debug.assert(idx < numel);
            self.data[idx] = value;
        }

        /// Get element at multi-dimensional index.
        pub fn get(self: *const Self, indices: [ndim]usize) T {
            const offset = self.computeOffset(indices);
            return self.data[offset];
        }

        /// Set element at multi-dimensional index.
        pub fn set(self: *Self, indices: [ndim]usize, value: T) void {
            const offset = self.computeOffset(indices);
            self.data[offset] = value;
        }

        /// Compute flat offset from multi-dimensional indices.
        fn computeOffset(self: *const Self, indices: [ndim]usize) usize {
            if (ndim == 0) return 0;

            // Debug bounds checking
            if (std.debug.runtime_safety) {
                for (indices, shape) |idx, dim| {
                    std.debug.assert(idx < dim);
                }
            }

            return strides_mod.flatOffset(ndim, indices, self.strides);
        }

        /// Check if the tensor is contiguous in memory.
        pub fn isContiguous(self: *const Self) bool {
            return strides_mod.isContiguous(ndim, shape, self.strides);
        }

        /// Copy data from another tensor of the same type.
        pub fn copyFrom(self: *Self, other: *const Self) void {
            std.debug.assert(self.isContiguous() and other.isContiguous());
            @memcpy(self.slice(), other.constSlice());
        }

        /// Clone the tensor (allocate new memory and copy data).
        pub fn clone(self: *const Self, allocator: std.mem.Allocator) !Self {
            var new_tensor = try init(allocator);
            @memcpy(new_tensor.slice(), self.constSlice());
            return new_tensor;
        }

        /// Print tensor for debugging.
        pub fn debugPrint(self: *const Self, writer: anytype) !void {
            try writer.print("Tensor({s}, {any})\n", .{ @typeName(T), shape });
            if (ndim == 0) {
                try writer.print("  scalar: {}\n", .{self.data[0]});
            } else if (ndim == 1) {
                try writer.print("  [", .{});
                for (0..@min(shape[0], 10)) |i| {
                    if (i > 0) try writer.print(", ", .{});
                    try writer.print("{d:.4}", .{self.get(.{i})});
                }
                if (shape[0] > 10) try writer.print(", ...", .{});
                try writer.print("]\n", .{});
            } else if (ndim == 2) {
                for (0..@min(shape[0], 5)) |i| {
                    try writer.print("  [", .{});
                    for (0..@min(shape[1], 10)) |j| {
                        if (j > 0) try writer.print(", ", .{});
                        try writer.print("{d:.4}", .{self.get(.{ i, j })});
                    }
                    if (shape[1] > 10) try writer.print(", ...", .{});
                    try writer.print("]\n", .{});
                }
                if (shape[0] > 5) try writer.print("  ...\n", .{});
            } else {
                try writer.print("  (tensor too high-dimensional to display)\n", .{});
            }
        }

        // ==================================================================
        // Arithmetic operations (return expression types when implemented)
        // For now, these are placeholder comments showing the intended API
        // ==================================================================
        // pub fn add(self: Self, other: anytype) BinaryExpr(.add, Self, @TypeOf(other))
        // pub fn sub(self: Self, other: anytype) BinaryExpr(.sub, Self, @TypeOf(other))
        // pub fn mul(self: Self, other: anytype) BinaryExpr(.mul, Self, @TypeOf(other))
        // pub fn div(self: Self, other: anytype) BinaryExpr(.div, Self, @TypeOf(other))
        // pub fn matmul(self: Self, other: anytype) MatmulExpr(Self, @TypeOf(other))
        // pub fn relu(self: Self) UnaryExpr(.relu, Self)
        // etc.
    };
}

/// Flatten a multi-dimensional array to 1D at comptime.
fn flattenArray(arr: anytype) [countElements(arr)]@typeInfo(@TypeOf(arr)).array.child {
    const T = @TypeOf(arr);
    const info = @typeInfo(T);

    if (info != .array) {
        @compileError("Expected array type");
    }

    const Child = info.array.child;
    const child_info = @typeInfo(Child);
    const total = countElements(arr);

    var result: [total]innerType(T) = undefined;

    if (child_info == .array) {
        // Nested array - recursively flatten
        var idx: usize = 0;
        for (arr) |sub| {
            const flat_sub = flattenArray(sub);
            for (flat_sub) |val| {
                result[idx] = val;
                idx += 1;
            }
        }
    } else {
        // 1D array - just copy
        result = arr;
    }

    return result;
}

/// Count total elements in a potentially nested array.
fn countElements(arr: anytype) usize {
    const T = @TypeOf(arr);
    const info = @typeInfo(T);

    if (info != .array) {
        return 1;
    }

    const Child = info.array.child;
    const child_info = @typeInfo(Child);

    if (child_info == .array) {
        // Nested array
        var total: usize = 0;
        for (arr) |sub| {
            total += countElements(sub);
        }
        return total;
    } else {
        return info.array.len;
    }
}

/// Get the innermost element type of a nested array.
fn innerType(comptime T: type) type {
    const info = @typeInfo(T);
    if (info == .array) {
        return innerType(info.array.child);
    }
    return T;
}

// ============================================================================
// Tests
// ============================================================================

test "Tensor basic creation" {
    const Vec = Tensor(f32, .{4});
    var vec = try Vec.init(std.testing.allocator);
    defer vec.deinit();

    try std.testing.expectEqual(@as(usize, 4), Vec.numel);
    try std.testing.expectEqual(@as(usize, 1), Vec.ndim);
}

test "Tensor zeros and ones" {
    const Mat = Tensor(f32, .{ 2, 3 });

    var zeros = try Mat.zeros(std.testing.allocator);
    defer zeros.deinit();

    for (zeros.slice()) |v| {
        try std.testing.expectEqual(@as(f32, 0), v);
    }

    var ones = try Mat.ones(std.testing.allocator);
    defer ones.deinit();

    for (ones.slice()) |v| {
        try std.testing.expectEqual(@as(f32, 1), v);
    }
}

test "Tensor full" {
    const Vec = Tensor(f32, .{10});
    var vec = try Vec.full(std.testing.allocator, 42.0);
    defer vec.deinit();

    for (vec.slice()) |v| {
        try std.testing.expectEqual(@as(f32, 42.0), v);
    }
}

test "Tensor fromSlice" {
    const Vec = Tensor(f32, .{4});
    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };

    var vec = try Vec.fromSlice(std.testing.allocator, &data);
    defer vec.deinit();

    try std.testing.expectEqualSlices(f32, &data, vec.slice());
}

test "Tensor get/set" {
    const Mat = Tensor(f32, .{ 2, 3 });
    var mat = try Mat.zeros(std.testing.allocator);
    defer mat.deinit();

    mat.set(.{ 0, 0 }, 1.0);
    mat.set(.{ 0, 1 }, 2.0);
    mat.set(.{ 1, 2 }, 3.0);

    try std.testing.expectEqual(@as(f32, 1.0), mat.get(.{ 0, 0 }));
    try std.testing.expectEqual(@as(f32, 2.0), mat.get(.{ 0, 1 }));
    try std.testing.expectEqual(@as(f32, 3.0), mat.get(.{ 1, 2 }));
    try std.testing.expectEqual(@as(f32, 0.0), mat.get(.{ 1, 0 }));
}

test "Tensor flat index" {
    const Vec = Tensor(f32, .{4});
    var vec = try Vec.zeros(std.testing.allocator);
    defer vec.deinit();

    vec.setFlatIndex(0, 1.0);
    vec.setFlatIndex(3, 4.0);

    try std.testing.expectEqual(@as(f32, 1.0), vec.getFlatIndex(0));
    try std.testing.expectEqual(@as(f32, 4.0), vec.getFlatIndex(3));
}

test "Tensor clone" {
    const Mat = Tensor(f32, .{ 2, 2 });
    var original = try Mat.fromSlice(std.testing.allocator, &[_]f32{ 1, 2, 3, 4 });
    defer original.deinit();

    var cloned = try original.clone(std.testing.allocator);
    defer cloned.deinit();

    // Modify original
    original.set(.{ 0, 0 }, 100.0);

    // Clone should be unchanged
    try std.testing.expectEqual(@as(f32, 1.0), cloned.get(.{ 0, 0 }));
}

test "Tensor isContiguous" {
    const Mat = Tensor(f32, .{ 2, 3 });
    var mat = try Mat.init(std.testing.allocator);
    defer mat.deinit();

    try std.testing.expect(mat.isContiguous());
}

test "Tensor properties" {
    const T = Tensor(f32, .{ 4, 5, 6 });

    try std.testing.expectEqual(@as(usize, 3), T.ndim);
    try std.testing.expectEqual(@as(usize, 120), T.numel);
    try std.testing.expectEqual(DType.f32, T.dtype);
    try std.testing.expectEqual(@as(usize, 4), T.shape[0]);
    try std.testing.expectEqual(@as(usize, 5), T.shape[1]);
    try std.testing.expectEqual(@as(usize, 6), T.shape[2]);
}

test "Tensor scalar" {
    const Scalar = Tensor(f32, .{});
    var s = try Scalar.init(std.testing.allocator);
    defer s.deinit();

    try std.testing.expectEqual(@as(usize, 0), Scalar.ndim);
    try std.testing.expectEqual(@as(usize, 1), Scalar.numel);
}

test "Tensor different dtypes" {
    // f64
    const F64Vec = Tensor(f64, .{4});
    var f64_vec = try F64Vec.zeros(std.testing.allocator);
    defer f64_vec.deinit();
    try std.testing.expectEqual(DType.f64, F64Vec.dtype);

    // i32
    const I32Vec = Tensor(i32, .{4});
    var i32_vec = try I32Vec.full(std.testing.allocator, 42);
    defer i32_vec.deinit();
    try std.testing.expectEqual(DType.i32, I32Vec.dtype);
    try std.testing.expectEqual(@as(i32, 42), i32_vec.get(.{0}));

    // u8
    const U8Vec = Tensor(u8, .{4});
    var u8_vec = try U8Vec.full(std.testing.allocator, 255);
    defer u8_vec.deinit();
    try std.testing.expectEqual(DType.u8, U8Vec.dtype);
}
