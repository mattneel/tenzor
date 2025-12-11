//! Comptime shape algebra for tensor dimensions.
//!
//! Shapes are compile-time known tuples of dimensions. All shape operations
//! (broadcasting, matmul compatibility, etc.) are validated at compile time,
//! turning potential runtime errors into compile errors.

const std = @import("std");

/// Maximum number of dimensions supported.
pub const MAX_RANK: usize = 8;

/// A shape is a comptime tuple of dimensions.
/// This type constructor creates a shape type from dimension values.
pub fn Shape(comptime dims: anytype) type {
    const dims_array = asArray(dims);

    return struct {
        pub const ndim: usize = dims_array.len;
        pub const rank: usize = dims_array.len;
        pub const dimensions: [ndim]usize = dims_array;

        /// Returns the total number of elements in this shape.
        pub fn numel() comptime_int {
            if (ndim == 0) return 1; // Scalar has 1 element
            var product: comptime_int = 1;
            for (dimensions) |d| {
                product *= d;
            }
            return product;
        }

        /// Returns true if this is a scalar (0-dimensional).
        pub fn isScalar() bool {
            return ndim == 0;
        }

        /// Returns true if this is a vector (1-dimensional).
        pub fn isVector() bool {
            return ndim == 1;
        }

        /// Returns true if this is a matrix (2-dimensional).
        pub fn isMatrix() bool {
            return ndim == 2;
        }

        /// Get the dimension at the given axis (supports negative indexing).
        pub fn dim(comptime axis: isize) usize {
            const actual_axis = normalizeAxis(ndim, axis);
            return dimensions[actual_axis];
        }

        /// Returns the dimensions as a runtime slice.
        pub fn toSlice() []const usize {
            return &dimensions;
        }

        /// Check if this shape equals another shape.
        pub fn eql(comptime Other: type) bool {
            if (!isShapeType(Other)) return false;
            if (ndim != Other.ndim) return false;
            for (dimensions, Other.dimensions) |a, b| {
                if (a != b) return false;
            }
            return true;
        }

        /// Compute strides for row-major (C-contiguous) layout.
        pub fn strides() [ndim]usize {
            if (ndim == 0) return .{};

            var result: [ndim]usize = undefined;
            var stride: usize = 1;
            var i: usize = ndim;
            while (i > 0) {
                i -= 1;
                result[i] = stride;
                stride *= dimensions[i];
            }
            return result;
        }

        /// Format the shape for display.
        pub fn format() []const u8 {
            return formatDims(&dimensions);
        }
    };
}

/// Scalar shape (0 dimensions).
pub const Scalar = Shape(.{});

/// Convert various dimension representations to a fixed array.
fn asArray(comptime dims: anytype) [countDims(dims)]usize {
    const T = @TypeOf(dims);
    const len = countDims(dims);

    if (len == 0) return .{};

    var result: [len]usize = undefined;

    if (T == @TypeOf(.{})) {
        // Empty tuple
        return result;
    }

    const info = @typeInfo(T);
    if (info == .@"struct" and info.@"struct".is_tuple) {
        // It's a tuple
        inline for (0..len) |i| {
            result[i] = dims[i];
        }
    } else if (info == .array) {
        // It's an array
        result = dims;
    } else {
        @compileError("Expected tuple or array of dimensions");
    }

    return result;
}

/// Count the number of dimensions.
fn countDims(comptime dims: anytype) usize {
    const T = @TypeOf(dims);
    const info = @typeInfo(T);

    if (info == .@"struct" and info.@"struct".is_tuple) {
        return info.@"struct".fields.len;
    } else if (info == .array) {
        return info.array.len;
    } else {
        @compileError("Expected tuple or array of dimensions, got " ++ @typeName(T));
    }
}

/// Check if a type is a Shape type.
pub fn isShapeType(comptime T: type) bool {
    return @hasDecl(T, "ndim") and @hasDecl(T, "dimensions") and @hasDecl(T, "numel");
}

/// Normalize an axis index (supports negative indexing like Python).
pub fn normalizeAxis(comptime ndim: usize, comptime axis: isize) usize {
    if (axis >= 0) {
        const a: usize = @intCast(axis);
        if (a >= ndim) {
            @compileError(std.fmt.comptimePrint(
                "Axis {} out of bounds for shape with {} dimensions",
                .{ axis, ndim },
            ));
        }
        return a;
    } else {
        const neg: usize = @intCast(-axis);
        if (neg > ndim) {
            @compileError(std.fmt.comptimePrint(
                "Axis {} out of bounds for shape with {} dimensions",
                .{ axis, ndim },
            ));
        }
        return ndim - neg;
    }
}

/// Check if two shapes are broadcast-compatible.
/// Broadcasting follows NumPy semantics: dimensions are compared from right to left,
/// and are compatible if they are equal or one of them is 1.
pub fn broadcastCompatible(comptime A: type, comptime B: type) bool {
    if (!isShapeType(A) or !isShapeType(B)) return false;

    const max_rank = @max(A.ndim, B.ndim);

    inline for (0..max_rank) |i| {
        const a_dim = if (i < A.ndim) A.dimensions[A.ndim - 1 - i] else 1;
        const b_dim = if (i < B.ndim) B.dimensions[B.ndim - 1 - i] else 1;

        if (a_dim != b_dim and a_dim != 1 and b_dim != 1) {
            return false;
        }
    }
    return true;
}

/// Compute the broadcast result shape of two shapes.
pub fn BroadcastShape(comptime A: type, comptime B: type) type {
    if (!broadcastCompatible(A, B)) {
        @compileError(std.fmt.comptimePrint(
            "Shapes are not broadcast compatible: {s} and {s}",
            .{ formatDims(&A.dimensions), formatDims(&B.dimensions) },
        ));
    }

    const max_rank = @max(A.ndim, B.ndim);
    var result_dims: [max_rank]usize = undefined;

    for (0..max_rank) |i| {
        const a_dim = if (i < A.ndim) A.dimensions[A.ndim - 1 - i] else 1;
        const b_dim = if (i < B.ndim) B.dimensions[B.ndim - 1 - i] else 1;
        result_dims[max_rank - 1 - i] = @max(a_dim, b_dim);
    }

    return Shape(result_dims);
}

/// Check if two shapes are compatible for matrix multiplication.
/// For 2D: [M, K] @ [K, N] -> [M, N]
/// For higher dims: batch dimensions must broadcast, last two dims follow matmul rules.
pub fn matmulCompatible(comptime A: type, comptime B: type) bool {
    if (!isShapeType(A) or !isShapeType(B)) return false;
    if (A.ndim < 1 or B.ndim < 1) return false;

    // Get the contracting dimension
    const a_k = A.dimensions[A.ndim - 1];
    const b_k = if (B.ndim >= 2) B.dimensions[B.ndim - 2] else B.dimensions[0];

    return a_k == b_k;
}

/// Compute the result shape of matrix multiplication.
pub fn MatmulShape(comptime A: type, comptime B: type) type {
    if (!matmulCompatible(A, B)) {
        @compileError(std.fmt.comptimePrint(
            "Shapes not compatible for matmul: {s} @ {s}",
            .{ formatDims(&A.dimensions), formatDims(&B.dimensions) },
        ));
    }

    // Handle various dimension cases
    if (A.ndim == 1 and B.ndim == 1) {
        // [K] @ [K] -> scalar
        return Scalar;
    } else if (A.ndim == 1 and B.ndim == 2) {
        // [K] @ [K, N] -> [N]
        return Shape(.{B.dimensions[1]});
    } else if (A.ndim == 2 and B.ndim == 1) {
        // [M, K] @ [K] -> [M]
        return Shape(.{A.dimensions[0]});
    } else if (A.ndim == 2 and B.ndim == 2) {
        // [M, K] @ [K, N] -> [M, N]
        return Shape(.{ A.dimensions[0], B.dimensions[1] });
    } else {
        // Batched matmul - not yet implemented
        @compileError("Batched matmul not yet implemented");
    }
}

/// Compute the shape after reduction along given axes.
pub fn ReduceShape(comptime Input: type, comptime axes: anytype, comptime keepdims: bool) type {
    if (!isShapeType(Input)) @compileError("Expected shape type");

    const axes_array = asArray(axes);
    const num_axes = axes_array.len;

    if (num_axes == 0) {
        // Reduce all dimensions
        if (keepdims) {
            var ones: [Input.ndim]usize = undefined;
            for (&ones) |*o| o.* = 1;
            return Shape(ones);
        } else {
            return Scalar;
        }
    }

    if (keepdims) {
        var result: [Input.ndim]usize = Input.dimensions;
        for (axes_array) |axis| {
            const norm = normalizeAxis(Input.ndim, @as(isize, @intCast(axis)));
            result[norm] = 1;
        }
        return Shape(result);
    } else {
        const new_rank = Input.ndim - num_axes;
        if (new_rank == 0) return Scalar;

        var result: [new_rank]usize = undefined;
        var j: usize = 0;
        for (0..Input.ndim) |i| {
            var is_reduced = false;
            for (axes_array) |axis| {
                const norm = normalizeAxis(Input.ndim, @as(isize, @intCast(axis)));
                if (i == norm) {
                    is_reduced = true;
                    break;
                }
            }
            if (!is_reduced) {
                result[j] = Input.dimensions[i];
                j += 1;
            }
        }
        return Shape(result);
    }
}

/// Compute the shape after transpose with given permutation.
pub fn TransposeShape(comptime Input: type, comptime perm: anytype) type {
    if (!isShapeType(Input)) @compileError("Expected shape type");

    const perm_array = asArray(perm);
    if (perm_array.len != Input.ndim) {
        @compileError("Permutation length must match number of dimensions");
    }

    // Validate permutation
    var seen: [MAX_RANK]bool = .{false} ** MAX_RANK;
    for (perm_array) |p| {
        if (p >= Input.ndim) @compileError("Invalid permutation index");
        if (seen[p]) @compileError("Duplicate index in permutation");
        seen[p] = true;
    }

    var result: [Input.ndim]usize = undefined;
    for (0..Input.ndim) |i| {
        result[i] = Input.dimensions[perm_array[i]];
    }
    return Shape(result);
}

/// Compute the shape after a reshape operation.
/// One dimension can be -1 to be inferred.
pub fn ReshapeShape(comptime Input: type, comptime new_dims: anytype) type {
    if (!isShapeType(Input)) @compileError("Expected shape type");

    const dims_array = asArray(new_dims);
    const input_numel = Input.numel();

    // Count -1s and compute known product
    var infer_idx: ?usize = null;
    var known_product: usize = 1;
    for (dims_array, 0..) |d, i| {
        if (d == @as(usize, @bitCast(@as(isize, -1)))) {
            if (infer_idx != null) @compileError("Can only have one inferred dimension (-1)");
            infer_idx = i;
        } else {
            known_product *= d;
        }
    }

    var result: [dims_array.len]usize = undefined;
    for (dims_array, 0..) |d, i| {
        if (d == @as(usize, @bitCast(@as(isize, -1)))) {
            if (input_numel % known_product != 0) {
                @compileError("Cannot infer dimension: total elements not divisible");
            }
            result[i] = input_numel / known_product;
        } else {
            result[i] = d;
        }
    }

    // Validate total elements match
    var result_numel: usize = 1;
    for (result) |d| result_numel *= d;
    if (result_numel != input_numel) {
        @compileError(std.fmt.comptimePrint(
            "Reshape changes number of elements: {} -> {}",
            .{ input_numel, result_numel },
        ));
    }

    return Shape(result);
}

/// Format dimensions for error messages.
fn formatDims(comptime dims: []const usize) []const u8 {
    if (dims.len == 0) return "()";

    var buf: [256]u8 = undefined;
    var len: usize = 0;

    buf[len] = '(';
    len += 1;

    for (dims, 0..) |d, i| {
        if (i > 0) {
            buf[len] = ',';
            len += 1;
            buf[len] = ' ';
            len += 1;
        }
        const num_str = std.fmt.comptimePrint("{}", .{d});
        @memcpy(buf[len..][0..num_str.len], num_str);
        len += num_str.len;
    }

    buf[len] = ')';
    len += 1;

    // Return a comptime slice
    return buf[0..len];
}

// ============================================================================
// Tests
// ============================================================================

test "Shape basic properties" {
    const S = Shape(.{ 2, 3, 4 });
    try std.testing.expectEqual(@as(usize, 3), S.ndim);
    try std.testing.expectEqual(@as(usize, 3), S.rank);
    try std.testing.expectEqual(@as(comptime_int, 24), S.numel());
    try std.testing.expect(!S.isScalar());
    try std.testing.expect(!S.isVector());
    try std.testing.expect(!S.isMatrix());
}

test "Shape scalar" {
    const S = Scalar;
    try std.testing.expectEqual(@as(usize, 0), S.ndim);
    try std.testing.expectEqual(@as(comptime_int, 1), S.numel());
    try std.testing.expect(S.isScalar());
}

test "Shape vector and matrix" {
    const V = Shape(.{10});
    try std.testing.expect(V.isVector());
    try std.testing.expect(!V.isMatrix());

    const M = Shape(.{ 3, 4 });
    try std.testing.expect(!M.isVector());
    try std.testing.expect(M.isMatrix());
}

test "Shape dim access" {
    const S = Shape(.{ 2, 3, 4 });
    try std.testing.expectEqual(@as(usize, 2), S.dim(0));
    try std.testing.expectEqual(@as(usize, 3), S.dim(1));
    try std.testing.expectEqual(@as(usize, 4), S.dim(2));
    // Negative indexing
    try std.testing.expectEqual(@as(usize, 4), S.dim(-1));
    try std.testing.expectEqual(@as(usize, 3), S.dim(-2));
    try std.testing.expectEqual(@as(usize, 2), S.dim(-3));
}

test "Shape strides" {
    const S = Shape(.{ 2, 3, 4 });
    const strides = S.strides();
    try std.testing.expectEqual(@as(usize, 12), strides[0]); // 3 * 4
    try std.testing.expectEqual(@as(usize, 4), strides[1]); // 4
    try std.testing.expectEqual(@as(usize, 1), strides[2]); // 1
}

test "Shape equality" {
    const A = Shape(.{ 2, 3 });
    const B = Shape(.{ 2, 3 });
    const C = Shape(.{ 3, 2 });

    try std.testing.expect(A.eql(B));
    try std.testing.expect(!A.eql(C));
}

test "broadcast compatibility" {
    // Same shapes
    try std.testing.expect(broadcastCompatible(Shape(.{ 3, 4 }), Shape(.{ 3, 4 })));

    // Scalar broadcasts with anything
    try std.testing.expect(broadcastCompatible(Scalar, Shape(.{ 3, 4 })));

    // One dimension is 1
    try std.testing.expect(broadcastCompatible(Shape(.{ 3, 1 }), Shape(.{ 3, 4 })));
    try std.testing.expect(broadcastCompatible(Shape(.{ 1, 4 }), Shape(.{ 3, 4 })));

    // Different ranks
    try std.testing.expect(broadcastCompatible(Shape(.{4}), Shape(.{ 3, 4 })));
    try std.testing.expect(broadcastCompatible(Shape(.{ 3, 4 }), Shape(.{4})));

    // Incompatible
    try std.testing.expect(!broadcastCompatible(Shape(.{ 3, 4 }), Shape(.{ 3, 5 })));
}

test "BroadcastShape" {
    const A = Shape(.{ 3, 1 });
    const B = Shape(.{ 1, 4 });
    const C = BroadcastShape(A, B);

    try std.testing.expectEqual(@as(usize, 2), C.ndim);
    try std.testing.expectEqual(@as(usize, 3), C.dimensions[0]);
    try std.testing.expectEqual(@as(usize, 4), C.dimensions[1]);

    // Different ranks
    const D = Shape(.{4});
    const E = Shape(.{ 3, 4 });
    const F = BroadcastShape(D, E);

    try std.testing.expectEqual(@as(usize, 2), F.ndim);
    try std.testing.expectEqual(@as(usize, 3), F.dimensions[0]);
    try std.testing.expectEqual(@as(usize, 4), F.dimensions[1]);
}

test "matmul compatibility" {
    // 2D @ 2D
    try std.testing.expect(matmulCompatible(Shape(.{ 3, 4 }), Shape(.{ 4, 5 })));
    try std.testing.expect(!matmulCompatible(Shape(.{ 3, 4 }), Shape(.{ 5, 6 })));

    // Vector cases
    try std.testing.expect(matmulCompatible(Shape(.{4}), Shape(.{4})));
    try std.testing.expect(matmulCompatible(Shape(.{4}), Shape(.{ 4, 5 })));
    try std.testing.expect(matmulCompatible(Shape(.{ 3, 4 }), Shape(.{4})));
}

test "MatmulShape" {
    // [M, K] @ [K, N] -> [M, N]
    const A = Shape(.{ 3, 4 });
    const B = Shape(.{ 4, 5 });
    const C = MatmulShape(A, B);

    try std.testing.expectEqual(@as(usize, 2), C.ndim);
    try std.testing.expectEqual(@as(usize, 3), C.dimensions[0]);
    try std.testing.expectEqual(@as(usize, 5), C.dimensions[1]);

    // [K] @ [K] -> scalar
    const V1 = Shape(.{4});
    const V2 = Shape(.{4});
    const S = MatmulShape(V1, V2);
    try std.testing.expect(S.isScalar());

    // [K] @ [K, N] -> [N]
    const R = MatmulShape(Shape(.{4}), Shape(.{ 4, 5 }));
    try std.testing.expectEqual(@as(usize, 1), R.ndim);
    try std.testing.expectEqual(@as(usize, 5), R.dimensions[0]);
}

test "ReduceShape" {
    const Input = Shape(.{ 2, 3, 4 });

    // Reduce single axis, keepdims=false
    const R1 = ReduceShape(Input, .{1}, false);
    try std.testing.expectEqual(@as(usize, 2), R1.ndim);
    try std.testing.expectEqual(@as(usize, 2), R1.dimensions[0]);
    try std.testing.expectEqual(@as(usize, 4), R1.dimensions[1]);

    // Reduce single axis, keepdims=true
    const R2 = ReduceShape(Input, .{1}, true);
    try std.testing.expectEqual(@as(usize, 3), R2.ndim);
    try std.testing.expectEqual(@as(usize, 2), R2.dimensions[0]);
    try std.testing.expectEqual(@as(usize, 1), R2.dimensions[1]);
    try std.testing.expectEqual(@as(usize, 4), R2.dimensions[2]);

    // Reduce all dimensions
    const R3 = ReduceShape(Input, .{}, false);
    try std.testing.expect(R3.isScalar());
}

test "TransposeShape" {
    const Input = Shape(.{ 2, 3, 4 });
    const T = TransposeShape(Input, .{ 2, 0, 1 });

    try std.testing.expectEqual(@as(usize, 3), T.ndim);
    try std.testing.expectEqual(@as(usize, 4), T.dimensions[0]);
    try std.testing.expectEqual(@as(usize, 2), T.dimensions[1]);
    try std.testing.expectEqual(@as(usize, 3), T.dimensions[2]);
}

test "ReshapeShape" {
    const Input = Shape(.{ 2, 3, 4 }); // 24 elements

    // Explicit reshape
    const R1 = ReshapeShape(Input, .{ 6, 4 });
    try std.testing.expectEqual(@as(usize, 2), R1.ndim);
    try std.testing.expectEqual(@as(usize, 6), R1.dimensions[0]);
    try std.testing.expectEqual(@as(usize, 4), R1.dimensions[1]);

    // Flatten
    const R2 = ReshapeShape(Input, .{24});
    try std.testing.expectEqual(@as(usize, 1), R2.ndim);
    try std.testing.expectEqual(@as(usize, 24), R2.dimensions[0]);
}
