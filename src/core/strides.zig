//! Stride computation and memory layout utilities.
//!
//! Strides define the memory layout of a tensor - how many elements to skip
//! when moving along each dimension. This module provides utilities for:
//! - Computing contiguous strides
//! - Converting multi-dimensional indices to flat offsets
//! - Handling broadcasting with virtual strides
//! - Checking memory contiguity

const std = @import("std");
const shape_mod = @import("shape.zig");
const Shape = shape_mod.Shape;
const MAX_RANK = shape_mod.MAX_RANK;

/// Memory layout order.
pub const Layout = enum {
    /// Row-major (C-style): last dimension is contiguous.
    row_major,
    /// Column-major (Fortran-style): first dimension is contiguous.
    col_major,
};

/// Compute contiguous strides for a given shape and layout.
pub fn contiguousStrides(
    comptime ndim: usize,
    dimensions: [ndim]usize,
    layout: Layout,
) [ndim]usize {
    if (ndim == 0) return .{};

    var strides: [ndim]usize = undefined;

    switch (layout) {
        .row_major => {
            var stride: usize = 1;
            var i: usize = ndim;
            while (i > 0) {
                i -= 1;
                strides[i] = stride;
                stride *= dimensions[i];
            }
        },
        .col_major => {
            var stride: usize = 1;
            for (0..ndim) |i| {
                strides[i] = stride;
                stride *= dimensions[i];
            }
        },
    }

    return strides;
}

/// Compute the flat offset for a multi-dimensional index.
pub fn flatOffset(
    comptime ndim: usize,
    indices: [ndim]usize,
    strides: [ndim]usize,
) usize {
    var offset: usize = 0;
    for (indices, strides) |idx, stride| {
        offset += idx * stride;
    }
    return offset;
}

/// Convert a flat index to multi-dimensional indices (for row-major layout).
pub fn unflattenIndex(
    comptime ndim: usize,
    flat_idx: usize,
    dimensions: [ndim]usize,
) [ndim]usize {
    if (ndim == 0) return .{};

    var indices: [ndim]usize = undefined;
    var remaining = flat_idx;

    // Compute from last dimension to first
    var i: usize = ndim;
    while (i > 0) {
        i -= 1;
        indices[i] = remaining % dimensions[i];
        remaining /= dimensions[i];
    }

    return indices;
}

/// Compute broadcast strides for a shape being broadcast to a target shape.
/// Dimensions that are broadcast (size 1 -> size N) get stride 0.
pub fn broadcastStrides(
    comptime src_ndim: usize,
    comptime dst_ndim: usize,
    src_dims: [src_ndim]usize,
    src_strides: [src_ndim]usize,
) [dst_ndim]usize {
    var result: [dst_ndim]usize = undefined;

    for (0..dst_ndim) |i| {
        const src_i = if (i + src_ndim >= dst_ndim) i + src_ndim - dst_ndim else null;

        if (src_i) |si| {
            if (si < src_ndim) {
                // If source dimension is 1, broadcast with stride 0
                if (src_dims[si] == 1) {
                    result[i] = 0;
                } else {
                    result[i] = src_strides[si];
                }
            } else {
                result[i] = 0;
            }
        } else {
            // Prepended dimension - stride 0
            result[i] = 0;
        }
    }

    return result;
}

/// Check if strides represent a contiguous row-major layout.
pub fn isContiguous(
    comptime ndim: usize,
    dimensions: [ndim]usize,
    strides: [ndim]usize,
) bool {
    if (ndim == 0) return true;

    var expected_stride: usize = 1;
    var i: usize = ndim;
    while (i > 0) {
        i -= 1;
        // Skip dimensions of size 1 (they don't affect contiguity)
        if (dimensions[i] == 1) continue;

        if (strides[i] != expected_stride) return false;
        expected_stride *= dimensions[i];
    }

    return true;
}

/// Check if two sets of strides are compatible for a pointwise operation.
/// Compatible means either both have the same stride, or one has stride 0 (broadcast).
pub fn stridesCompatible(
    comptime ndim: usize,
    strides_a: [ndim]usize,
    strides_b: [ndim]usize,
) bool {
    for (strides_a, strides_b) |sa, sb| {
        if (sa != sb and sa != 0 and sb != 0) return false;
    }
    return true;
}

/// Compute the total memory span of a tensor (distance from first to last element + 1).
/// This is the minimum buffer size needed to hold the tensor data.
pub fn memorySpan(
    comptime ndim: usize,
    dimensions: [ndim]usize,
    strides: [ndim]usize,
) usize {
    if (ndim == 0) return 1;

    // Find the offset of the last element
    var last_offset: usize = 0;
    for (dimensions, strides) |dim, stride| {
        if (dim > 0) {
            last_offset += (dim - 1) * stride;
        }
    }

    return last_offset + 1;
}

/// Iterator for traversing tensor elements in memory order.
pub fn StridedIterator(comptime ndim: usize) type {
    return struct {
        dimensions: [ndim]usize,
        strides: [ndim]usize,
        indices: [ndim]usize,
        offset: usize,
        done: bool,

        const Self = @This();

        pub fn init(dimensions: [ndim]usize, strides: [ndim]usize) Self {
            return .{
                .dimensions = dimensions,
                .strides = strides,
                .indices = [_]usize{0} ** ndim,
                .offset = 0,
                .done = numel(dimensions) == 0,
            };
        }

        fn numel(dimensions: [ndim]usize) usize {
            var result: usize = 1;
            for (dimensions) |d| result *= d;
            return result;
        }

        pub fn next(self: *Self) ?usize {
            if (self.done) return null;

            const current_offset = self.offset;

            // Advance indices
            var i: usize = ndim;
            while (i > 0) {
                i -= 1;
                self.indices[i] += 1;
                self.offset += self.strides[i];

                if (self.indices[i] < self.dimensions[i]) {
                    break;
                } else {
                    // Carry over
                    self.offset -= self.indices[i] * self.strides[i];
                    self.indices[i] = 0;

                    if (i == 0) {
                        self.done = true;
                    }
                }
            }

            return current_offset;
        }

        pub fn reset(self: *Self) void {
            self.indices = [_]usize{0} ** ndim;
            self.offset = 0;
            self.done = numel(self.dimensions) == 0;
        }
    };
}

/// Comptime stride utilities for Shape types.
pub fn ShapeStrides(comptime S: type) type {
    if (!shape_mod.isShapeType(S)) {
        @compileError("Expected a Shape type");
    }

    return struct {
        pub const ndim = S.ndim;

        /// Row-major contiguous strides.
        pub fn rowMajor() [ndim]usize {
            return contiguousStrides(ndim, S.dimensions, .row_major);
        }

        /// Column-major contiguous strides.
        pub fn colMajor() [ndim]usize {
            return contiguousStrides(ndim, S.dimensions, .col_major);
        }

        /// Default strides (row-major).
        pub fn default() [ndim]usize {
            return rowMajor();
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

test "contiguousStrides row_major" {
    const strides = contiguousStrides(3, .{ 2, 3, 4 }, .row_major);
    try std.testing.expectEqual(@as(usize, 12), strides[0]); // 3 * 4
    try std.testing.expectEqual(@as(usize, 4), strides[1]); // 4
    try std.testing.expectEqual(@as(usize, 1), strides[2]); // 1
}

test "contiguousStrides col_major" {
    const strides = contiguousStrides(3, .{ 2, 3, 4 }, .col_major);
    try std.testing.expectEqual(@as(usize, 1), strides[0]); // 1
    try std.testing.expectEqual(@as(usize, 2), strides[1]); // 2
    try std.testing.expectEqual(@as(usize, 6), strides[2]); // 2 * 3
}

test "flatOffset" {
    // For a 2x3x4 tensor with row-major strides [12, 4, 1]
    const offset = flatOffset(3, .{ 1, 2, 3 }, .{ 12, 4, 1 });
    // 1*12 + 2*4 + 3*1 = 12 + 8 + 3 = 23
    try std.testing.expectEqual(@as(usize, 23), offset);
}

test "unflattenIndex" {
    const indices = unflattenIndex(3, 23, .{ 2, 3, 4 });
    try std.testing.expectEqual(@as(usize, 1), indices[0]);
    try std.testing.expectEqual(@as(usize, 2), indices[1]);
    try std.testing.expectEqual(@as(usize, 3), indices[2]);
}

test "broadcastStrides" {
    // Broadcasting [1, 4] to [3, 4]
    const src_dims = [_]usize{ 1, 4 };
    const src_strides = [_]usize{ 4, 1 };
    const result = broadcastStrides(2, 2, src_dims, src_strides);

    try std.testing.expectEqual(@as(usize, 0), result[0]); // Broadcast dimension
    try std.testing.expectEqual(@as(usize, 1), result[1]); // Normal stride
}

test "isContiguous" {
    // Contiguous
    try std.testing.expect(isContiguous(3, .{ 2, 3, 4 }, .{ 12, 4, 1 }));

    // Non-contiguous (transposed)
    try std.testing.expect(!isContiguous(2, .{ 3, 4 }, .{ 1, 3 }));

    // Size-1 dimensions don't affect contiguity
    try std.testing.expect(isContiguous(3, .{ 2, 1, 4 }, .{ 4, 4, 1 }));
}

test "memorySpan" {
    // Contiguous 2x3x4
    const span1 = memorySpan(3, .{ 2, 3, 4 }, .{ 12, 4, 1 });
    try std.testing.expectEqual(@as(usize, 24), span1);

    // Non-contiguous with gaps
    const span2 = memorySpan(2, .{ 2, 3 }, .{ 10, 1 });
    try std.testing.expectEqual(@as(usize, 13), span2); // (2-1)*10 + (3-1)*1 + 1 = 13
}

test "StridedIterator" {
    var iter = StridedIterator(2).init(.{ 2, 3 }, .{ 3, 1 });

    // Should produce offsets: 0, 1, 2, 3, 4, 5
    try std.testing.expectEqual(@as(?usize, 0), iter.next());
    try std.testing.expectEqual(@as(?usize, 1), iter.next());
    try std.testing.expectEqual(@as(?usize, 2), iter.next());
    try std.testing.expectEqual(@as(?usize, 3), iter.next());
    try std.testing.expectEqual(@as(?usize, 4), iter.next());
    try std.testing.expectEqual(@as(?usize, 5), iter.next());
    try std.testing.expectEqual(@as(?usize, null), iter.next());
}

test "ShapeStrides" {
    const S = Shape(.{ 2, 3, 4 });
    const Strides = ShapeStrides(S);

    const row = Strides.rowMajor();
    try std.testing.expectEqual(@as(usize, 12), row[0]);
    try std.testing.expectEqual(@as(usize, 4), row[1]);
    try std.testing.expectEqual(@as(usize, 1), row[2]);

    const col = Strides.colMajor();
    try std.testing.expectEqual(@as(usize, 1), col[0]);
    try std.testing.expectEqual(@as(usize, 2), col[1]);
    try std.testing.expectEqual(@as(usize, 6), col[2]);
}
