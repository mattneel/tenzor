//! Gather/embedding lookup CPU kernel.
//!
//! Looks up embeddings from a table using indices.
//! table: [vocab_size, embed_dim], indices: [*] -> output: [*, embed_dim]
//!
//! Used for token embedding lookup in transformers.

const std = @import("std");
const simd = @import("../simd.zig");

/// Gather embeddings from a table using indices.
/// table: [vocab_size, embed_dim]
/// indices: flat array of integer indices
/// output: [num_indices, embed_dim]
pub fn gather(
    comptime T: type,
    comptime IndexT: type,
    table: []const T,
    indices: []const IndexT,
    output: []T,
    vocab_size: usize,
    embed_dim: usize,
) void {
    const vec_len = simd.suggestVectorLength(T);

    for (indices, 0..) |idx, i| {
        const idx_usize: usize = @intCast(idx);

        // Bounds check in debug mode
        if (std.debug.runtime_safety) {
            if (idx_usize >= vocab_size) {
                std.debug.panic("Gather index {} out of bounds for vocab_size {}", .{ idx_usize, vocab_size });
            }
        }

        const src_offset = idx_usize * embed_dim;
        const dst_offset = i * embed_dim;
        const src = table[src_offset..][0..embed_dim];
        const dst = output[dst_offset..][0..embed_dim];

        // SIMD copy
        var j: usize = 0;
        while (j + vec_len <= embed_dim) : (j += vec_len) {
            const v = simd.load(T, src[j..]);
            simd.store(T, v, dst[j..]);
        }

        // Scalar remainder
        while (j < embed_dim) : (j += 1) {
            dst[j] = src[j];
        }
    }
}

/// Gather with batched indices.
/// table: [vocab_size, embed_dim]
/// indices: [batch, seq_len]
/// output: [batch, seq_len, embed_dim]
pub fn gatherBatched(
    comptime T: type,
    comptime IndexT: type,
    table: []const T,
    indices: []const IndexT,
    output: []T,
    vocab_size: usize,
    embed_dim: usize,
    batch_size: usize,
    seq_len: usize,
) void {
    _ = batch_size; // Not needed, just iterate over all indices
    _ = seq_len;

    // Treat as flat indices
    gather(T, IndexT, table, indices, output, vocab_size, embed_dim);
}

// ============================================================================
// Tests
// ============================================================================

test "gather basic" {
    // 4 vocab entries, 3 embedding dims
    const table = [_]f32{
        1, 2, 3, // entry 0
        4, 5, 6, // entry 1
        7, 8, 9, // entry 2
        10, 11, 12, // entry 3
    };
    const indices = [_]u32{ 0, 2, 1 };
    var output: [9]f32 = undefined;

    gather(f32, u32, &table, &indices, &output, 4, 3);

    // Entry 0
    try std.testing.expectEqual(@as(f32, 1), output[0]);
    try std.testing.expectEqual(@as(f32, 2), output[1]);
    try std.testing.expectEqual(@as(f32, 3), output[2]);

    // Entry 2
    try std.testing.expectEqual(@as(f32, 7), output[3]);
    try std.testing.expectEqual(@as(f32, 8), output[4]);
    try std.testing.expectEqual(@as(f32, 9), output[5]);

    // Entry 1
    try std.testing.expectEqual(@as(f32, 4), output[6]);
    try std.testing.expectEqual(@as(f32, 5), output[7]);
    try std.testing.expectEqual(@as(f32, 6), output[8]);
}

test "gather single index" {
    const table = [_]f32{
        1, 2, 3, 4, // entry 0
        5, 6, 7, 8, // entry 1
    };
    const indices = [_]u32{1};
    var output: [4]f32 = undefined;

    gather(f32, u32, &table, &indices, &output, 2, 4);

    try std.testing.expectEqual(@as(f32, 5), output[0]);
    try std.testing.expectEqual(@as(f32, 6), output[1]);
    try std.testing.expectEqual(@as(f32, 7), output[2]);
    try std.testing.expectEqual(@as(f32, 8), output[3]);
}

test "gather repeated indices" {
    const table = [_]f32{
        1, 2, // entry 0
        3, 4, // entry 1
    };
    const indices = [_]u32{ 0, 0, 1, 0 };
    var output: [8]f32 = undefined;

    gather(f32, u32, &table, &indices, &output, 2, 2);

    // All should be entry 0 except index 2
    try std.testing.expectEqual(@as(f32, 1), output[0]);
    try std.testing.expectEqual(@as(f32, 2), output[1]);
    try std.testing.expectEqual(@as(f32, 1), output[2]);
    try std.testing.expectEqual(@as(f32, 2), output[3]);
    try std.testing.expectEqual(@as(f32, 3), output[4]);
    try std.testing.expectEqual(@as(f32, 4), output[5]);
    try std.testing.expectEqual(@as(f32, 1), output[6]);
    try std.testing.expectEqual(@as(f32, 2), output[7]);
}

test "gather transformer pattern" {
    // Simulating token embedding lookup
    // vocab_size=100, embed_dim=8, batch=1, seq_len=4
    var table: [100 * 8]f32 = undefined;

    // Initialize: entry i has value i in all positions
    for (0..100) |i| {
        for (0..8) |j| {
            table[i * 8 + j] = @floatFromInt(i);
        }
    }

    const indices = [_]u32{ 5, 10, 3, 99 };
    var output: [4 * 8]f32 = undefined;

    gather(f32, u32, &table, &indices, &output, 100, 8);

    // Check first embedding (index 5)
    for (0..8) |j| {
        try std.testing.expectEqual(@as(f32, 5), output[j]);
    }

    // Check last embedding (index 99)
    for (0..8) |j| {
        try std.testing.expectEqual(@as(f32, 99), output[24 + j]);
    }
}
