//! Buffer pool for efficient memory reuse.
//!
//! Maintains a pool of recently freed buffers, bucketed by size,
//! to reduce allocation overhead during repeated tensor operations.

const std = @import("std");

/// Number of size buckets in the pool.
/// Buckets are powers of 2: 64B, 128B, 256B, ... up to 64MB
const NUM_BUCKETS = 20;

/// Minimum allocation size (64 bytes)
const MIN_SIZE = 64;

/// Maximum poolable size (64MB)
const MAX_SIZE = 64 * 1024 * 1024;

/// Maximum buffers per bucket to prevent unbounded memory growth.
const MAX_BUFFERS_PER_BUCKET = 8;

/// A pooled buffer with its original size and alignment preserved.
const PooledBuffer = struct {
    ptr: [*]align(1) u8,
    size: usize,
    log2_align: u8, // log2 of original alignment
};

/// Buffer pool that caches recently freed allocations for reuse.
pub const BufferPool = struct {
    /// One list of free buffers per size bucket
    buckets: [NUM_BUCKETS]std.ArrayListUnmanaged(PooledBuffer),
    /// Backing allocator for new allocations
    backing: std.mem.Allocator,
    /// Statistics
    hits: usize,
    misses: usize,
    total_pooled_bytes: usize,

    const Self = @This();

    /// Initialize a new buffer pool.
    pub fn init(backing: std.mem.Allocator) Self {
        var buckets: [NUM_BUCKETS]std.ArrayListUnmanaged(PooledBuffer) = undefined;
        for (&buckets) |*bucket| {
            bucket.* = .{};
        }
        return Self{
            .buckets = buckets,
            .backing = backing,
            .hits = 0,
            .misses = 0,
            .total_pooled_bytes = 0,
        };
    }

    /// Deinitialize, freeing all pooled buffers.
    pub fn deinit(self: *Self) void {
        for (&self.buckets) |*bucket| {
            for (bucket.items) |item| {
                const alignment: std.mem.Alignment = @enumFromInt(item.log2_align);
                self.backing.rawFree(item.ptr[0..item.size], alignment, @returnAddress());
            }
            bucket.deinit(self.backing);
        }
    }

    /// Try to get a buffer from the pool.
    /// Returns null if no suitable buffer is available.
    pub fn get(self: *Self, size: usize) ?[]u8 {
        if (size < MIN_SIZE or size > MAX_SIZE) {
            return null;
        }

        const bucket_idx = getBucketIndex(size);
        var bucket = &self.buckets[bucket_idx];

        if (bucket.items.len > 0) {
            const item = bucket.pop().?;
            self.hits += 1;
            self.total_pooled_bytes -= item.size;
            // Return the full original buffer
            return item.ptr[0..item.size];
        }

        self.misses += 1;
        return null;
    }

    /// Return a buffer to the pool for reuse.
    /// If the pool for this size is full, the buffer is freed.
    /// alignment should be the alignment the buffer was originally allocated with.
    pub fn putAligned(self: *Self, buf: []u8, alignment: std.mem.Alignment) void {
        if (buf.len < MIN_SIZE or buf.len > MAX_SIZE) {
            self.backing.rawFree(buf, alignment, @returnAddress());
            return;
        }

        const bucket_idx = getBucketIndex(buf.len);
        var bucket = &self.buckets[bucket_idx];

        // Check if bucket is full
        if (bucket.items.len >= MAX_BUFFERS_PER_BUCKET) {
            self.backing.rawFree(buf, alignment, @returnAddress());
            return;
        }

        // Add to pool with original size and alignment preserved
        const item = PooledBuffer{
            .ptr = buf.ptr,
            .size = buf.len,
            .log2_align = @intFromEnum(alignment),
        };
        bucket.append(self.backing, item) catch {
            self.backing.rawFree(buf, alignment, @returnAddress());
            return;
        };
        self.total_pooled_bytes += buf.len;
    }

    /// Return a buffer to the pool for reuse (assumes alignment 1).
    /// If the pool for this size is full, the buffer is freed.
    pub fn put(self: *Self, buf: []u8) void {
        self.putAligned(buf, .@"1");
    }

    /// Get statistics about pool usage.
    pub fn getStats(self: *const Self) Stats {
        return Stats{
            .hits = self.hits,
            .misses = self.misses,
            .hit_rate = if (self.hits + self.misses > 0)
                @as(f64, @floatFromInt(self.hits)) / @as(f64, @floatFromInt(self.hits + self.misses))
            else
                0.0,
            .total_pooled_bytes = self.total_pooled_bytes,
        };
    }

    /// Clear all pooled buffers, freeing memory.
    pub fn clear(self: *Self) void {
        for (&self.buckets) |*bucket| {
            for (bucket.items) |item| {
                const alignment: std.mem.Alignment = @enumFromInt(item.log2_align);
                self.backing.rawFree(item.ptr[0..item.size], alignment, @returnAddress());
            }
            bucket.clearRetainingCapacity();
        }
        self.total_pooled_bytes = 0;
    }

    /// Get the bucket index for a given size.
    fn getBucketIndex(size: usize) usize {
        // Round up to nearest power of 2, then compute log2
        const rounded = std.math.ceilPowerOfTwo(usize, size) catch MAX_SIZE;
        const log2_size = @ctz(rounded);
        const log2_min = @ctz(@as(usize, MIN_SIZE));

        if (log2_size < log2_min) return 0;
        const idx = log2_size - log2_min;
        return @min(idx, NUM_BUCKETS - 1);
    }
};

/// Pool statistics.
pub const Stats = struct {
    hits: usize,
    misses: usize,
    hit_rate: f64,
    total_pooled_bytes: usize,
};

// ============================================================================
// Tests
// ============================================================================

test "basic pool operations" {
    var pool = BufferPool.init(std.testing.allocator);
    defer pool.deinit();

    // Pool should be empty
    try std.testing.expect(pool.get(64) == null);

    // Allocate and return a buffer
    const buf = try std.testing.allocator.alloc(u8, 64);
    pool.put(buf);

    // Should get it back
    const retrieved = pool.get(64);
    try std.testing.expect(retrieved != null);
    try std.testing.expectEqual(@as(usize, 64), retrieved.?.len);

    // Pool should be empty again
    try std.testing.expect(pool.get(64) == null);

    // Clean up the buffer we got
    std.testing.allocator.free(retrieved.?);
}

test "pool bucket sizing" {
    var pool = BufferPool.init(std.testing.allocator);
    defer pool.deinit();

    // Put a 128-byte buffer
    const buf = try std.testing.allocator.alloc(u8, 128);
    pool.put(buf);

    // Request for 128 bytes should hit and return original size
    const retrieved = pool.get(128);
    try std.testing.expect(retrieved != null);
    try std.testing.expectEqual(@as(usize, 128), retrieved.?.len);
    std.testing.allocator.free(retrieved.?);
}

test "pool statistics" {
    var pool = BufferPool.init(std.testing.allocator);
    defer pool.deinit();

    // Miss
    _ = pool.get(64);
    try std.testing.expectEqual(@as(usize, 1), pool.getStats().misses);

    // Put and hit
    const buf = try std.testing.allocator.alloc(u8, 64);
    pool.put(buf);
    const retrieved = pool.get(64);
    try std.testing.expectEqual(@as(usize, 1), pool.getStats().hits);

    std.testing.allocator.free(retrieved.?);
}

test "pool max buffers per bucket" {
    var pool = BufferPool.init(std.testing.allocator);
    defer pool.deinit();

    // Fill a bucket beyond capacity
    var bufs: [MAX_BUFFERS_PER_BUCKET + 2][]u8 = undefined;
    for (&bufs) |*buf| {
        buf.* = try std.testing.allocator.alloc(u8, 64);
    }

    // Put all buffers - some should be freed
    for (bufs) |buf| {
        pool.put(buf);
    }

    // Should only get MAX_BUFFERS_PER_BUCKET back
    var count: usize = 0;
    while (pool.get(64)) |buf| {
        std.testing.allocator.free(buf);
        count += 1;
    }
    try std.testing.expectEqual(@as(usize, MAX_BUFFERS_PER_BUCKET), count);
}

test "pool clear" {
    var pool = BufferPool.init(std.testing.allocator);
    defer pool.deinit();

    const buf = try std.testing.allocator.alloc(u8, 64);
    pool.put(buf);

    try std.testing.expect(pool.total_pooled_bytes > 0);

    pool.clear();
    try std.testing.expectEqual(@as(usize, 0), pool.total_pooled_bytes);
}

test "pool ignores too small/large allocations" {
    var pool = BufferPool.init(std.testing.allocator);
    defer pool.deinit();

    // Too small - should not be pooled
    try std.testing.expect(pool.get(1) == null);

    // Too large - should not be pooled
    try std.testing.expect(pool.get(MAX_SIZE + 1) == null);
}
