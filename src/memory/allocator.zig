//! Tensor-aware memory allocator with buffer pooling.
//!
//! Provides an allocator interface that caches recently freed buffers
//! for reuse, reducing allocation overhead during tensor computation.

const std = @import("std");
const pool_mod = @import("pool.zig");

const BufferPool = pool_mod.BufferPool;

/// Tensor allocator with buffer pooling.
/// Use this allocator for tensor data to benefit from buffer reuse.
pub const TensorAllocator = struct {
    /// The pool for buffer reuse
    pool: BufferPool,
    /// Backing allocator for new allocations
    backing: std.mem.Allocator,
    /// Statistics
    allocations: usize,
    deallocations: usize,
    bytes_allocated: usize,
    bytes_freed: usize,
    pool_hits: usize,
    pool_misses: usize,

    const Self = @This();

    /// Initialize a new tensor allocator.
    pub fn init(backing: std.mem.Allocator) Self {
        return Self{
            .pool = BufferPool.init(backing),
            .backing = backing,
            .allocations = 0,
            .deallocations = 0,
            .bytes_allocated = 0,
            .bytes_freed = 0,
            .pool_hits = 0,
            .pool_misses = 0,
        };
    }

    /// Deinitialize, freeing all pooled buffers.
    pub fn deinit(self: *Self) void {
        self.pool.deinit();
    }

    /// Allocate memory for a tensor.
    pub fn alloc(self: *Self, comptime T: type, count: usize) ![]T {
        const byte_count = count * @sizeOf(T);

        // Try pool first
        if (self.pool.get(byte_count)) |buf| {
            self.pool_hits += 1;
            self.allocations += 1;
            self.bytes_allocated += byte_count;
            return @as([*]T, @ptrCast(@alignCast(buf.ptr)))[0..count];
        }

        // Fall back to backing allocator
        self.pool_misses += 1;
        const result = try self.backing.alloc(T, count);
        self.allocations += 1;
        self.bytes_allocated += byte_count;
        return result;
    }

    /// Free memory, returning it to the pool if possible.
    pub fn free(self: *Self, buf: anytype) void {
        const T = std.meta.Elem(@TypeOf(buf));
        const alignment = comptime std.mem.Alignment.of(T);
        const bytes = std.mem.sliceAsBytes(buf);
        self.deallocations += 1;
        self.bytes_freed += bytes.len;

        // Try to pool it with original alignment
        self.pool.putAligned(@constCast(bytes), alignment);
    }

    /// Get an allocator interface.
    /// This allows TensorAllocator to be used with Zig's standard allocator patterns.
    pub fn allocator(self: *Self) std.mem.Allocator {
        return .{
            .ptr = self,
            .vtable = &vtable,
        };
    }

    const vtable = std.mem.Allocator.VTable{
        .alloc = allocFn,
        .resize = resizeFn,
        .remap = remapFn,
        .free = freeFn,
    };

    fn remapFn(ctx: *anyopaque, buf: []u8, buf_align: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        _ = ctx;
        _ = buf;
        _ = buf_align;
        _ = new_len;
        _ = ret_addr;
        return null;
    }

    fn allocFn(ctx: *anyopaque, len: usize, ptr_align: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *Self = @ptrCast(@alignCast(ctx));

        // Try pool first
        if (self.pool.get(len)) |buf| {
            self.pool_hits += 1;
            self.allocations += 1;
            self.bytes_allocated += len;
            return buf.ptr;
        }

        // Fall back to backing allocator
        self.pool_misses += 1;
        const result = self.backing.rawAlloc(len, ptr_align, ret_addr);
        if (result != null) {
            self.allocations += 1;
            self.bytes_allocated += len;
        }
        return result;
    }

    fn resizeFn(ctx: *anyopaque, buf: []u8, buf_align: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        _ = ctx;
        _ = buf;
        _ = buf_align;
        _ = new_len;
        _ = ret_addr;
        // Pooled allocator doesn't support resize
        return false;
    }

    fn freeFn(ctx: *anyopaque, buf: []u8, buf_align: std.mem.Alignment, ret_addr: usize) void {
        _ = ret_addr;
        const self: *Self = @ptrCast(@alignCast(ctx));
        self.deallocations += 1;
        self.bytes_freed += buf.len;
        self.pool.putAligned(buf, buf_align);
    }

    /// Get allocation statistics.
    pub fn getStats(self: *const Self) Stats {
        const pool_stats = self.pool.getStats();
        return Stats{
            .allocations = self.allocations,
            .deallocations = self.deallocations,
            .bytes_allocated = self.bytes_allocated,
            .bytes_freed = self.bytes_freed,
            .bytes_in_use = self.bytes_allocated - self.bytes_freed,
            .pool_hits = pool_stats.hits,
            .pool_misses = pool_stats.misses,
            .pool_hit_rate = pool_stats.hit_rate,
            .pooled_bytes = pool_stats.total_pooled_bytes,
        };
    }

    /// Clear the buffer pool, freeing all cached buffers.
    pub fn clearPool(self: *Self) void {
        self.pool.clear();
    }
};

/// Allocator statistics.
pub const Stats = struct {
    allocations: usize,
    deallocations: usize,
    bytes_allocated: usize,
    bytes_freed: usize,
    bytes_in_use: usize,
    pool_hits: usize,
    pool_misses: usize,
    pool_hit_rate: f64,
    pooled_bytes: usize,
};

/// Arena allocator for a computation graph evaluation.
/// Provides a scratch space that can be reset between evaluations.
pub const ComputeArena = struct {
    /// Fixed buffer for small allocations
    buffer: []u8,
    /// Current position in buffer
    offset: usize,
    /// Fallback allocator for large allocations
    fallback: std.mem.Allocator,
    /// Track large allocations for cleanup
    large_allocs: std.ArrayListUnmanaged([]u8),

    const Self = @This();
    const ARENA_SIZE = 64 * 1024; // 64KB fixed arena

    /// Initialize with backing allocator.
    pub fn init(backing: std.mem.Allocator) !Self {
        return Self{
            .buffer = try backing.alloc(u8, ARENA_SIZE),
            .offset = 0,
            .fallback = backing,
            .large_allocs = .{},
        };
    }

    /// Deinitialize, freeing all memory.
    pub fn deinit(self: *Self) void {
        for (self.large_allocs.items) |buf| {
            self.fallback.free(buf);
        }
        self.large_allocs.deinit(self.fallback);
        self.fallback.free(self.buffer);
    }

    /// Allocate from the arena.
    pub fn alloc(self: *Self, comptime T: type, count: usize) ![]T {
        const byte_count = count * @sizeOf(T);
        const alignment = @alignOf(T);

        // Align offset
        const aligned_offset = std.mem.alignForward(usize, self.offset, alignment);

        // Check if it fits in fixed buffer
        if (aligned_offset + byte_count <= self.buffer.len) {
            const ptr = self.buffer[aligned_offset..][0..byte_count];
            self.offset = aligned_offset + byte_count;
            return @as([*]T, @ptrCast(@alignCast(ptr.ptr)))[0..count];
        }

        // Fall back to allocator for large allocations
        const result = try self.fallback.alloc(T, count);
        try self.large_allocs.append(self.fallback, std.mem.sliceAsBytes(result));
        return result;
    }

    /// Reset the arena, freeing all allocations.
    /// The fixed buffer is retained for reuse.
    pub fn reset(self: *Self) void {
        self.offset = 0;
        for (self.large_allocs.items) |buf| {
            self.fallback.free(buf);
        }
        self.large_allocs.clearRetainingCapacity();
    }

    /// Get an allocator interface.
    pub fn allocator(self: *Self) std.mem.Allocator {
        return .{
            .ptr = self,
            .vtable = &arena_vtable,
        };
    }

    const arena_vtable = std.mem.Allocator.VTable{
        .alloc = arenaAllocFn,
        .resize = arenaResizeFn,
        .remap = arenaRemapFn,
        .free = arenaFreeFn,
    };

    fn arenaRemapFn(ctx: *anyopaque, buf: []u8, buf_align: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        _ = ctx;
        _ = buf;
        _ = buf_align;
        _ = new_len;
        _ = ret_addr;
        return null;
    }

    fn arenaAllocFn(ctx: *anyopaque, len: usize, ptr_align: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *Self = @ptrCast(@alignCast(ctx));
        const alignment: usize = @as(usize, 1) << @intFromEnum(ptr_align);

        const aligned_offset = std.mem.alignForward(usize, self.offset, alignment);

        if (aligned_offset + len <= self.buffer.len) {
            const ptr = self.buffer[aligned_offset..][0..len];
            self.offset = aligned_offset + len;
            return ptr.ptr;
        }

        // Fall back
        const result = self.fallback.rawAlloc(len, ptr_align, ret_addr);
        if (result) |ptr| {
            self.large_allocs.append(self.fallback, ptr[0..len]) catch return null;
        }
        return result;
    }

    fn arenaResizeFn(ctx: *anyopaque, buf: []u8, buf_align: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        _ = ctx;
        _ = buf;
        _ = buf_align;
        _ = new_len;
        _ = ret_addr;
        return false;
    }

    fn arenaFreeFn(ctx: *anyopaque, buf: []u8, buf_align: std.mem.Alignment, ret_addr: usize) void {
        _ = ctx;
        _ = buf;
        _ = buf_align;
        _ = ret_addr;
        // Arena doesn't free individual allocations - use reset() instead
    }

    /// Get current usage statistics.
    pub fn getUsage(self: *const Self) Usage {
        var large_bytes: usize = 0;
        for (self.large_allocs.items) |buf| {
            large_bytes += buf.len;
        }
        return Usage{
            .buffer_used = self.offset,
            .buffer_size = self.buffer.len,
            .large_allocs = self.large_allocs.items.len,
            .large_bytes = large_bytes,
            .total_bytes = self.offset + large_bytes,
        };
    }
};

/// Arena usage statistics.
pub const Usage = struct {
    buffer_used: usize,
    buffer_size: usize,
    large_allocs: usize,
    large_bytes: usize,
    total_bytes: usize,
};

// ============================================================================
// Tests
// ============================================================================

test "tensor allocator basic" {
    var alloc = TensorAllocator.init(std.testing.allocator);
    defer alloc.deinit();

    const buf = try alloc.alloc(f32, 16);
    alloc.free(buf);

    // Second allocation should hit pool
    const buf2 = try alloc.alloc(f32, 16);
    alloc.free(buf2);

    const stats = alloc.getStats();
    try std.testing.expectEqual(@as(usize, 2), stats.allocations);
    try std.testing.expect(stats.pool_hits > 0);
}

test "tensor allocator interface" {
    var tensor_alloc = TensorAllocator.init(std.testing.allocator);
    defer tensor_alloc.deinit();

    const alloc = tensor_alloc.allocator();

    const buf = try alloc.alloc(f32, 16);
    alloc.free(buf);

    const buf2 = try alloc.alloc(f32, 16);
    alloc.free(buf2);
}

test "compute arena basic" {
    var arena = try ComputeArena.init(std.testing.allocator);
    defer arena.deinit();

    const buf1 = try arena.alloc(f32, 16);
    const buf2 = try arena.alloc(f32, 16);

    // Verify different pointers
    try std.testing.expect(@intFromPtr(buf1.ptr) != @intFromPtr(buf2.ptr));

    const usage = arena.getUsage();
    try std.testing.expect(usage.buffer_used > 0);

    arena.reset();
    try std.testing.expectEqual(@as(usize, 0), arena.getUsage().buffer_used);
}

test "compute arena large allocation" {
    var arena = try ComputeArena.init(std.testing.allocator);
    defer arena.deinit();

    // Allocate more than arena buffer size
    const large = try arena.alloc(u8, ComputeArena.ARENA_SIZE + 1000);
    try std.testing.expectEqual(@as(usize, ComputeArena.ARENA_SIZE + 1000), large.len);

    const usage = arena.getUsage();
    try std.testing.expectEqual(@as(usize, 1), usage.large_allocs);
}

test "compute arena allocator interface" {
    var arena = try ComputeArena.init(std.testing.allocator);
    defer arena.deinit();

    const alloc = arena.allocator();

    const buf = try alloc.alloc(f32, 16);
    _ = buf;

    // Free is a no-op for arena
    // Memory is freed on reset or deinit
}
