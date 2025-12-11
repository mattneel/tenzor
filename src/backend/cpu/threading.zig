//! Thread pool and parallel execution utilities for CPU backend.
//!
//! Provides parallelization primitives for tensor operations,
//! including parallel for loops and work partitioning.

const std = @import("std");
const builtin = @import("builtin");

/// Default chunk size for parallel operations (in elements).
/// Chosen to balance overhead vs utilization.
const DEFAULT_CHUNK_SIZE = 4096;

/// Minimum work size to justify parallelization.
/// Below this, sequential execution is faster.
const MIN_PARALLEL_SIZE = 8192;

/// Configuration for the thread pool.
pub const ThreadPoolConfig = struct {
    /// Number of worker threads. null means auto-detect from CPU count.
    thread_count: ?u32 = null,
    /// Stack size per worker thread in bytes.
    stack_size: usize = 16 * 1024 * 1024, // 16MB default
};

/// A simple thread pool for parallel tensor operations.
/// Note: This struct must not be moved after init due to self-referential pointers.
/// Use create() to allocate on heap, or init() with a stable pointer.
pub const ThreadPool = struct {
    pool: std.Thread.Pool,
    thread_count: u32,
    allocator: std.mem.Allocator,

    const Self = @This();

    /// Initialize the thread pool in place (struct must not be moved after this).
    pub fn init(self: *Self, config: ThreadPoolConfig) !void {
        const cpu_count: u32 = @intCast(std.Thread.getCpuCount() catch 4);
        self.thread_count = config.thread_count orelse cpu_count;

        try self.pool.init(.{
            .allocator = self.allocator,
            .n_jobs = self.thread_count,
            .stack_size = config.stack_size,
        });
    }

    /// Create a heap-allocated thread pool.
    pub fn create(allocator: std.mem.Allocator, config: ThreadPoolConfig) !*Self {
        const self = try allocator.create(Self);
        self.allocator = allocator;
        errdefer allocator.destroy(self);
        try self.init(config);
        return self;
    }

    /// Deinitialize the thread pool.
    pub fn deinit(self: *Self) void {
        self.pool.deinit();
    }

    /// Destroy a heap-allocated thread pool.
    pub fn destroy(self: *Self) void {
        const allocator = self.allocator;
        self.deinit();
        allocator.destroy(self);
    }

    /// Execute a function in parallel over a range [start, end).
    /// The function receives (chunk_start, chunk_end) for each chunk.
    pub fn parallelFor(
        self: *Self,
        start: usize,
        end: usize,
        chunk_size_opt: ?usize,
        context: anytype,
        comptime func: fn (@TypeOf(context), usize, usize) void,
    ) void {
        const total = end - start;
        if (total == 0) return;

        const chunk_size = chunk_size_opt orelse DEFAULT_CHUNK_SIZE;

        // For small workloads or single-threaded, execute sequentially
        if (builtin.single_threaded or total < MIN_PARALLEL_SIZE or self.thread_count <= 1) {
            func(context, start, end);
            return;
        }

        // Spawn work for each chunk
        var wg: std.Thread.WaitGroup = .{};

        var chunk_start = start;
        while (chunk_start < end) {
            const chunk_end = @min(chunk_start + chunk_size, end);
            const cs = chunk_start;
            const ce = chunk_end;

            self.pool.spawnWg(&wg, struct {
                fn work(ctx: @TypeOf(context), s: usize, e: usize) void {
                    func(ctx, s, e);
                }
            }.work, .{ context, cs, ce });

            chunk_start = chunk_end;
        }

        // Wait for all work to complete
        self.pool.waitAndWork(&wg);
    }

    /// Get the number of threads in the pool.
    pub fn getThreadCount(self: *const Self) u32 {
        return self.thread_count;
    }

    /// Execute a function in parallel over batches where each batch item is significant work.
    /// Unlike parallelFor, this doesn't check MIN_PARALLEL_SIZE since batch-level
    /// parallelism is worth it when batch >= thread_count.
    pub fn parallelForBatch(
        self: *Self,
        batch_size: usize,
        context: anytype,
        comptime func: fn (@TypeOf(context), usize, usize) void,
    ) void {
        if (batch_size == 0) return;

        // For single-threaded or tiny batches, run sequentially
        if (builtin.single_threaded or self.thread_count <= 1 or batch_size < 2) {
            func(context, 0, batch_size);
            return;
        }

        // Divide batches among threads
        const batches_per_thread = (batch_size + self.thread_count - 1) / self.thread_count;

        var wg: std.Thread.WaitGroup = .{};
        var batch_start: usize = 0;

        while (batch_start < batch_size) {
            const batch_end = @min(batch_start + batches_per_thread, batch_size);
            const bs = batch_start;
            const be = batch_end;

            self.pool.spawnWg(&wg, struct {
                fn work(ctx: @TypeOf(context), s: usize, e: usize) void {
                    func(ctx, s, e);
                }
            }.work, .{ context, bs, be });

            batch_start = batch_end;
        }

        self.pool.waitAndWork(&wg);
    }
};

/// Partition work into chunks suitable for parallel execution.
pub fn partitionWork(total: usize, num_threads: usize) WorkPartition {
    if (total == 0 or num_threads == 0) {
        return WorkPartition{
            .chunk_size = 0,
            .num_chunks = 0,
            .remainder = 0,
        };
    }

    // Aim for at least 2 chunks per thread to allow load balancing
    const target_chunks = num_threads * 2;
    var chunk_size = (total + target_chunks - 1) / target_chunks;

    // Ensure minimum chunk size
    chunk_size = @max(chunk_size, 64);

    const num_chunks = (total + chunk_size - 1) / chunk_size;
    const remainder = total % chunk_size;

    return WorkPartition{
        .chunk_size = chunk_size,
        .num_chunks = num_chunks,
        .remainder = if (remainder == 0) chunk_size else remainder,
    };
}

/// Describes how work should be partitioned.
pub const WorkPartition = struct {
    chunk_size: usize,
    num_chunks: usize,
    remainder: usize, // Size of last chunk

    /// Get the range [start, end) for chunk i.
    pub fn getRange(self: WorkPartition, chunk_idx: usize, total: usize) struct { start: usize, end: usize } {
        const start = chunk_idx * self.chunk_size;
        const end = @min(start + self.chunk_size, total);
        return .{ .start = start, .end = end };
    }
};

/// Execute a parallel reduction operation.
/// Each thread computes a partial result, then results are combined.
pub fn parallelReduce(
    comptime T: type,
    pool: *ThreadPool,
    data: []const T,
    initial: T,
    comptime reduce_fn: fn (T, T) T,
    allocator: std.mem.Allocator,
) !T {
    const total = data.len;
    if (total == 0) return initial;

    // For small data, sequential reduction
    if (total < MIN_PARALLEL_SIZE or pool.thread_count <= 1) {
        var result = initial;
        for (data) |x| {
            result = reduce_fn(result, x);
        }
        return result;
    }

    // Allocate space for partial results
    const partial_results = try allocator.alloc(T, pool.thread_count);
    defer allocator.free(partial_results);

    // Initialize partial results
    for (partial_results) |*p| {
        p.* = initial;
    }

    // Partition work
    const partition = partitionWork(total, pool.thread_count);

    // Each thread reduces its chunk
    const Context = struct {
        data: []const T,
        partial_results: []T,
        partition: WorkPartition,
        initial: T,
    };

    const ctx = Context{
        .data = data,
        .partial_results = partial_results,
        .partition = partition,
        .initial = initial,
    };

    pool.parallelFor(0, partition.num_chunks, 1, ctx, struct {
        fn work(c: Context, chunk_start: usize, chunk_end: usize) void {
            _ = chunk_end;
            const range = c.partition.getRange(chunk_start, c.data.len);
            var acc = c.initial;
            for (c.data[range.start..range.end]) |x| {
                acc = reduce_fn(acc, x);
            }
            c.partial_results[chunk_start] = acc;
        }
    }.work);

    // Combine partial results
    var result = initial;
    for (partial_results[0..partition.num_chunks]) |p| {
        result = reduce_fn(result, p);
    }

    return result;
}

// ============================================================================
// Tests
// ============================================================================

test "thread pool creation" {
    const pool = try ThreadPool.create(std.testing.allocator, .{});
    defer pool.destroy();

    try std.testing.expect(pool.thread_count > 0);
}

test "parallel for sequential fallback" {
    const pool = try ThreadPool.create(std.testing.allocator, .{ .thread_count = 1 });
    defer pool.destroy();

    var sum: usize = 0;
    const Context = struct { sum: *usize };
    pool.parallelFor(0, 100, 10, Context{ .sum = &sum }, struct {
        fn work(ctx: Context, start: usize, end: usize) void {
            for (start..end) |i| {
                ctx.sum.* += i;
            }
        }
    }.work);

    // Sum of 0..99 = 99 * 100 / 2 = 4950
    try std.testing.expectEqual(@as(usize, 4950), sum);
}

test "work partition" {
    const partition = partitionWork(1000, 4);
    try std.testing.expect(partition.chunk_size > 0);
    try std.testing.expect(partition.num_chunks > 0);

    // Verify ranges cover all elements
    var covered: usize = 0;
    for (0..partition.num_chunks) |i| {
        const range = partition.getRange(i, 1000);
        covered += range.end - range.start;
    }
    try std.testing.expectEqual(@as(usize, 1000), covered);
}

test "work partition edge cases" {
    // Empty
    const empty = partitionWork(0, 4);
    try std.testing.expectEqual(@as(usize, 0), empty.num_chunks);

    // Single element
    const single = partitionWork(1, 4);
    try std.testing.expect(single.num_chunks >= 1);
}

test "parallel reduce sequential fallback" {
    const pool = try ThreadPool.create(std.testing.allocator, .{ .thread_count = 1 });
    defer pool.destroy();

    const data = [_]f32{ 1, 2, 3, 4, 5 };
    const sum = try parallelReduce(
        f32,
        pool,
        &data,
        0,
        struct {
            fn add(a: f32, b: f32) f32 {
                return a + b;
            }
        }.add,
        std.testing.allocator,
    );

    try std.testing.expectApproxEqAbs(@as(f32, 15), sum, 0.001);
}
