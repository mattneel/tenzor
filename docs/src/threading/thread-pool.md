# Thread Pool

Tenzor provides a thread pool for parallel tensor operations.

## ThreadPool

```zig
const threading = @import("tenzor").backend.cpu.threading;

// Create thread pool
const pool = try threading.ThreadPool.create(allocator, .{});
defer pool.destroy();

// Use for parallel operations
pool.parallelFor(0, n, chunk_size, context, workFn);
```

## Creation

### Heap Allocation (Recommended)

```zig
const pool = try ThreadPool.create(allocator, .{});
defer pool.destroy();
```

### In-Place Initialization

For advanced use when you control memory:

```zig
var pool_storage: ThreadPool = undefined;
pool_storage.allocator = allocator;
try pool_storage.init(.{});
defer pool_storage.deinit();
```

**Warning:** The pool must not be moved after `init()` due to internal self-references.

## Configuration

```zig
pub const ThreadPoolConfig = struct {
    /// Number of worker threads. null = auto-detect from CPU count.
    thread_count: ?u32 = null,

    /// Stack size per worker thread.
    stack_size: usize = 16 * 1024 * 1024,  // 16 MB
};

// Examples
const pool = try ThreadPool.create(allocator, .{});  // Auto threads
const pool = try ThreadPool.create(allocator, .{ .thread_count = 4 });  // 4 threads
```

## Thread Count

By default, uses available CPU cores:

```zig
const cpu_count = std.Thread.getCpuCount() catch 4;
```

Query pool size:

```zig
const n_threads = pool.getThreadCount();
std.debug.print("Using {} threads\n", .{n_threads});
```

## Lifecycle

```
create() → Use → destroy()

Or:

Stack var → init() → Use → deinit()
```

## Implementation Details

Based on `std.Thread.Pool`:

```zig
pub const ThreadPool = struct {
    pool: std.Thread.Pool,
    thread_count: u32,
    allocator: std.mem.Allocator,

    pub fn create(allocator: std.mem.Allocator, config: ThreadPoolConfig) !*Self {
        const self = try allocator.create(Self);
        self.allocator = allocator;
        errdefer allocator.destroy(self);
        try self.init(config);
        return self;
    }

    pub fn init(self: *Self, config: ThreadPoolConfig) !void {
        const cpu_count: u32 = @intCast(std.Thread.getCpuCount() catch 4);
        self.thread_count = config.thread_count orelse cpu_count;

        try self.pool.init(.{
            .allocator = self.allocator,
            .n_jobs = self.thread_count,
            .stack_size = config.stack_size,
        });
    }

    pub fn destroy(self: *Self) void {
        const allocator = self.allocator;
        self.deinit();
        allocator.destroy(self);
    }
};
```

## Best Practices

### One Pool Per Application

```zig
// Global or long-lived pool
var global_pool: ?*ThreadPool = null;

pub fn getPool() *ThreadPool {
    if (global_pool == null) {
        global_pool = ThreadPool.create(allocator, .{}) catch @panic("OOM");
    }
    return global_pool.?;
}
```

### Match Workload

```zig
// CPU-bound: use all cores
const pool = try ThreadPool.create(allocator, .{});

// Memory-bound: fewer threads may be better
const pool = try ThreadPool.create(allocator, .{
    .thread_count = @max(1, cpu_count / 2),
});
```

### Graceful Shutdown

```zig
defer {
    if (global_pool) |p| {
        p.destroy();
        global_pool = null;
    }
}
```

## Next Steps

- [Parallel Execution](./parallel-execution.md) - Using parallelFor
- [Work Partitioning](./partitioning.md) - Dividing work
