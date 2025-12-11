# Work Partitioning

Efficiently divide work across threads.

## partitionWork

Calculate optimal chunk sizes:

```zig
const partition = threading.partitionWork(total_elements, num_threads);

// partition.chunk_size  - Elements per chunk
// partition.num_chunks  - Total number of chunks
// partition.remainder   - Size of last chunk
```

## Algorithm

```zig
pub fn partitionWork(total: usize, num_threads: usize) WorkPartition {
    if (total == 0 or num_threads == 0) {
        return .{ .chunk_size = 0, .num_chunks = 0, .remainder = 0 };
    }

    // Aim for 2 chunks per thread for load balancing
    const target_chunks = num_threads * 2;
    var chunk_size = (total + target_chunks - 1) / target_chunks;

    // Minimum chunk size
    chunk_size = @max(chunk_size, 64);

    const num_chunks = (total + chunk_size - 1) / chunk_size;
    const remainder = total % chunk_size;

    return .{
        .chunk_size = chunk_size,
        .num_chunks = num_chunks,
        .remainder = if (remainder == 0) chunk_size else remainder,
    };
}
```

## WorkPartition

```zig
pub const WorkPartition = struct {
    chunk_size: usize,
    num_chunks: usize,
    remainder: usize,

    /// Get range for chunk i
    pub fn getRange(self: WorkPartition, chunk_idx: usize, total: usize) struct {
        start: usize,
        end: usize,
    } {
        const start = chunk_idx * self.chunk_size;
        const end = @min(start + self.chunk_size, total);
        return .{ .start = start, .end = end };
    }
};
```

## Usage Example

```zig
const total = 1000000;
const num_threads = pool.getThreadCount();

const partition = partitionWork(total, num_threads);

// Process each chunk
for (0..partition.num_chunks) |chunk_idx| {
    const range = partition.getRange(chunk_idx, total);
    // Work on [range.start, range.end)
}
```

## Load Balancing

### Multiple Chunks Per Thread

Creating 2x chunks vs threads helps with load balancing:

```
4 threads, 1000 elements:
  4 chunks: [250] [250] [250] [250]
    ↳ If one chunk is slow, thread sits idle

  8 chunks: [125] [125] [125] [125] [125] [125] [125] [125]
    ↳ Threads can pick up additional chunks
```

### Work Stealing

`std.Thread.Pool` uses work stealing:

1. Thread finishes its chunk
2. Checks queue for more work
3. Takes next available chunk

## Optimal Chunk Sizes

### Too Small

- High overhead from thread synchronization
- Cache thrashing

### Too Large

- Load imbalance
- Wasted parallelism

### Guidelines

| Total Elements | Recommended Chunks |
|---------------|-------------------|
| < 8192 | 1 (sequential) |
| 8K - 64K | num_threads * 2 |
| 64K - 1M | num_threads * 4 |
| > 1M | num_threads * 8 |

## Cache Considerations

### Chunk Alignment

Align chunks to cache line boundaries:

```zig
const CACHE_LINE = 64;
const elements_per_line = CACHE_LINE / @sizeOf(f32);

// Round chunk size to cache line multiple
chunk_size = (chunk_size + elements_per_line - 1) / elements_per_line * elements_per_line;
```

### False Sharing

Avoid threads writing to adjacent memory:

```zig
// Bad: threads write to adjacent array elements
var results: [num_threads]f32 = undefined;  // May share cache lines

// Good: pad to cache line
const Padded = struct {
    value: f32,
    _padding: [60]u8 = undefined,  // Pad to 64 bytes
};
var results: [num_threads]Padded = undefined;
```

## Parallel Reduce

Partitioning for reductions:

```zig
pub fn parallelReduce(
    comptime T: type,
    pool: *ThreadPool,
    data: []const T,
    initial: T,
    comptime reduce_fn: fn (T, T) T,
    allocator: std.mem.Allocator,
) !T {
    const partition = partitionWork(data.len, pool.thread_count);

    // Allocate partial results
    const partial = try allocator.alloc(T, partition.num_chunks);
    defer allocator.free(partial);

    // Each thread reduces its chunk
    pool.parallelFor(0, partition.num_chunks, 1, .{
        .data = data,
        .partial = partial,
        .partition = partition,
        .initial = initial,
    }, struct {
        fn work(ctx: anytype, chunk_idx: usize, _: usize) void {
            const range = ctx.partition.getRange(chunk_idx, ctx.data.len);
            var acc = ctx.initial;
            for (ctx.data[range.start..range.end]) |x| {
                acc = reduce_fn(acc, x);
            }
            ctx.partial[chunk_idx] = acc;
        }
    }.work);

    // Combine partial results
    var result = initial;
    for (partial) |p| {
        result = reduce_fn(result, p);
    }
    return result;
}
```

## Next Steps

- [Thread Pool](./thread-pool.md) - Pool management
- [Parallel Execution](./parallel-execution.md) - Using parallelFor
