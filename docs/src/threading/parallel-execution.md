# Parallel Execution

Execute operations in parallel using the thread pool.

## parallelFor

Divide a range across threads:

```zig
pool.parallelFor(
    start,      // Range start
    end,        // Range end (exclusive)
    chunk_size, // Elements per chunk (or null for default)
    context,    // Data passed to work function
    workFn,     // Function to execute
);
```

## Basic Example

```zig
const pool = try ThreadPool.create(allocator, .{});
defer pool.destroy();

var results: [1000]f32 = undefined;

const Context = struct {
    input: []const f32,
    output: []f32,
};

pool.parallelFor(0, 1000, null, Context{
    .input = &input_data,
    .output = &results,
}, struct {
    fn work(ctx: Context, start: usize, end: usize) void {
        for (start..end) |i| {
            ctx.output[i] = @exp(ctx.input[i]);
        }
    }
}.work);
```

## Sequential Fallback

Small workloads run sequentially:

```zig
// If total < MIN_PARALLEL_SIZE or thread_count <= 1,
// executes on main thread without spawning
```

Threshold: `MIN_PARALLEL_SIZE = 8192` elements

## Chunk Size

### Default

```zig
const DEFAULT_CHUNK_SIZE = 4096;

// Use default
pool.parallelFor(0, n, null, ctx, work);
```

### Custom

```zig
// Smaller chunks: more parallelism, more overhead
pool.parallelFor(0, n, 1024, ctx, work);

// Larger chunks: less overhead, potential load imbalance
pool.parallelFor(0, n, 16384, ctx, work);
```

### Guidelines

| Workload | Recommended Chunk |
|----------|------------------|
| Light ops (add, mul) | 8192+ |
| Medium ops (exp, log) | 4096 |
| Heavy ops (matmul tile) | 1024 |

## Context Pattern

### Immutable Context

```zig
const Context = struct {
    input: []const f32,
    scale: f32,
};

pool.parallelFor(0, n, null, Context{
    .input = data,
    .scale = 2.0,
}, struct {
    fn work(ctx: Context, start: usize, end: usize) void {
        // Read ctx.input, use ctx.scale
    }
}.work);
```

### Mutable Output

```zig
const Context = struct {
    input: []const f32,
    output: []f32,  // Threads write to disjoint regions
};

pool.parallelFor(0, n, null, ctx, struct {
    fn work(c: Context, start: usize, end: usize) void {
        for (start..end) |i| {
            c.output[i] = process(c.input[i]);
        }
    }
}.work);
```

### Per-Thread Accumulator

```zig
const Context = struct {
    data: []const f32,
    partial_sums: []f32,  // One per chunk
    chunk_size: usize,
};

pool.parallelFor(0, num_chunks, 1, ctx, struct {
    fn work(c: Context, chunk_idx: usize, _: usize) void {
        const start = chunk_idx * c.chunk_size;
        const end = @min(start + c.chunk_size, c.data.len);

        var sum: f32 = 0;
        for (c.data[start..end]) |x| {
            sum += x;
        }
        c.partial_sums[chunk_idx] = sum;
    }
}.work);

// Combine partial sums
var total: f32 = 0;
for (ctx.partial_sums) |s| total += s;
```

## SIMD + Threads

Combine threading with SIMD:

```zig
fn work(ctx: Context, start: usize, end: usize) void {
    const vec_len = simd.suggestVectorLength(f32);
    var i = start;

    // SIMD within each thread's chunk
    while (i + vec_len <= end) : (i += vec_len) {
        const v = simd.load(f32, ctx.input[i..]);
        const result = simd.exp(f32, v);
        simd.store(f32, result, ctx.output[i..]);
    }

    // Scalar remainder
    while (i < end) : (i += 1) {
        ctx.output[i] = @exp(ctx.input[i]);
    }
}
```

## WaitGroup Integration

```zig
var wg: std.Thread.WaitGroup = .{};

// parallelFor uses WaitGroup internally
// Equivalent to:
var chunk_start = start;
while (chunk_start < end) {
    const chunk_end = @min(chunk_start + chunk_size, end);
    pool.pool.spawnWg(&wg, workFn, .{ context, chunk_start, chunk_end });
    chunk_start = chunk_end;
}
pool.pool.waitAndWork(&wg);
```

## Error Handling

Work functions cannot return errors. Handle errors via context:

```zig
const Context = struct {
    // ... data fields ...
    error_flag: *std.atomic.Value(bool),
};

fn work(ctx: Context, start: usize, end: usize) void {
    if (ctx.error_flag.load(.acquire)) return;  // Early exit

    for (start..end) |i| {
        if (somethingFailed()) {
            ctx.error_flag.store(true, .release);
            return;
        }
    }
}
```

## Next Steps

- [Work Partitioning](./partitioning.md) - Optimal work division
- [Thread Pool](./thread-pool.md) - Pool configuration
