# Compute Arenas

Arena allocators provide fast, bulk deallocation for computation workloads.

## What is an Arena?

An arena allocates memory from a contiguous region and frees everything at once:

```
┌────────────────────────────────────────────────────┐
│                    Arena Buffer                     │
├────────┬────────┬────────┬────────┬───────────────┤
│ Alloc1 │ Alloc2 │ Alloc3 │ Alloc4 │   Free Space  │
├────────┴────────┴────────┴────────┴───────────────┤
│ ◄──────── Used ─────────►│◄──── Available ───────►│
└────────────────────────────────────────────────────┘
```

## ComputeArena

Tenzor provides `ComputeArena` for computation scratch space:

```zig
const memory = @import("tenzor").memory;

var arena = memory.ComputeArena.init(std.heap.page_allocator);
defer arena.deinit();

// Allocate freely
const temp1 = try arena.alloc(f32, 1024);
const temp2 = try arena.alloc(f32, 2048);
const temp3 = try arena.alloc(f32, 512);

// No individual frees needed - reset clears all
arena.reset();
```

## Implementation

```zig
pub const ComputeArena = struct {
    arena: std.heap.ArenaAllocator,

    pub fn init(backing: std.mem.Allocator) Self {
        return .{
            .arena = std.heap.ArenaAllocator.init(backing),
        };
    }

    pub fn deinit(self: *Self) void {
        self.arena.deinit();
    }

    pub fn allocator(self: *Self) std.mem.Allocator {
        return self.arena.allocator();
    }

    pub fn alloc(self: *Self, comptime T: type, count: usize) ![]T {
        return try self.arena.allocator().alloc(T, count);
    }

    pub fn reset(self: *Self) void {
        _ = self.arena.reset(.retain_capacity);
    }
};
```

## Use Cases

### Forward Pass

```zig
var arena = ComputeArena.init(backing);
defer arena.deinit();

for (batches) |batch| {
    // All intermediates allocated from arena
    const h1 = try tz.eval(layer1(batch), arena.allocator());
    const h2 = try tz.eval(layer2(h1), arena.allocator());
    const out = try tz.eval(layer3(h2), arena.allocator());

    // Process output...

    // Clear all at once
    arena.reset();
}
```

### Temporary Workspace

```zig
fn complexComputation(input: []const f32, output: []f32, arena: *ComputeArena) !void {
    // Workspace is automatically cleaned up
    const temp1 = try arena.alloc(f32, input.len);
    const temp2 = try arena.alloc(f32, input.len);

    // Use temp buffers...
}
```

### Expression Evaluation

```zig
pub fn evalWithArena(expr: anytype, arena: *ComputeArena) ![]@TypeOf(expr).ElementType {
    // All intermediate buffers from arena
    // Final result copied to separate allocation

    const result = try arena.alloc(@TypeOf(expr).ElementType, expr.numel());
    evalInto(expr, result, arena);
    return result;
}
```

## Comparison with Pool

| Feature | Buffer Pool | Arena |
|---------|-------------|-------|
| Allocation speed | Fast (after warmup) | Very fast |
| Deallocation | Per-buffer | Bulk reset |
| Memory reuse | Same-size buffers | All allocations |
| Fragmentation | Low (buckets) | None |
| Best for | Repeated same-size | Batch processing |

## Best Practices

### Scope Arena to Computation

```zig
fn processRequest(request: Request) !Response {
    var arena = ComputeArena.init(backing);
    defer arena.deinit();

    // All computation uses arena
    const result = try compute(request, &arena);

    // Copy result out before arena is freed
    return try copyToResponse(result);
}
```

### Reset Between Iterations

```zig
var arena = ComputeArena.init(backing);
defer arena.deinit();

for (items) |item| {
    const result = try process(item, &arena);
    output.append(result);

    arena.reset();  // Clear for next item
}
```

### Pre-size for Large Workloads

```zig
// Estimate total workspace needed
const workspace_size = estimateWorkspace(model);

// Pre-allocate to avoid resizing
var arena = ComputeArena.initWithCapacity(backing, workspace_size);
```

## Memory Growth

The arena grows as needed:

```
Initial: 4KB
After first batch: 16KB
After second batch: 16KB (reused)
Large batch: 64KB (expanded)
```

With `reset(.retain_capacity)`, memory is retained but marked as free.

## Thread Safety

Each thread should have its own arena:

```zig
// Per-thread arena
threadlocal var arena: ?ComputeArena = null;

fn getArena() *ComputeArena {
    if (arena == null) {
        arena = ComputeArena.init(backing);
    }
    return &arena.?;
}
```

## Next Steps

- [Buffer Pooling](./pooling.md) - Alternative strategy
- [Threading](../threading/thread-pool.md) - Per-thread allocation
