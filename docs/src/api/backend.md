# Backend API Reference

## Module Structure

```zig
const backend = @import("tenzor").backend;
const cpu = backend.cpu;
```

---

## Evaluation

### `eval(expr, allocator) !ResultTensor`

Evaluate expression graph.

```zig
const result = try cpu.eval(expression, allocator);
defer result.deinit();
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `expr` | Expression | Expression to evaluate |
| `allocator` | `Allocator` | Memory allocator |

**Returns:** `ResultTensor` - Materialized tensor with result data.

---

## SIMD

### `simd.suggestVectorLength(T) comptime_int`

Get optimal SIMD vector length for type.

```zig
const vec_len = simd.suggestVectorLength(f32);  // 8 on AVX2
```

### `simd.load(T, slice) @Vector`

Load values into SIMD vector.

```zig
const v = simd.load(f32, data[i..][0..8]);
```

### `simd.store(T, vector, slice) void`

Store SIMD vector to memory.

```zig
simd.store(f32, result_vec, output[i..][0..8]);
```

### `simd.splat(T, len, value) @Vector`

Create vector filled with value.

```zig
const zeros = simd.splat(f32, 8, 0.0);
```

### Vectorized Operations

```zig
simd.exp(f32, 8, v)      // Element-wise exp
simd.log(f32, 8, v)      // Element-wise log
simd.tanh(f32, 8, v)     // Element-wise tanh
simd.sigmoid(f32, 8, v)  // Element-wise sigmoid
simd.relu(f32, 8, v)     // Element-wise relu
```

---

## Threading

### ThreadPool

```zig
const threading = cpu.threading;
```

#### `ThreadPool.create(allocator, config) !*ThreadPool`

Create thread pool.

```zig
const pool = try ThreadPool.create(allocator, .{});
defer pool.destroy();
```

#### `ThreadPool.destroy() void`

Destroy pool and free resources.

#### `ThreadPool.getThreadCount() u32`

Get number of worker threads.

```zig
const n = pool.getThreadCount();
```

#### `ThreadPool.parallelFor(start, end, chunk_size, context, work_fn) void`

Execute parallel for loop.

```zig
pool.parallelFor(0, 1000, null, context, struct {
    fn work(ctx: Context, start: usize, end: usize) void {
        for (start..end) |i| {
            ctx.output[i] = process(ctx.input[i]);
        }
    }
}.work);
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `start` | `usize` | Range start (inclusive) |
| `end` | `usize` | Range end (exclusive) |
| `chunk_size` | `?usize` | Elements per chunk (null = default) |
| `context` | `anytype` | Data passed to work function |
| `work_fn` | `fn` | Function to execute |

### ThreadPoolConfig

```zig
pub const ThreadPoolConfig = struct {
    thread_count: ?u32 = null,      // null = auto-detect
    stack_size: usize = 16 * 1024 * 1024,  // 16 MB
};
```

### Work Partitioning

#### `partitionWork(total, num_threads) WorkPartition`

Calculate optimal chunk sizes.

```zig
const partition = partitionWork(1000000, 8);
// partition.chunk_size
// partition.num_chunks
// partition.remainder
```

#### `WorkPartition.getRange(chunk_idx, total) Range`

Get index range for chunk.

```zig
const range = partition.getRange(chunk_idx, total);
// range.start, range.end
```

---

## Dispatch

### `dispatch(expr, output) void`

Dispatch expression to kernel.

```zig
cpu.dispatch(expr, output_buffer);
```

### Kernel Selection

The dispatcher selects kernels based on:

1. Expression type (unary, binary, matmul, reduce)
2. Element type (f32, f64, etc.)
3. Available SIMD support

---

## Fusion Engine

### Pattern Detection

```zig
const fusion = cpu.fusion;
const pattern = fusion.detectPattern(expr);
```

**Returns:**

```zig
pub const FusionPattern = enum {
    none,
    elementwise_chain,
    matmul_epilogue,
    reduce_epilogue,
};
```

### Fused Execution

```zig
if (pattern != .none) {
    fusion.executeFused(pattern, expr, output);
} else {
    cpu.dispatch(expr, output);
}
```

---

## Memory

### PoolAllocator

```zig
const memory = @import("tenzor").memory;
```

#### `PoolAllocator.init(backing_allocator) PoolAllocator`

Create pooling allocator.

```zig
var pool_alloc = PoolAllocator.init(std.heap.page_allocator);
defer pool_alloc.deinit();
```

#### `PoolAllocator.allocator() Allocator`

Get std.mem.Allocator interface.

```zig
const alloc = pool_alloc.allocator();
```

#### `PoolAllocator.alloc(T, n) ![]T`

Allocate typed slice.

```zig
const data = try pool_alloc.alloc(f32, 1000);
defer pool_alloc.free(data);
```

#### `PoolAllocator.free(slice) void`

Return allocation to pool.

```zig
pool_alloc.free(data);
```

#### `PoolAllocator.clear() void`

Return all allocations to pool without freeing underlying memory.

```zig
pool_alloc.clear();
```

#### `PoolAllocator.deinit() void`

Free all memory and destroy pool.

```zig
pool_alloc.deinit();
```

### BufferPool

Lower-level buffer pooling.

#### `BufferPool.init(allocator) BufferPool`

```zig
var pool = BufferPool.init(allocator);
defer pool.deinit();
```

#### `BufferPool.get(size) ?[]u8`

Get buffer from pool.

```zig
if (pool.get(1024)) |buf| {
    // Use buf
    pool.put(buf);
}
```

#### `BufferPool.put(buf) void`

Return buffer to pool.

#### `BufferPool.getOrAlloc(size) ![]u8`

Get from pool or allocate.

```zig
const buf = try pool.getOrAlloc(1024);
defer pool.put(buf);
```
