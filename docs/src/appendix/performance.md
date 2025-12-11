# Performance Guide

## Optimization Strategies

### 1. Use Lazy Evaluation

Build expression graphs before evaluating:

```zig
// Good: Single fused evaluation
const result = input
    .matmul(weights)
    .add(bias)
    .relu()
    .eval(allocator);

// Avoid: Multiple separate evaluations
const tmp1 = input.matmul(weights).eval(allocator);
const tmp2 = tmp1.add(bias).eval(allocator);
const result = tmp2.relu().eval(allocator);
```

### 2. Batch Operations

Process data in batches to amortize overhead:

```zig
// Good: Batched
const Batch = Tensor(f32, .{ 32, 784 });  // 32 samples
const result = batch.matmul(weights).eval(allocator);

// Slower: One at a time
for (samples) |sample| {
    const result = sample.matmul(weights).eval(allocator);
}
```

### 3. Reuse Memory

Use pooling allocator between batches:

```zig
var pool = PoolAllocator.init(page_allocator);
defer pool.deinit();

for (batches) |batch| {
    const result = try expr.eval(pool.allocator());
    process(result);
    pool.clear();  // Reuse memory
}
```

---

## SIMD Optimization

### Vector Length

Tenzor automatically selects optimal SIMD width:

| Architecture | f32 Vector Length |
|--------------|-------------------|
| AVX-512 | 16 |
| AVX2/AVX | 8 |
| SSE | 4 |
| NEON | 4 |
| Scalar | 1 |

### Alignment

Data is aligned for SIMD access:

```zig
// Automatic alignment to SIMD width
const data = try allocator.alignedAlloc(f32, 32, n);
```

### Vectorized Operations

All element-wise operations use SIMD:

```zig
// Vectorized internally
const result = tensor.exp().mul(other).add(bias);
```

---

## Threading

### When Threading Helps

| Total Elements | Recommendation |
|---------------|----------------|
| < 8K | Sequential |
| 8K - 64K | 2-4 threads |
| 64K - 1M | All cores |
| > 1M | All cores |

### Chunk Size Selection

| Operation Type | Recommended Chunk |
|----------------|-------------------|
| Light (add, mul) | 8192+ |
| Medium (exp, log) | 4096 |
| Heavy (matmul tile) | 1024 |

### Thread Pool Reuse

```zig
// Create once, reuse
var pool: ?*ThreadPool = null;

pub fn getPool() *ThreadPool {
    if (pool == null) {
        pool = ThreadPool.create(allocator, .{}) catch @panic("OOM");
    }
    return pool.?;
}
```

---

## Memory Performance

### Cache Efficiency

1. **Sequential Access**: Row-major iteration
2. **Blocking**: Tile large operations
3. **Prefetching**: Access patterns hint to CPU

### False Sharing

Avoid threads writing adjacent memory:

```zig
// Pad per-thread accumulators
const Padded = struct {
    value: f32,
    _padding: [60]u8 = undefined,  // 64-byte cache line
};
```

### Memory Bandwidth

| Operation | Intensity |
|-----------|-----------|
| Element-wise | Memory bound |
| Matmul (small) | Memory bound |
| Matmul (large) | Compute bound |
| Reduction | Memory bound |

---

## Fusion Benefits

### Without Fusion

```
matmul: Read A, B → Compute → Write C
add:    Read C, bias → Compute → Write D
relu:   Read D → Compute → Write E

Memory traffic: 3 reads + 3 writes
```

### With Fusion

```
fused: Read A, B, bias → Compute all → Write E

Memory traffic: 1 read + 1 write
```

### Fusion Patterns

| Pattern | Speedup |
|---------|---------|
| elementwise_chain | 2-3x |
| matmul_epilogue | 1.5-2x |
| reduce_epilogue | 1.3-1.5x |

---

## Profiling

### Timing Operations

```zig
const timer = std.time.Timer{};

timer.reset();
const result = expr.eval(allocator);
const elapsed = timer.read();

std.debug.print("Elapsed: {} ns\n", .{elapsed});
```

### Memory Usage

```zig
const before = @import("std").process.getrusage().maxrss;
// ... operations ...
const after = @import("std").process.getrusage().maxrss;
std.debug.print("Memory delta: {} KB\n", .{after - before});
```

---

## Benchmarks

### Element-wise Operations

| Size | Scalar | SIMD | Speedup |
|------|--------|------|---------|
| 1K | 0.5 μs | 0.1 μs | 5x |
| 1M | 500 μs | 80 μs | 6x |
| 100M | 50 ms | 8 ms | 6x |

### Matrix Multiplication

| Shape | Naive | Tiled | Tiled+SIMD |
|-------|-------|-------|------------|
| 128×128 | 2 ms | 0.5 ms | 0.1 ms |
| 512×512 | 100 ms | 20 ms | 4 ms |
| 1024×1024 | 800 ms | 150 ms | 30 ms |

### Threading Scaling

| Cores | Speedup (1M elements) |
|-------|----------------------|
| 1 | 1.0x |
| 2 | 1.9x |
| 4 | 3.7x |
| 8 | 7.0x |
| 16 | 12x |

---

## Common Pitfalls

### 1. Evaluating Too Early

```zig
// Bad: Breaks fusion
const tmp = a.add(b).eval(allocator);
const result = tmp.mul(c).eval(allocator);

// Good: Let fusion work
const result = a.add(b).mul(c).eval(allocator);
```

### 2. Allocating in Hot Loops

```zig
// Bad: Allocation per iteration
for (0..1000) |_| {
    const result = expr.eval(allocator);
    allocator.free(result.data);
}

// Good: Reuse buffer
var pool = PoolAllocator.init(allocator);
for (0..1000) |_| {
    const result = expr.eval(pool.allocator());
    pool.clear();
}
```

### 3. Small Parallel Work

```zig
// Bad: Overhead exceeds benefit
pool.parallelFor(0, 100, 10, ctx, work);  // 100 elements

// Good: Sequential for small work
for (0..100) |i| work(ctx, i);
```

---

## Platform-Specific

### x86-64

- Best SIMD support (AVX-512, AVX2)
- Use `-mcpu=native` for best codegen

### ARM64

- NEON SIMD (128-bit)
- Good for Apple Silicon

### Compilation

```bash
# Release build with native optimizations
zig build -Doptimize=ReleaseFast

# Profile guided
zig build -Doptimize=ReleaseFast -Dcpu=native
```
