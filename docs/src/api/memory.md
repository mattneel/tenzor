# Memory API Reference

## Overview

```zig
const memory = @import("tenzor").memory;
```

---

## PoolAllocator

High-level allocator with buffer pooling.

### Type Definition

```zig
pub const PoolAllocator = struct {
    pool: BufferPool,

    pub fn init(backing: std.mem.Allocator) Self;
    pub fn deinit(self: *Self) void;
    pub fn allocator(self: *Self) std.mem.Allocator;
    pub fn alloc(self: *Self, comptime T: type, n: usize) ![]T;
    pub fn free(self: *Self, buf: anytype) void;
    pub fn clear(self: *Self) void;
};
```

### `init(backing_allocator) PoolAllocator`

Create a pooling allocator.

```zig
var pool = PoolAllocator.init(std.heap.page_allocator);
defer pool.deinit();
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `backing` | `Allocator` | Underlying allocator for actual allocations |

### `deinit() void`

Free all memory and destroy the pool.

```zig
pool.deinit();
```

### `allocator() std.mem.Allocator`

Get standard allocator interface for use with APIs expecting `Allocator`.

```zig
const alloc = pool.allocator();
const result = try someFunction(alloc);
```

### `alloc(T, n) ![]T`

Allocate typed slice. May return pooled buffer.

```zig
const data = try pool.alloc(f32, 1024);
defer pool.free(data);
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `T` | `type` | Element type |
| `n` | `usize` | Number of elements |

**Returns:** `[]T` slice of `n` elements.

### `free(buf) void`

Return buffer to pool for reuse.

```zig
pool.free(data);
```

### `clear() void`

Return all active allocations to pool without freeing memory.

```zig
// Batch operation complete, reuse memory
pool.clear();
```

---

## BufferPool

Low-level buffer pooling with size buckets.

### Type Definition

```zig
pub const BufferPool = struct {
    allocator: std.mem.Allocator,
    buckets: [NUM_BUCKETS]Bucket,

    pub fn init(allocator: std.mem.Allocator) Self;
    pub fn deinit(self: *Self) void;
    pub fn get(self: *Self, size: usize) ?[]u8;
    pub fn getAligned(self: *Self, size: usize, alignment: Alignment) ?[]u8;
    pub fn getOrAlloc(self: *Self, size: usize) ![]u8;
    pub fn put(self: *Self, buf: []u8) void;
    pub fn putAligned(self: *Self, buf: []u8, alignment: Alignment) void;
    pub fn clear(self: *Self) void;
};
```

### `init(allocator) BufferPool`

Create buffer pool.

```zig
var pool = BufferPool.init(page_allocator);
defer pool.deinit();
```

### `deinit() void`

Free all pooled memory.

### `get(size) ?[]u8`

Get buffer of at least `size` bytes from pool.

```zig
if (pool.get(4096)) |buf| {
    // Use buffer
    pool.put(buf);
} else {
    // Pool empty, allocate fresh
}
```

**Returns:** Buffer slice or `null` if pool empty.

### `getAligned(size, alignment) ?[]u8`

Get buffer with specific alignment.

```zig
const buf = pool.getAligned(4096, .@"64");
```

### `getOrAlloc(size) ![]u8`

Get from pool or allocate new buffer.

```zig
const buf = try pool.getOrAlloc(4096);
defer pool.put(buf);
```

### `put(buf) void`

Return buffer to pool.

```zig
pool.put(buf);
```

### `putAligned(buf, alignment) void`

Return buffer with alignment tracking.

```zig
pool.putAligned(buf, .@"32");
```

### `clear() void`

Return all buffers to pool.

---

## Size Buckets

BufferPool uses power-of-2 size buckets:

| Bucket | Size Range |
|--------|------------|
| 0 | 64 - 127 bytes |
| 1 | 128 - 255 bytes |
| 2 | 256 - 511 bytes |
| 3 | 512 - 1023 bytes |
| ... | ... |
| 20 | 64 MB - 128 MB |

When requesting a buffer:
1. Round size up to next power of 2
2. Check corresponding bucket
3. Return pooled buffer or allocate

---

## Arena Allocator

For batch allocations freed together.

### Usage Pattern

```zig
var arena = std.heap.ArenaAllocator.init(page_allocator);
defer arena.deinit();

const alloc = arena.allocator();

// Many allocations
const a = try alloc.alloc(f32, 1000);
const b = try alloc.alloc(f32, 2000);
const c = try alloc.alloc(f32, 3000);

// All freed at once via arena.deinit()
```

### With Tenzor

```zig
var arena = std.heap.ArenaAllocator.init(page_allocator);
defer arena.deinit();

// Evaluate expression tree
const result = try expression.eval(arena.allocator());
// All intermediate allocations freed with arena
```

---

## Memory Layout

### Tensor Storage

Tensors store data in row-major (C-contiguous) order:

```
Shape: [2, 3, 4]
Index: [i, j, k]
Offset: i * 12 + j * 4 + k
```

### Strides

```zig
fn computeStrides(comptime shape: anytype) [shape.len]usize {
    var strides: [shape.len]usize = undefined;
    var stride: usize = 1;

    comptime var i = shape.len;
    inline while (i > 0) {
        i -= 1;
        strides[i] = stride;
        stride *= shape[i];
    }

    return strides;
}
```

### Alignment

Default alignment follows element type:

| Type | Alignment |
|------|-----------|
| f16 | 2 bytes |
| f32 | 4 bytes |
| f64 | 8 bytes |
| SIMD | 32/64 bytes |

---

## Best Practices

### 1. Use PoolAllocator for Tensors

```zig
var pool = PoolAllocator.init(page_allocator);
defer pool.deinit();

for (batches) |batch| {
    const result = try expr.eval(pool.allocator());
    process(result);
    pool.clear();  // Reuse memory
}
```

### 2. Arena for Expression Trees

```zig
var arena = ArenaAllocator.init(page_allocator);
defer arena.deinit();

const result = try complex_expression.eval(arena.allocator());
// All intermediates freed together
```

### 3. Explicit Cleanup

```zig
const result = try expr.eval(allocator);
defer result.deinit();  // Always clean up
```

### 4. Batch Operations

```zig
pool.clear();  // Reset between batches
// Not deinit - keep underlying memory
```
