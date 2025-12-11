# Allocator Design

Tenzor provides specialized allocators for tensor computations.

## TensorAllocator

A pooling allocator optimized for tensor workloads.

```zig
const memory = @import("tenzor").memory;

var tensor_alloc = memory.TensorAllocator.init(std.heap.page_allocator);
defer tensor_alloc.deinit();

// Allocate tensor data
const data = try tensor_alloc.alloc(f32, 1024);
defer tensor_alloc.free(data);
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TensorAllocator                           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                   Buffer Pool                         │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐    │   │
│  │  │ 64-127  │ │128-255  │ │256-511  │ │512-1023 │ ...│   │
│  │  │  bytes  │ │ bytes   │ │ bytes   │ │ bytes   │    │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘    │   │
│  └──────────────────────────────────────────────────────┘   │
│                              │                               │
│                              ▼                               │
│                    Backing Allocator                         │
└─────────────────────────────────────────────────────────────┘
```

## Features

### Buffer Pooling

Recently freed buffers are cached for reuse:

```zig
// First allocation
const a = try alloc.alloc(f32, 256);  // Allocates from backing
alloc.free(a);                         // Returns to pool

// Second allocation
const b = try alloc.alloc(f32, 256);  // Reuses from pool (fast!)
```

### Size Bucketing

Buffers are organized by size class:

| Bucket | Size Range |
|--------|-----------|
| 0 | 64 - 127 bytes |
| 1 | 128 - 255 bytes |
| 2 | 256 - 511 bytes |
| 3 | 512 - 1023 bytes |
| ... | Power of 2 ranges |

### Statistics

Track allocation patterns:

```zig
const stats = alloc.getStats();

std.debug.print("Allocations: {}\n", .{stats.allocations});
std.debug.print("Pool hits: {}\n", .{stats.pool_hits});
std.debug.print("Pool misses: {}\n", .{stats.pool_misses});
std.debug.print("Hit rate: {d:.1}%\n", .{stats.pool_hit_rate * 100});
```

## Implementation

```zig
pub const TensorAllocator = struct {
    pool: BufferPool,
    backing: std.mem.Allocator,
    allocations: usize,
    deallocations: usize,
    bytes_allocated: usize,
    bytes_freed: usize,
    pool_hits: usize,
    pool_misses: usize,

    pub fn init(backing: std.mem.Allocator) Self {
        return .{
            .pool = BufferPool.init(backing),
            .backing = backing,
            // ... initialize stats
        };
    }

    pub fn alloc(self: *Self, comptime T: type, count: usize) ![]T {
        const byte_count = count * @sizeOf(T);

        // Try pool first
        if (self.pool.get(byte_count)) |buf| {
            self.pool_hits += 1;
            return @alignCast(std.mem.bytesAsSlice(T, buf));
        }

        // Fall back to backing allocator
        self.pool_misses += 1;
        return try self.backing.alloc(T, count);
    }

    pub fn free(self: *Self, buf: anytype) void {
        const bytes = std.mem.sliceAsBytes(buf);
        self.pool.put(bytes);  // Return to pool
    }
};
```

## Allocator Interface

TensorAllocator implements `std.mem.Allocator`:

```zig
// Get standard allocator interface
const allocator = tensor_alloc.allocator();

// Use with standard library
var list = std.ArrayList(f32).init(allocator);
```

## Usage Patterns

### Computation Context

```zig
var tensor_alloc = memory.TensorAllocator.init(gpa);
defer tensor_alloc.deinit();

// All tensor operations use this allocator
const result = try tz.eval(expr, tensor_alloc.allocator());
```

### Repeated Computations

```zig
var alloc = memory.TensorAllocator.init(backing);
defer alloc.deinit();

for (0..1000) |_| {
    const result = try tz.eval(expr, alloc.allocator());
    process(result);
    alloc.free(result);  // Returns to pool for next iteration
}
```

## Configuration

```zig
pub const Config = struct {
    max_pool_size: usize = 64 * 1024 * 1024,  // 64 MB
    max_buffers_per_bucket: usize = 8,
};

var alloc = memory.TensorAllocator.initWithConfig(backing, .{
    .max_pool_size = 128 * 1024 * 1024,
});
```

## Next Steps

- [Buffer Pooling](./pooling.md) - Pool implementation details
- [Compute Arenas](./arenas.md) - Arena allocation pattern
