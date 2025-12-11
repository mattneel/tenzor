# Buffer Pooling

The buffer pool caches recently freed allocations for fast reuse.

## How It Works

```
Free buffer → Pool → Reuse

Instead of:
Free buffer → OS → Allocate new → Format
```

## Pool Structure

```zig
pub const BufferPool = struct {
    buckets: [NUM_BUCKETS]std.ArrayListUnmanaged(PooledBuffer),
    backing: std.mem.Allocator,
    hits: usize,
    misses: usize,
    total_pooled_bytes: usize,
};
```

### Size Buckets

Buffers are grouped by power-of-2 size:

```
Bucket 0: 64 bytes    (2^6)
Bucket 1: 128 bytes   (2^7)
Bucket 2: 256 bytes   (2^8)
...
Bucket 15: 2 MB       (2^21)
```

## Operations

### get() - Retrieve Buffer

```zig
pub fn get(self: *Self, size: usize) ?[]u8 {
    if (size < MIN_SIZE or size > MAX_SIZE) {
        return null;  // Size not poolable
    }

    const bucket_idx = getBucketIndex(size);
    var bucket = &self.buckets[bucket_idx];

    if (bucket.popOrNull()) |item| {
        self.hits += 1;
        self.total_pooled_bytes -= item.size;
        return item.ptr[0..item.size];
    }

    self.misses += 1;
    return null;
}
```

### put() - Return Buffer

```zig
pub fn putAligned(self: *Self, buf: []u8, alignment: std.mem.Alignment) void {
    if (buf.len < MIN_SIZE or buf.len > MAX_SIZE) {
        // Too small or large, just free it
        self.backing.rawFree(buf, alignment, @returnAddress());
        return;
    }

    const bucket_idx = getBucketIndex(buf.len);
    var bucket = &self.buckets[bucket_idx];

    // Check bucket capacity
    if (bucket.items.len >= MAX_BUFFERS_PER_BUCKET) {
        self.backing.rawFree(buf, alignment, @returnAddress());
        return;
    }

    // Add to pool
    bucket.append(self.backing, .{
        .ptr = buf.ptr,
        .size = buf.len,
        .log2_align = @intFromEnum(alignment),
    }) catch {
        self.backing.rawFree(buf, alignment, @returnAddress());
    };

    self.total_pooled_bytes += buf.len;
}
```

## Bucket Sizing

```zig
fn getBucketIndex(size: usize) usize {
    // Round up to power of 2, compute log2
    const rounded = std.math.ceilPowerOfTwo(usize, size) catch size;
    const log2 = std.math.log2_int(usize, rounded);

    // Bucket 0 is 2^MIN_LOG2_SIZE
    const idx = log2 - MIN_LOG2_SIZE;
    return @min(idx, NUM_BUCKETS - 1);
}
```

## Statistics

```zig
pub const Stats = struct {
    hits: usize,
    misses: usize,
    hit_rate: f64,
    total_pooled_bytes: usize,
};

pub fn getStats(self: *const Self) Stats {
    const total = self.hits + self.misses;
    return .{
        .hits = self.hits,
        .misses = self.misses,
        .hit_rate = if (total > 0)
            @as(f64, @floatFromInt(self.hits)) / @as(f64, @floatFromInt(total))
        else
            0.0,
        .total_pooled_bytes = self.total_pooled_bytes,
    };
}
```

## Configuration

```zig
const MIN_SIZE = 64;           // Don't pool tiny buffers
const MAX_SIZE = 2 * 1024 * 1024;  // 2 MB max
const NUM_BUCKETS = 16;        // 64B to 2MB
const MAX_BUFFERS_PER_BUCKET = 8;  // Limit memory usage
```

## Memory Management

### clear() - Free All Pooled Buffers

```zig
pub fn clear(self: *Self) void {
    for (&self.buckets) |*bucket| {
        for (bucket.items) |item| {
            const alignment: std.mem.Alignment = @enumFromInt(item.log2_align);
            self.backing.rawFree(item.ptr[0..item.size], alignment, @returnAddress());
        }
        bucket.clearRetainingCapacity();
    }
    self.total_pooled_bytes = 0;
}
```

### deinit() - Cleanup

```zig
pub fn deinit(self: *Self) void {
    for (&self.buckets) |*bucket| {
        for (bucket.items) |item| {
            const alignment: std.mem.Alignment = @enumFromInt(item.log2_align);
            self.backing.rawFree(item.ptr[0..item.size], alignment, @returnAddress());
        }
        bucket.deinit(self.backing);
    }
}
```

## Best Practices

### Reuse Pattern

```zig
// Good: same sizes reuse pooled buffers
for (0..1000) |_| {
    const buf = try alloc.alloc(f32, 256);
    // ... use buf ...
    alloc.free(buf);  // Returns to pool
}
// Only 1 actual allocation, 999 pool hits
```

### Size Consistency

```zig
// Good: consistent sizes
const BATCH_SIZE = 32;
const FEATURES = 128;
const buf = try alloc.alloc(f32, BATCH_SIZE * FEATURES);

// Avoid: varying sizes
for (sizes) |size| {
    const buf = try alloc.alloc(f32, size);  // Different bucket each time
}
```

## Next Steps

- [Compute Arenas](./arenas.md) - Arena allocation
- [Allocator Design](./allocator.md) - TensorAllocator overview
