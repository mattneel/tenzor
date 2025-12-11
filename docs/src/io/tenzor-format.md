# .tenzor Format

The `.tenzor` format is a mmap-friendly binary format designed for instant model loading.

## Features

- **Instant load**: mmap-based, no parsing overhead
- **Zero-copy**: Tensor data accessed directly from file
- **Page-aligned**: Data aligned for efficient memory mapping
- **Metadata**: JSON metadata for model info, training state

## File Layout

```
┌─────────────────────────────────────┐ 0x0000
│ Header (64 bytes)                   │
│   magic: "TENZOR\x00\x00" (8 bytes) │
│   version: u32                      │
│   flags: u32                        │
│   tensor_count: u32                 │
│   index_offset: u64                 │
│   data_offset: u64                  │
│   metadata_size: u32                │
│   reserved: [20]u8                  │
├─────────────────────────────────────┤ 0x0040
│ Metadata (JSON, variable size)      │
│   { "model": "lenet",               │
│     "epoch": 5,                     │
│     ... }                           │
├─────────────────────────────────────┤ (page aligned)
│ Tensor Index (48 bytes × N)         │
│   name_hash: u64                    │
│   dtype: u8                         │
│   ndim: u8                          │
│   shape: [6]u32                     │
│   data_offset: u64                  │
│   data_size: u64                    │
├─────────────────────────────────────┤ (page aligned, 4KB)
│ Tensor Data                         │
│   (contiguous, aligned)             │
└─────────────────────────────────────┘
```

## Reading Files

### Open and Access

```zig
const tenzor_format = @import("tenzor").io.tenzor_format;

var file = try tenzor_format.TenzorFile.open(allocator, "model.tenzor");
defer file.close();

// Access metadata
std.debug.print("Tensors: {}\n", .{file.header.tensor_count});

// Get tensor by name (hashed lookup)
if (file.getTensor("layer1.weight")) |data| {
    // data is []const f32 directly from mmap
}

// Get tensor by hash
const hash = tenzor_format.hashName("layer1.weight");
if (file.getTensorByHash(hash)) |data| {
    // ...
}
```

### Iterate Tensors

```zig
for (file.index) |entry| {
    std.debug.print("Tensor: hash=0x{x}, shape={any}\n", .{
        entry.name_hash,
        entry.shape[0..entry.ndim],
    });
}
```

## Writing Files

```zig
var writer = try tenzor_format.TenzorWriter.create(allocator, "output.tenzor");
defer writer.deinit();

// Add tensors
try writer.addTensor("layer1.weight", weights_data, &.{ 64, 128 });
try writer.addTensor("layer1.bias", bias_data, &.{128});

// Set metadata
try writer.setMetadata(.{
    .model = "mymodel",
    .version = "1.0",
});

// Finalize and write
try writer.finish();
```

## Converting from SafeTensors

```zig
try tenzor_format.convertFromSafetensors(
    allocator,
    "model.safetensors",
    "model.tenzor",
);
```

Or via CLI:

```bash
tenzor convert model.safetensors -o model.tenzor
```

## Data Types

| DType | Value | Size |
|-------|-------|------|
| f32 | 0 | 4 bytes |
| f16 | 1 | 2 bytes |
| bf16 | 2 | 2 bytes |
| i32 | 3 | 4 bytes |
| i64 | 4 | 8 bytes |

## Performance Comparison

| Operation | SafeTensors | .tenzor |
|-----------|-------------|---------|
| Open file | ~50ms | <1ms |
| First tensor access | ~10ms | <0.1ms |
| Memory usage | Full model in RAM | mmap (lazy) |
| Parse overhead | JSON header | None |

## Use Cases

### Model Inference

```zig
// Load once at startup
var model_file = try TenzorFile.open(allocator, "model.tenzor");

// Use weights directly (zero-copy)
const weights = model_file.getTensor("encoder.weight").?;
```

### Training Checkpoints

Checkpoints include model weights + optimizer state:

```zig
// Writer adds optimizer state tensors
try writer.addTensor("optim/layer1.weight/momentum", momentum_data, shape);
```

### Large Models

For models larger than RAM, mmap provides virtual memory paging:

```zig
// Only accessed pages are loaded
var file = try TenzorFile.open(allocator, "large_model.tenzor");
// 86GB model works on 16GB machine
```
