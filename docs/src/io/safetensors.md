# SafeTensors

Load models in the HuggingFace SafeTensors format.

## Overview

SafeTensors is a simple format for storing tensors safely (no arbitrary code execution). Tenzor supports reading SafeTensors files for compatibility with HuggingFace models.

## Loading SafeTensors

```zig
const safetensors = @import("tenzor").io.safetensors;

var loaded = try safetensors.load(allocator, "model.safetensors");
defer loaded.st.deinit();
defer allocator.free(loaded.data);

// Access tensors
const tensor = loaded.st.getTensor("encoder.weight");
if (tensor) |t| {
    std.debug.print("Shape: {any}\n", .{t.shape});
    std.debug.print("DType: {}\n", .{t.dtype});

    // Get data as f32
    const data = t.asF32Slice();
}
```

## SafeTensors Structure

```zig
const SafeTensors = struct {
    header: std.json.Value,
    tensors: std.StringHashMap(TensorInfo),

    pub fn getTensor(self: *SafeTensors, name: []const u8) ?TensorInfo;
    pub fn tensorNames(self: *SafeTensors) []const []const u8;
};

const TensorInfo = struct {
    dtype: DType,
    shape: []const usize,
    data_offset: usize,
    data_size: usize,

    pub fn asF32Slice(self: TensorInfo) []const f32;
};
```

## Supported Data Types

| Type | Description |
|------|-------------|
| F32 | 32-bit float |
| F16 | 16-bit float |
| BF16 | Brain float 16 |
| I32 | 32-bit integer |
| I64 | 64-bit integer |

## Converting to .tenzor

For faster repeated loads, convert to native format:

```zig
const tenzor_format = @import("tenzor").io.tenzor_format;

try tenzor_format.convertFromSafetensors(
    allocator,
    "model.safetensors",
    "model.tenzor",
);
```

Or via CLI:

```bash
tenzor convert model.safetensors
```

## File Format

SafeTensors files have a simple structure:

```
┌─────────────────────────────────────┐
│ Header size (8 bytes, little-endian)│
├─────────────────────────────────────┤
│ JSON Header                         │
│ {                                   │
│   "tensor_name": {                  │
│     "dtype": "F32",                 │
│     "shape": [64, 128],             │
│     "data_offsets": [0, 32768]      │
│   },                                │
│   ...                               │
│ }                                   │
├─────────────────────────────────────┤
│ Tensor Data (contiguous)            │
└─────────────────────────────────────┘
```

## Integration with HuggingFace

Download and load HuggingFace models:

```zig
const HuggingFace = @import("tenzor").io.huggingface.HuggingFace;

var hf = HuggingFace.init(allocator, null);
defer hf.deinit();

// Downloads model.safetensors and converts to .tenzor
const tenzor_path = try hf.downloadAndConvert(
    "Snowflake/snowflake-arctic-embed-xs",
    null,
);
```
