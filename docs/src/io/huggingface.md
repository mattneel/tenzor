# HuggingFace Hub

Download and convert models from the HuggingFace Hub.

## Overview

Tenzor can download models directly from HuggingFace and convert them to the native `.tenzor` format for optimal loading performance.

## Basic Usage

```zig
const HuggingFace = @import("tenzor").io.huggingface.HuggingFace;

var hf = HuggingFace.init(allocator, null);  // null = default cache dir
defer hf.deinit();

// Download and convert to .tenzor
const tenzor_path = try hf.downloadAndConvert(
    "Snowflake/snowflake-arctic-embed-xs",
    null,  // output path (null = auto)
);
defer allocator.free(tenzor_path);

std.debug.print("Model saved to: {s}\n", .{tenzor_path});
```

## API

### HuggingFace.init

```zig
pub fn init(allocator: std.mem.Allocator, cache_dir: ?[]const u8) HuggingFace
```

Initialize with optional custom cache directory.

### downloadAndConvert

```zig
pub fn downloadAndConvert(
    self: *HuggingFace,
    model_id: []const u8,
    output_path: ?[]const u8,
) ![]const u8
```

Download model and convert to `.tenzor` format.

- `model_id`: HuggingFace model ID (e.g., `"Snowflake/snowflake-arctic-embed-xs"`)
- `output_path`: Optional output path. If null, uses cache directory.

Returns path to the created `.tenzor` file.

## CLI Usage

```bash
# Download to default cache
tenzor download Snowflake/snowflake-arctic-embed-xs

# Download to specific path
tenzor download Snowflake/snowflake-arctic-embed-xs -o models/arctic.tenzor
```

## What Gets Downloaded

For a typical model, downloads:

1. `model.safetensors` - Model weights
2. `config.json` - Model configuration
3. `vocab.txt` or `tokenizer.json` - Tokenizer files

## Cache Directory

Default cache location: `~/.cache/huggingface/`

Structure:
```
~/.cache/huggingface/
  Snowflake--snowflake-arctic-embed-xs/
    model.safetensors
    config.json
    vocab.txt
    model.tenzor  (converted)
```

## Network Requirements

- HTTPS access to `huggingface.co`
- API endpoint: `https://huggingface.co/api/models/{model_id}`
- File downloads: `https://huggingface.co/{model_id}/resolve/main/{filename}`

## Progress Reporting

Downloads show progress:

```
Downloading config.json... done (1.2 KB)
Downloading model.safetensors... 45.2 MB / 86.4 MB (52%)
```

## Error Handling

```zig
const path = hf.downloadAndConvert(model_id, null) catch |err| switch (err) {
    error.ModelNotFound => {
        std.debug.print("Model not found: {s}\n", .{model_id});
        return;
    },
    error.NetworkError => {
        std.debug.print("Network error, check connection\n", .{});
        return;
    },
    else => return err,
};
```

## Supported Models

Any model with `model.safetensors` can be downloaded. Currently tested with:

- `Snowflake/snowflake-arctic-embed-xs` (embedding model)
- BERT-style models with WordPiece tokenization
