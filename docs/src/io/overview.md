# I/O Formats Overview

Tenzor supports multiple file formats for model weights and datasets.

## Supported Formats

| Format | Extension | Read | Write | Use Case |
|--------|-----------|------|-------|----------|
| .tenzor | `.tenzor` | Yes | Yes | Native format, mmap-based |
| SafeTensors | `.safetensors` | Yes | No | HuggingFace models |
| MNIST | IDX | Yes | No | MNIST dataset |

## Format Comparison

### .tenzor (Native)

- **Fastest loading**: mmap-based, <1ms load time
- **Zero-copy**: Tensor data used directly from file
- **Training checkpoints**: Includes optimizer state and metadata

### SafeTensors

- **Compatibility**: Load HuggingFace models directly
- **Conversion**: Convert to .tenzor for faster repeated loads

### MNIST

- **Standard format**: IDX binary format
- **Dataset loading**: Images and labels

## Module Structure

```zig
const tenzor = @import("tenzor");

// .tenzor format
const TenzorFile = tenzor.io.tenzor_format.TenzorFile;
const TenzorWriter = tenzor.io.tenzor_format.TenzorWriter;

// SafeTensors
const safetensors = tenzor.io.safetensors;

// HuggingFace Hub
const HuggingFace = tenzor.io.huggingface.HuggingFace;

// MNIST dataset
const MNISTDataset = tenzor.io.mnist.MNISTDataset;
```

## Quick Examples

### Load .tenzor

```zig
var file = try TenzorFile.open(allocator, "model.tenzor");
defer file.close();

const weights = file.getTensor("layer1.weight");
```

### Load SafeTensors

```zig
var loaded = try safetensors.load(allocator, "model.safetensors");
defer loaded.st.deinit();
defer allocator.free(loaded.data);

const tensor = loaded.st.getTensor("layer1.weight");
```

### Convert Formats

```zig
try tenzor_format.convertFromSafetensors(
    allocator,
    "model.safetensors",
    "model.tenzor",
);
```

### Download from HuggingFace

```zig
var hf = HuggingFace.init(allocator, null);
defer hf.deinit();

const path = try hf.downloadAndConvert(
    "Snowflake/snowflake-arctic-embed-xs",
    null,
);
```
