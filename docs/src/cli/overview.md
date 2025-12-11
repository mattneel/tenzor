# CLI Overview

Tenzor provides a command-line interface for training models, generating embeddings, and managing model files.

## Installation

Build the CLI:

```bash
zig build
```

The binary is at `zig-out/bin/tenzor`.

## Commands

| Command | Description |
|---------|-------------|
| `train` | Train LeNet-5 on MNIST |
| `embed` | Generate text embeddings |
| `download` | Download model from HuggingFace |
| `convert` | Convert safetensors to .tenzor |
| `info` | Show .tenzor file information |
| `help` | Show help message |

## Quick Examples

```bash
# Train with TUI dashboard
tenzor train -e 10 -b 64 --lr 0.01

# Generate embeddings
tenzor embed -m model.safetensors "Hello world"

# Download from HuggingFace
tenzor download Snowflake/snowflake-arctic-embed-xs

# Convert model format
tenzor convert model.safetensors -o model.tenzor

# Show file info
tenzor info model.tenzor
```

## Global Options

| Option | Description |
|--------|-------------|
| `-h, --help` | Show help message |

## Getting Help

```bash
tenzor help
tenzor --help
tenzor train --help
```
