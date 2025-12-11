# download Command

Download models from HuggingFace Hub and convert to `.tenzor` format.

## Usage

```bash
tenzor download <MODEL_ID> [-o <OUTPUT>]
```

## Arguments

| Argument | Description |
|----------|-------------|
| `<MODEL_ID>` | HuggingFace model ID (e.g., `Snowflake/snowflake-arctic-embed-xs`) |

## Options

| Option | Description |
|--------|-------------|
| `-o, --output <PATH>` | Output .tenzor file path |

## Examples

### Download Arctic Embed

```bash
tenzor download Snowflake/snowflake-arctic-embed-xs
```

### Custom Output Path

```bash
tenzor download Snowflake/snowflake-arctic-embed-xs -o models/arctic.tenzor
```

## What Gets Downloaded

1. `model.safetensors` - Model weights
2. `config.json` - Model configuration
3. `vocab.txt` or `tokenizer.json` - Tokenizer files

## Cache Location

Downloaded files are cached in:
- `~/.cache/huggingface/` (default)
- Or specified cache directory

## Output

```
HuggingFace Model Download
==========================
Model: Snowflake/snowflake-arctic-embed-xs

Downloading config.json...
Downloading model.safetensors...
Converting to .tenzor format...

Download complete!
  Output: models/snowflake-arctic-embed-xs.tenzor
  Size:   86.2 MB
  Time:   12.3 s
```

## Supported Models

Any model with `model.safetensors` can be downloaded. Currently optimized for:
- Arctic-embed models
- BERT-style models

## Network Requirements

- HTTPS access to `huggingface.co`
- Sufficient disk space for model files
