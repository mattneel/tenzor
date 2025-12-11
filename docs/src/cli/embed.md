# embed Command

Generate text embeddings using Arctic-embed-xs or compatible models.

## Usage

```bash
tenzor embed -m <MODEL_PATH> [OPTIONS] <TEXT>...
tenzor embed -m <MODEL_PATH> -i <INPUT_FILE>
```

## Options

| Option | Description |
|--------|-------------|
| `-m, --model <PATH>` | Path to model.safetensors (required) |
| `-i, --input <PATH>` | Read text from file |
| `<TEXT>...` | Text(s) to embed |

## Examples

### Single Text

```bash
tenzor embed -m models/arctic/model.safetensors "Hello world"
```

### Multiple Texts

```bash
tenzor embed -m model.safetensors "First text" "Second text" "Third text"
```

### From File

```bash
tenzor embed -m model.safetensors -i document.txt
```

## Model Requirements

The model directory should contain:
- `model.safetensors` - Model weights
- `vocab.txt` - WordPiece vocabulary

## Output Format

### Single Text
```
Embedding: [0.012345, -0.023456, 0.034567, ...]
```

### Multiple Texts
```json
[
  {
    "text": "First text",
    "tokens": 4,
    "embedding": [0.012345, -0.023456, ...]
  },
  {
    "text": "Second text",
    "tokens": 5,
    "embedding": [0.045678, -0.056789, ...]
  }
]
```

## Long Documents

For documents exceeding the model's context length (512 tokens):
- Text is automatically chunked with overlap
- Each chunk is embedded separately
- Chunk embeddings are mean-pooled and L2-normalized

```bash
tenzor embed -m model.safetensors -i long_document.txt
# Output shows: Total tokens: 2048, Chunks: 5 (size=510, overlap=50)
```

## Supported Models

- Snowflake/snowflake-arctic-embed-xs (384 dimensions)
- Other BERT-style models with WordPiece tokenization

## Performance

- Parallel chunk processing on multi-core CPUs
- SIMD-accelerated attention and matrix operations
