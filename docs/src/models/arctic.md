# Arctic Embed XS

[Snowflake Arctic Embed](https://huggingface.co/Snowflake/snowflake-arctic-embed-xs) is a family of text embedding models optimized for retrieval tasks.

## Model Details

| Property | Value |
|----------|-------|
| Parameters | ~22M |
| Hidden size | 384 |
| Layers | 6 |
| Attention heads | 12 |
| Intermediate size | 1536 |
| Vocab size | 30522 |
| Max sequence length | 512 |
| Output dimension | 384 |

## Architecture

Arctic Embed XS uses a BERT-style encoder architecture:

```
Input tokens
    ↓
Word Embeddings + Position Embeddings
    ↓
LayerNorm
    ↓
┌─────────────────────────────────┐
│  Transformer Block (×6)         │
│  ├── Multi-Head Self-Attention  │
│  │   └── Q, K, V projections    │
│  │   └── Scaled dot-product     │
│  │   └── Output projection      │
│  ├── Residual + LayerNorm       │
│  ├── Feed-Forward Network       │
│  │   └── Linear → GELU → Linear │
│  └── Residual + LayerNorm       │
└─────────────────────────────────┘
    ↓
CLS Token Pooling (first token)
    ↓
L2 Normalization
    ↓
384-dim embedding
```

## Usage

### Generate Fixtures

```bash
pip install torch transformers safetensors
python scripts/parity/arctic.py
```

### Load and Run

```zig
const std = @import("std");
const arctic = @import("tenzor").model.arctic;
const safetensors = @import("tenzor").io.safetensors;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Load model
    const load_result = try safetensors.load(allocator, "test_fixtures/model.safetensors");
    var st = load_result.st;
    defer st.deinit();
    defer allocator.free(load_result.data);

    const config = arctic.arctic_embed_xs_config;
    var weights = try arctic.ModelWeights.fromSafeTensors(allocator, st, config);
    defer weights.deinit(allocator);

    // Initialize context (reusable across inferences)
    var ctx = try arctic.InferenceContext.init(allocator, config, 128);
    defer ctx.deinit();

    // Token IDs from your tokenizer
    const tokens = &[_]u32{ 101, 7592, 2088, 102 }; // [CLS] hello world [SEP]

    // Run inference
    const output = try allocator.alloc(f32, config.hidden_size);
    defer allocator.free(output);

    arctic.forward(output, tokens, weights, &ctx);

    // output is now a 384-dim L2-normalized embedding
    std.debug.print("Embedding[0:5]: {d:.4}\n", .{output[0..5]});
}
```

## Implementation Details

### Weight Mapping

HuggingFace weights use `embeddings.word_embeddings.weight` naming. Tenzor handles both:
- Standard: `embeddings.word_embeddings.weight`
- BERT-prefixed: `bert.embeddings.word_embeddings.weight`

LayerNorm weights map `gamma`/`beta` to `weight`/`bias`.

### Buffer Management

`InferenceContext` pre-allocates all intermediate buffers to avoid allocations during inference:
- `hidden` - Current hidden states
- `qkv` - Query/Key/Value projections
- `attn_scores` - Attention weights
- `attn_out` - Attention output
- `ffn_intermediate` - FFN hidden layer
- `encoder_input/output` - Ping-pong buffers for encoder layers

### Numerical Precision

The implementation matches HuggingFace output with cosine similarity > 0.999, verified across multiple test sentences.

## Parity Tests

Located in `src/tests/arctic_integration_test.zig`:

- **Embeddings test** - Verifies embedding layer output
- **Layer 0 test** - Verifies first transformer block
- **Full model test** - End-to-end inference across 3 test sentences
