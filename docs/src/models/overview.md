# Model Support

Tenzor includes reference implementations of popular models, verified against HuggingFace outputs.

## Supported Models

| Model | Parameters | Output | Status |
|-------|------------|--------|--------|
| [Arctic Embed XS](./arctic.md) | 22M | 384-dim embedding | ✅ Complete |

## Architecture

Each model implementation includes:

- **Weight loader** - Loads from SafeTensors format
- **Forward pass** - Pure Zig inference
- **Parity tests** - Verified against HuggingFace (cosine similarity > 0.99)

## Running Models

### 1. Generate Test Fixtures

Models require fixtures generated from HuggingFace (weights + test data):

```bash
# Install Python dependencies
pip install torch transformers safetensors

# Generate fixtures for a model
python scripts/parity/arctic.py
```

Fixtures are saved to `test_fixtures/` (gitignored, ~88MB for arctic).

### 2. Run Tests

```bash
zig build test
```

### 3. Use in Code

```zig
const arctic = @import("tenzor").model.arctic;
const safetensors = @import("tenzor").io.safetensors;

// Load weights
const load_result = try safetensors.load(allocator, "model.safetensors");
var st = load_result.st;
defer st.deinit();
defer allocator.free(load_result.data);

// Initialize model
const config = arctic.arctic_embed_xs_config;
var weights = try arctic.ModelWeights.fromSafeTensors(allocator, st, config);
defer weights.deinit(allocator);

// Run inference
var ctx = try arctic.InferenceContext.init(allocator, config, max_seq_len);
defer ctx.deinit();

const output = try allocator.alloc(f32, config.hidden_size);
defer allocator.free(output);

arctic.forward(output, token_ids, weights, &ctx);
// output now contains L2-normalized 384-dim embedding
```
