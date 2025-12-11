# Parity Tests

This directory contains scripts that generate test fixtures from reference implementations (HuggingFace, PyTorch, etc.) to verify tenzor's model inference matches expected outputs.

## Prerequisites

```bash
pip install torch transformers safetensors
```

## Generating Fixtures

Each script generates fixtures for a specific model. Fixtures are saved to `test_fixtures/` (gitignored).

### Arctic Embed XS

```bash
python scripts/parity/arctic.py
```

Generates:
- `test_fixtures/model.safetensors` - Model weights (~88MB)
- `test_fixtures/test_N_tokens.bin` - Tokenized input (u32 array)
- `test_fixtures/test_N_embedding.bin` - Expected output embedding (f32 array)
- `test_fixtures/test_N_embeddings_output.bin` - Intermediate: after embedding layer
- `test_fixtures/test_N_layer0_output.bin` - Intermediate: after first transformer block
- `test_fixtures/metadata.json` - Model config and test metadata

## Running Parity Tests

After generating fixtures:

```bash
zig build test
```

Tests in `src/tests/` will load fixtures and compare against reference outputs using cosine similarity (threshold: 0.99).

## Adding New Models

1. Create `scripts/parity/<model_name>.py`
2. Generate tokenized inputs and expected outputs
3. Save model weights as safetensors
4. Create corresponding test in `src/tests/<model_name>_integration_test.zig`
5. Document the model in `docs/src/models/`
