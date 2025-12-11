#!/usr/bin/env python3
"""Generate test fixtures from HuggingFace arctic-embed-xs model.

Generates:
1. Token IDs for test sentences
2. Expected embedding outputs
3. Intermediate layer outputs for debugging

Usage:
    pip install torch transformers safetensors
    python scripts/generate_fixtures.py
"""

import json
import struct
import os
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


def save_f32_binary(path: Path, data: torch.Tensor):
    """Save tensor as raw f32 binary file."""
    data_np = data.detach().cpu().float().numpy().flatten()
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(data_np)}f", *data_np))


def save_u32_binary(path: Path, data: list[int]):
    """Save list of ints as raw u32 binary file."""
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(data)}I", *data))


def main():
    output_dir = Path("test_fixtures")
    output_dir.mkdir(exist_ok=True)

    print("Loading model and tokenizer...")
    model_name = "Snowflake/snowflake-arctic-embed-xs"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
    model.eval()

    # Test sentences
    test_sentences = [
        "Hello world",
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is transforming how we process information",
    ]

    fixtures = []

    print("\nGenerating fixtures...")
    for i, sentence in enumerate(test_sentences):
        print(f"  Processing: {sentence[:50]}...")

        # Tokenize
        inputs = tokenizer(sentence, return_tensors="pt", padding=False, truncation=True)
        token_ids = inputs["input_ids"][0].tolist()

        # Run model
        with torch.no_grad():
            outputs = model(**inputs)
            # CLS token embedding (first token of last hidden state)
            cls_embedding = outputs.last_hidden_state[0, 0, :]
            # L2 normalize (arctic-embed uses CLS pooling + L2 norm)
            cls_embedding = torch.nn.functional.normalize(cls_embedding, p=2, dim=0)

        # Save fixtures
        prefix = f"test_{i}"

        # Save token IDs
        token_path = output_dir / f"{prefix}_tokens.bin"
        save_u32_binary(token_path, token_ids)

        # Save expected embedding
        embedding_path = output_dir / f"{prefix}_embedding.bin"
        save_f32_binary(embedding_path, cls_embedding)

        fixtures.append({
            "name": f"test_{i}",
            "sentence": sentence,
            "num_tokens": len(token_ids),
            "token_ids": token_ids,
            "embedding_dim": cls_embedding.shape[0],
        })

        # Print first few values for verification
        print(f"    Tokens: {token_ids[:5]}... ({len(token_ids)} total)")
        print(f"    Embedding[0:5]: {cls_embedding[:5].tolist()}")

    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump({
            "model": model_name,
            "hidden_size": model.config.hidden_size,
            "num_layers": model.config.num_hidden_layers,
            "num_heads": model.config.num_attention_heads,
            "intermediate_size": model.config.intermediate_size,
            "vocab_size": model.config.vocab_size,
            "fixtures": fixtures,
        }, f, indent=2)

    print(f"\nFixtures saved to {output_dir}/")
    print(f"  - metadata.json")
    for i in range(len(test_sentences)):
        print(f"  - test_{i}_tokens.bin")
        print(f"  - test_{i}_embedding.bin")

    # Also print tensor names from the model for reference
    print("\nModel state dict keys (for weight mapping):")
    for key in list(model.state_dict().keys())[:20]:
        print(f"  {key}")
    print("  ...")

    # Save model weights as safetensors
    from safetensors.torch import save_file

    weights_path = output_dir / "model.safetensors"
    print(f"\nSaving model weights to {weights_path}...")
    state_dict = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(state_dict, weights_path)
    print(f"  Saved {len(state_dict)} tensors")

    # Print file size
    size_mb = weights_path.stat().st_size / (1024 * 1024)
    print(f"  File size: {size_mb:.1f} MB")

    # Save intermediate outputs for debugging
    print("\nGenerating debug fixtures...")
    for i, sentence in enumerate(test_sentences[:1]):  # Only first test
        inputs = tokenizer(sentence, return_tensors="pt", padding=False, truncation=True)

        # Hook to capture embeddings output
        embeddings_output = None
        def hook_embeddings(module, input, output):
            nonlocal embeddings_output
            embeddings_output = output

        handle = model.embeddings.register_forward_hook(hook_embeddings)

        with torch.no_grad():
            _ = model(**inputs)

        handle.remove()

        # Save embeddings output (after LayerNorm)
        emb_path = output_dir / f"test_{i}_embeddings_output.bin"
        save_f32_binary(emb_path, embeddings_output[0])  # [seq_len, hidden]
        print(f"  Embeddings output shape: {embeddings_output.shape}")
        print(f"  Embeddings output[0,0:5]: {embeddings_output[0,0,:5].tolist()}")

        # Also capture first encoder layer output
        layer0_output = None
        def hook_layer0(module, input, output):
            nonlocal layer0_output
            layer0_output = output[0]  # (hidden_states,) tuple

        handle2 = model.encoder.layer[0].register_forward_hook(hook_layer0)
        with torch.no_grad():
            _ = model(**inputs)
        handle2.remove()

        layer0_path = output_dir / f"test_{i}_layer0_output.bin"
        save_f32_binary(layer0_path, layer0_output[0])
        print(f"  Layer 0 output shape: {layer0_output.shape}")
        print(f"  Layer 0 output[0,0:5]: {layer0_output[0,0,:5].tolist()}")


if __name__ == "__main__":
    main()
