#!/usr/bin/env python3
"""Generate test fixtures for WordPiece tokenizer parity testing.

Generates token IDs for test sentences using HuggingFace tokenizer.

Usage:
    pip install transformers
    python scripts/parity/tokenizer.py
"""

import json
from pathlib import Path

from transformers import AutoTokenizer


def main():
    output_dir = Path("test_fixtures/tokenizer")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the arctic tokenizer (BERT-based)
    tokenizer = AutoTokenizer.from_pretrained("Snowflake/snowflake-arctic-embed-xs")

    # Test cases
    test_cases = [
        "Hello world",
        "hello world",
        "HELLO WORLD",
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "What is machine learning?",
        "testing",  # Should split to "test" + "##ing"
        "embeddings",  # Should split
        "I love programming in Zig!",
        "",  # Empty
        "a",  # Single char
        "123",  # Numbers
        "hello-world",  # Hyphen
        "don't",  # Apostrophe
        "   spaces   ",  # Multiple spaces
    ]

    results = []
    for text in test_cases:
        encoded = tokenizer.encode(text, add_special_tokens=True)
        tokens = tokenizer.convert_ids_to_tokens(encoded)

        results.append({
            "text": text,
            "token_ids": encoded,
            "tokens": tokens,
        })

        print(f"Text: {repr(text)}")
        print(f"  Tokens: {tokens}")
        print(f"  IDs: {encoded}")
        print()

    # Save results
    with open(output_dir / "test_cases.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} test cases to {output_dir}/test_cases.json")


if __name__ == "__main__":
    main()
