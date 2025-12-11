# Activation Functions

Activation functions introduce non-linearity in neural networks. Tenzor provides optimized implementations of common activations.

## ReLU Family

### relu() - Rectified Linear Unit

The most common activation function.

```zig
const y = x.relu();  // y = max(0, x)
```

**Properties:**
- Fast to compute
- Sparse activation (many zeros)
- Can suffer from "dying ReLU" problem

```
       y
       │    /
       │   /
       │  /
───────┼──────── x
       │
```

### leaky_relu() - Leaky ReLU

Allows small gradient for negative inputs.

```zig
const y = x.leaky_relu(0.01);  // y = max(0.01*x, x)
```

**Properties:**
- Prevents dying ReLU
- Small computational overhead

```
       y
       │    /
       │   /
      /│  /
─────/─┼──────── x
    /  │
```

## Sigmoid Family

### sigmoid() - Logistic Sigmoid

Squashes input to (0, 1).

```zig
const y = x.sigmoid();  // y = 1 / (1 + e^(-x))
```

**Properties:**
- Output in (0, 1), useful for probabilities
- Suffers from vanishing gradients
- Computationally more expensive than ReLU

```
       y
    1 ─┼────────────
       │      ╭─────
       │    ╭─╯
   0.5─┼───╱
       │ ╭─╯
    0 ─┼─╯────────── x
```

### silu() - Sigmoid Linear Unit (Swish)

Self-gated activation.

```zig
const y = x.silu();  // y = x * sigmoid(x)
```

**Properties:**
- Smooth, non-monotonic
- Often outperforms ReLU in deep networks
- More computationally expensive

```
       y
       │      /
       │    ╭╯
       │  ╭─╯
───────┼─╯─────── x
      ╯│
```

## Hyperbolic Tangent

### tanh() - Hyperbolic Tangent

Squashes input to (-1, 1).

```zig
const y = x.tanh();  // y = (e^x - e^(-x)) / (e^x + e^(-x))
```

**Properties:**
- Zero-centered output
- Stronger gradients than sigmoid
- Still suffers from vanishing gradients

```
       y
    1 ─┼────────────
       │      ╭─────
       │    ╭─╯
    0 ─┼───╱──────── x
       │ ╭─╯
   -1 ─┼─╯──────────
```

## Modern Activations

### gelu() - Gaussian Error Linear Unit

Used in transformers and modern architectures.

```zig
const y = x.gelu();  // y ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
```

**Properties:**
- Smooth approximation of ReLU
- Standard in BERT, GPT, etc.
- Slightly more expensive than ReLU

```
       y
       │      /
       │    ╭╯
       │  ╭─╯
───────┼─╯─────── x
     ──╯│
```

### softplus() - Softplus

Smooth approximation of ReLU.

```zig
const y = x.softplus();  // y = log(1 + e^x)
```

**Properties:**
- Smooth, always positive
- Derivative is sigmoid
- Useful when strictly positive output needed

```
       y
       │    /
       │   /
       │  ╯
───────┼─╯──────── x
       │╯
```

## Comparison Table

| Activation | Formula | Range | Speed |
|------------|---------|-------|-------|
| ReLU | max(0, x) | [0, ∞) | ★★★★★ |
| Leaky ReLU | max(αx, x) | (-∞, ∞) | ★★★★★ |
| Sigmoid | 1/(1+e^(-x)) | (0, 1) | ★★★☆☆ |
| Tanh | (e^x-e^(-x))/(e^x+e^(-x)) | (-1, 1) | ★★★☆☆ |
| GELU | x·Φ(x) | (-0.17, ∞) | ★★☆☆☆ |
| SiLU | x·σ(x) | (-0.28, ∞) | ★★★☆☆ |
| Softplus | log(1+e^x) | (0, ∞) | ★★★☆☆ |

## Usage Patterns

### Dense Layer with Activation

```zig
const linear = input.matmul(weights).add(bias);
const activated = linear.relu();  // or .gelu(), .silu(), etc.
```

### Transformer Block

```zig
// GELU is standard in transformers
const hidden = attn_output.matmul(ff_weights1);
const activated = hidden.gelu();
const output = activated.matmul(ff_weights2);
```

### Output Layer

```zig
// Sigmoid for binary classification
const logits = hidden.matmul(output_weights).add(output_bias);
const probs = logits.sigmoid();

// For multi-class, apply softmax externally
```

## Fusion with Matmul

Activations after matmul are automatically fused:

```zig
// These operations are fused into a single kernel
const output = input.matmul(weights).add(bias).relu();
```

The fusion engine detects the pattern and generates:
- Matmul with epilogue fusion
- Single memory write for the result

## Next Steps

- [Matrix Operations](./matrix.md) - Matmul that activations fuse with
- [Fusion Engine](../fusion/overview.md) - How activations are fused
