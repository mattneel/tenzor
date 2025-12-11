# Matrix Multiplication

Matrix multiplication is the computational core of neural networks.

## Basic Usage

```zig
const A = tz.Tensor(f32, .{ M, K });
const B = tz.Tensor(f32, .{ K, N });

var a = A{};
var b = B{};

const C = a.matmul(b);  // Result shape: { M, N }
```

## Shape Rules

For 2D matrices:

```
[M, K] @ [K, N] -> [M, N]
```

The inner dimensions (K) must match:

```zig
const A = tz.Tensor(f32, .{ 3, 4 });  // 3x4
const B = tz.Tensor(f32, .{ 4, 5 });  // 4x5

const C = a.matmul(b);  // 3x5 ✓

const D = tz.Tensor(f32, .{ 6, 5 });  // 6x5
// const E = a.matmul(d);  // Error: 4 != 6 ✗
```

## Batched Matrix Multiplication

Higher-dimensional tensors are batched:

```zig
// Batch of matrices
const A = tz.Tensor(f32, .{ batch, M, K });
const B = tz.Tensor(f32, .{ batch, K, N });
const C = a.matmul(b);  // Shape: { batch, M, N }

// Multi-headed attention style
const A = tz.Tensor(f32, .{ batch, heads, seq, dim });
const B = tz.Tensor(f32, .{ batch, heads, dim, seq });
const C = a.matmul(b);  // Shape: { batch, heads, seq, seq }
```

## Implementation Details

### Tiled Algorithm

The matmul kernel uses cache-blocking:

```
┌─────────────────┐     ┌─────────────────┐
│ A               │     │ B               │
│ ┌───┐           │     │     ┌───┐       │
│ │Tile│          │  @  │     │Tile│      │
│ └───┘           │     │     └───┘       │
└─────────────────┘     └─────────────────┘
                    ↓
         ┌─────────────────┐
         │ C               │
         │     ┌───┐       │
         │     │Tile│      │
         │     └───┘       │
         └─────────────────┘
```

Typical tile sizes:
- L1 cache: 32x32 or 64x64 for f32
- SIMD width: 8 (AVX) or 16 (AVX-512)

### SIMD Vectorization

Inner loop is vectorized:

```zig
// Conceptual implementation
for (0..M) |i| {
    for (0..N / vec_len) |j_vec| {
        var acc: @Vector(vec_len, f32) = @splat(0.0);
        for (0..K) |k| {
            const a_val: @Vector(vec_len, f32) = @splat(a[i * K + k]);
            const b_vec = b[k * N + j_vec * vec_len ..][0..vec_len].*;
            acc += a_val * b_vec;
        }
        // Store acc
    }
}
```

## Epilogue Fusion

Operations following matmul are fused:

```zig
// All fused into single kernel:
const output = input
    .matmul(weights)    // Matrix multiply
    .add(bias)          // Add bias
    .relu();            // Activation
```

The fusion engine detects:
- Matmul followed by binary add (bias)
- Binary add followed by activation

And generates a fused kernel that:
1. Computes matmul tiles
2. Adds bias to each row
3. Applies activation
4. Writes final result

### Supported Epilogues

| Pattern | Fused? |
|---------|--------|
| matmul + add | ✓ |
| matmul + add + relu | ✓ |
| matmul + add + gelu | ✓ |
| matmul + add + sigmoid | ✓ |
| matmul + add + tanh | ✓ |
| matmul + mul | ✓ |

## Common Patterns

### Dense Layer

```zig
fn dense(input: anytype, weights: anytype, bias: anytype) @TypeOf(input.matmul(weights).add(bias)) {
    return input.matmul(weights).add(bias);
}

const hidden = dense(input, w1, b1).relu();
const output = dense(hidden, w2, b2);
```

### Multi-Head Attention

```zig
// Q, K, V projections
const Q = input.matmul(w_q);
const K = input.matmul(w_k);
const V = input.matmul(w_v);

// Attention scores (assumes K is transposed)
const scores = Q.matmul(K_t).div(scale);

// Output
const attended = scores.matmul(V);
const output = attended.matmul(w_o);
```

### MLP Block

```zig
const hidden = input.matmul(w1).add(b1).gelu();
const output = hidden.matmul(w2).add(b2);
```

## Performance Tips

### Alignment

Keep dimensions aligned to SIMD width:

```zig
// Good: N divisible by 8 (AVX f32 width)
const A = tz.Tensor(f32, .{ 64, 128 });
const B = tz.Tensor(f32, .{ 128, 256 });

// Less optimal: odd dimensions
const C = tz.Tensor(f32, .{ 63, 127 });
```

### Memory Layout

Row-major layout is optimal for tenzor's matmul:

```
A[M, K] @ B[K, N] -> C[M, N]
```

Both A and B are accessed in cache-friendly patterns.

## Next Steps

- [Fusion Engine](../fusion/overview.md) - How epilogues are fused
- [SIMD Optimization](../backend/simd.md) - Vectorization details
