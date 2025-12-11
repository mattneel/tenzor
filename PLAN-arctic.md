# Arctic Embed XS Implementation Plan

## Current State Analysis

### What Tenzor Has
- Core tensor types with compile-time shapes
- Expression graphs with lazy evaluation
- Unary ops: neg, abs, exp, log, sqrt, sin, cos, tanh, sigmoid, relu, **gelu**, silu, softplus
- Binary ops: add, sub, mul, div, pow, max, min (with broadcasting)
- Matrix ops: 2D matmul (tiled + SIMD), basic transpose
- Reductions: sum, prod, max, min, mean
- SIMD vectorized kernels
- Thread pool with parallel execution
- Buffer pooling memory management

### Critical Gaps for Arctic
| Gap | Impact | Effort |
|-----|--------|--------|
| Batched matmul | Blocking - core of every layer | High |
| Softmax | Blocking - attention mechanism | Medium |
| LayerNorm | Blocking - every transformer block | Medium |
| Gather/Embedding | Blocking - token embeddings | Low |
| SafeTensors loader | Blocking - weight loading | Medium |
| Axis-specific reduction | Blocking - mean pooling | Low |

---

## Phase 1: Core Ops (P0)

### 1.1 Batched Matmul
**Files:** `src/core/shape.zig`, `src/backend/cpu/kernels/matmul.zig`, `src/backend/cpu/executor.zig`

Current limitation in `shape.zig:243`:
```zig
// Batched matmul - not yet implemented
@compileError("Batched matmul not yet implemented");
```

**Implementation:**
```zig
// shape.zig - MatmulShape for batched case
// [B, M, K] @ [B, K, N] -> [B, M, N]
// [B, M, K] @ [K, N] -> [B, M, N]  (broadcast)
const batch_a = A.dimensions[0..A.ndim-2];
const batch_b = B.dimensions[0..B.ndim-2];
// Broadcast batch dims, append M, N
```

```zig
// matmul.zig - batchedMatmul kernel
pub fn batchedMatmul(
    comptime T: type,
    a: []const T,
    b: []const T,
    c: []T,
    batch_size: usize,
    m: usize,
    k: usize,
    n: usize,
) void {
    const a_batch_stride = m * k;
    const b_batch_stride = k * n;
    const c_batch_stride = m * n;

    for (0..batch_size) |batch| {
        matmulTiled(
            T,
            a[batch * a_batch_stride..][0..a_batch_stride],
            b[batch * b_batch_stride..][0..b_batch_stride],
            c[batch * c_batch_stride..][0..c_batch_stride],
            m, k, n,
        );
    }
}
```

### 1.2 Softmax
**Files:** `src/ops/expr.zig`, `src/backend/cpu/kernels/reduce.zig` (or new `softmax.zig`)

```zig
// New op tag in expr.zig
pub const OpTag = enum {
    // ...existing...
    softmax,
};

// New expression type
pub fn SoftmaxExpr(comptime Input: type, comptime axis: isize) type {
    return struct {
        pub const ExpressionMarker = true;
        pub const kind: NodeKind = .softmax;
        pub const ElementType = ElementTypeOf(Input);
        pub const ndim = Input.ndim;
        pub const shape = Input.shape;
        pub const softmax_axis = normalizeAxis(ndim, axis);

        input: Input,
    };
}

// Kernel: softmax.zig
pub fn softmax(
    comptime T: type,
    input: []const T,
    output: []T,
    outer_size: usize,  // product of dims before axis
    axis_size: usize,   // size of softmax dim
    inner_size: usize,  // product of dims after axis
) void {
    for (0..outer_size) |outer| {
        for (0..inner_size) |inner| {
            const base = outer * axis_size * inner_size + inner;

            // Find max for numerical stability
            var max_val: T = -std.math.inf(T);
            for (0..axis_size) |i| {
                max_val = @max(max_val, input[base + i * inner_size]);
            }

            // Compute exp and sum
            var sum: T = 0;
            for (0..axis_size) |i| {
                const idx = base + i * inner_size;
                output[idx] = @exp(input[idx] - max_val);
                sum += output[idx];
            }

            // Normalize
            const inv_sum = 1.0 / sum;
            for (0..axis_size) |i| {
                output[base + i * inner_size] *= inv_sum;
            }
        }
    }
}
```

### 1.3 LayerNorm
**Files:** New `src/ops/layernorm.zig`, `src/backend/cpu/kernels/layernorm.zig`

```zig
// LayerNorm expression
pub fn LayerNormExpr(
    comptime Input: type,
    comptime Gamma: type,  // scale
    comptime Beta: type,   // bias
    comptime normalized_shape: anytype,
) type {
    return struct {
        pub const ExpressionMarker = true;
        pub const kind: NodeKind = .layernorm;
        pub const ElementType = ElementTypeOf(Input);
        pub const ndim = Input.ndim;
        pub const shape = Input.shape;

        input: Input,
        gamma: Gamma,
        beta: Beta,
    };
}

// Kernel
pub fn layernorm(
    comptime T: type,
    input: []const T,
    gamma: []const T,
    beta: []const T,
    output: []T,
    outer_size: usize,   // batch * seq
    norm_size: usize,    // 384 (hidden dim)
    eps: T,
) void {
    for (0..outer_size) |i| {
        const row = input[i * norm_size..][0..norm_size];
        const out_row = output[i * norm_size..][0..norm_size];

        // Compute mean
        var mean: T = 0;
        for (row) |x| mean += x;
        mean /= @as(T, @floatFromInt(norm_size));

        // Compute variance
        var variance: T = 0;
        for (row) |x| {
            const diff = x - mean;
            variance += diff * diff;
        }
        variance /= @as(T, @floatFromInt(norm_size));

        // Normalize and scale
        const inv_std = 1.0 / @sqrt(variance + eps);
        for (0..norm_size) |j| {
            out_row[j] = (row[j] - mean) * inv_std * gamma[j] + beta[j];
        }
    }
}
```

### 1.4 Gather (Embedding Lookup)
**Files:** `src/ops/expr.zig`, `src/backend/cpu/kernels/gather.zig`

```zig
// Gather expression for embedding lookup
pub fn GatherExpr(
    comptime Table: type,    // [vocab_size, embed_dim]
    comptime Indices: type,  // [batch, seq] of integers
) type {
    // Output: [batch, seq, embed_dim]
    const out_shape = Indices.shape ++ .{Table.shape[1]};

    return struct {
        pub const ExpressionMarker = true;
        pub const kind: NodeKind = .gather;
        pub const ElementType = ElementTypeOf(Table);
        pub const ndim = out_shape.len;
        pub const shape = out_shape;

        table: Table,
        indices: Indices,
    };
}

// Kernel
pub fn gather(
    comptime T: type,
    comptime IndexT: type,
    table: []const T,
    indices: []const IndexT,
    output: []T,
    vocab_size: usize,
    embed_dim: usize,
) void {
    for (indices, 0..) |idx, i| {
        const src = table[idx * embed_dim..][0..embed_dim];
        const dst = output[i * embed_dim..][0..embed_dim];
        @memcpy(dst, src);
    }
}
```

### 1.5 Axis-Specific Reduction with Mask
**Files:** `src/backend/cpu/kernels/reduce.zig`

Extend existing reduction to support:
- Reduction along specific axis (not just full reduction)
- Masked reduction for mean pooling with attention mask

```zig
pub fn maskedMeanPool(
    comptime T: type,
    input: []const T,      // [B, S, D]
    mask: []const T,       // [B, S] attention mask
    output: []T,           // [B, D]
    batch: usize,
    seq_len: usize,
    hidden_dim: usize,
) void {
    for (0..batch) |b| {
        // Count valid tokens
        var valid_count: T = 0;
        for (0..seq_len) |s| {
            valid_count += mask[b * seq_len + s];
        }

        // Sum over sequence with mask
        const out_row = output[b * hidden_dim..][0..hidden_dim];
        @memset(out_row, 0);

        for (0..seq_len) |s| {
            const m = mask[b * seq_len + s];
            if (m > 0) {
                const in_row = input[(b * seq_len + s) * hidden_dim..][0..hidden_dim];
                for (0..hidden_dim) |d| {
                    out_row[d] += in_row[d] * m;
                }
            }
        }

        // Divide by count
        for (out_row) |*v| {
            v.* /= valid_count;
        }
    }
}
```

---

## Phase 2: SafeTensors & Weight Loading

### 2.1 SafeTensors Parser
**File:** `src/models/safetensors.zig`

SafeTensors format:
```
[8 bytes: header_size (little endian u64)]
[header_size bytes: JSON header]
[remaining bytes: tensor data]
```

```zig
pub const SafeTensors = struct {
    header: Header,
    data: []const u8,

    pub const Header = struct {
        tensors: std.StringHashMap(TensorInfo),
        metadata: ?std.json.Value,
    };

    pub const TensorInfo = struct {
        dtype: DType,
        shape: []const usize,
        data_offsets: [2]usize,  // [start, end]
    };

    pub fn load(path: []const u8, allocator: Allocator) !SafeTensors {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        // Read header size
        var header_size_bytes: [8]u8 = undefined;
        _ = try file.readAll(&header_size_bytes);
        const header_size = std.mem.readInt(u64, &header_size_bytes, .little);

        // Read and parse JSON header
        const header_json = try allocator.alloc(u8, header_size);
        defer allocator.free(header_json);
        _ = try file.readAll(header_json);

        // Parse JSON...
        // Memory map remaining data...
    }

    pub fn getTensor(self: *const SafeTensors, name: []const u8) ?TensorView {
        const info = self.header.tensors.get(name) orelse return null;
        return TensorView{
            .data = self.data[info.data_offsets[0]..info.data_offsets[1]],
            .shape = info.shape,
            .dtype = info.dtype,
        };
    }
};
```

### 2.2 Model Weights Structure
**File:** `src/models/arctic.zig`

```zig
pub const ArcticEmbedXS = struct {
    // Embeddings
    word_embeddings: Tensor(f32, .{30522, 384}),
    position_embeddings: Tensor(f32, .{512, 384}),
    token_type_embeddings: Tensor(f32, .{2, 384}),
    embeddings_layernorm_weight: Tensor(f32, .{384}),
    embeddings_layernorm_bias: Tensor(f32, .{384}),

    // 12 transformer layers
    layers: [12]TransformerLayer,

    pub const TransformerLayer = struct {
        // Attention
        query_weight: Tensor(f32, .{384, 384}),
        query_bias: Tensor(f32, .{384}),
        key_weight: Tensor(f32, .{384, 384}),
        key_bias: Tensor(f32, .{384}),
        value_weight: Tensor(f32, .{384, 384}),
        value_bias: Tensor(f32, .{384}),
        attn_output_weight: Tensor(f32, .{384, 384}),
        attn_output_bias: Tensor(f32, .{384}),
        attn_layernorm_weight: Tensor(f32, .{384}),
        attn_layernorm_bias: Tensor(f32, .{384}),

        // FFN
        intermediate_weight: Tensor(f32, .{1536, 384}),
        intermediate_bias: Tensor(f32, .{1536}),
        output_weight: Tensor(f32, .{384, 1536}),
        output_bias: Tensor(f32, .{384}),
        output_layernorm_weight: Tensor(f32, .{384}),
        output_layernorm_bias: Tensor(f32, .{384}),
    };

    pub fn load(path: []const u8, allocator: Allocator) !ArcticEmbedXS {
        const st = try SafeTensors.load(path, allocator);
        // Map tensors by name...
    }
};
```

---

## Phase 3: Model Forward Pass

### 3.1 Embedding Layer
```zig
pub fn embeddings(
    self: *const ArcticEmbedXS,
    input_ids: Tensor(i32, .{B, S}),
    token_type_ids: Tensor(i32, .{B, S}),
    position_ids: Tensor(i32, .{B, S}),
) Tensor(f32, .{B, S, 384}) {
    // Gather word embeddings
    const word_emb = gather(self.word_embeddings, input_ids);

    // Gather position embeddings
    const pos_emb = gather(self.position_embeddings, position_ids);

    // Gather token type embeddings
    const type_emb = gather(self.token_type_embeddings, token_type_ids);

    // Sum and layer norm
    const sum = word_emb.add(pos_emb).add(type_emb);
    return layernorm(sum, self.embeddings_layernorm_weight, self.embeddings_layernorm_bias);
}
```

### 3.2 Multi-Head Attention
```zig
pub fn multiHeadAttention(
    hidden: Tensor(f32, .{B, S, 384}),
    layer: *const TransformerLayer,
    attention_mask: ?Tensor(f32, .{B, 1, 1, S}),
) Tensor(f32, .{B, S, 384}) {
    const num_heads = 6;
    const head_dim = 64;

    // Q, K, V projections: [B, S, 384] @ [384, 384] -> [B, S, 384]
    const Q = hidden.matmul(layer.query_weight.transpose()).add(layer.query_bias);
    const K = hidden.matmul(layer.key_weight.transpose()).add(layer.key_bias);
    const V = hidden.matmul(layer.value_weight.transpose()).add(layer.value_bias);

    // Reshape to [B, S, 6, 64] then transpose to [B, 6, S, 64]
    const Q_heads = Q.reshape(.{B, S, num_heads, head_dim}).transpose(.{0, 2, 1, 3});
    const K_heads = K.reshape(.{B, S, num_heads, head_dim}).transpose(.{0, 2, 1, 3});
    const V_heads = V.reshape(.{B, S, num_heads, head_dim}).transpose(.{0, 2, 1, 3});

    // Attention scores: [B, 6, S, 64] @ [B, 6, 64, S] -> [B, 6, S, S]
    const scale = 1.0 / @sqrt(@as(f32, head_dim));
    var scores = Q_heads.matmul(K_heads.transpose(.{0, 1, 3, 2})).mul(scale);

    // Apply mask if provided
    if (attention_mask) |mask| {
        scores = scores.add(mask);  // mask has -inf for padding
    }

    // Softmax over last dim
    const attn_weights = scores.softmax(-1);

    // Apply to values: [B, 6, S, S] @ [B, 6, S, 64] -> [B, 6, S, 64]
    const attn_output = attn_weights.matmul(V_heads);

    // Reshape back: [B, 6, S, 64] -> [B, S, 6, 64] -> [B, S, 384]
    const concat = attn_output.transpose(.{0, 2, 1, 3}).reshape(.{B, S, 384});

    // Output projection
    return concat.matmul(layer.attn_output_weight.transpose()).add(layer.attn_output_bias);
}
```

### 3.3 Feed-Forward Network
```zig
pub fn feedForward(
    hidden: Tensor(f32, .{B, S, 384}),
    layer: *const TransformerLayer,
) Tensor(f32, .{B, S, 384}) {
    // Up projection: [B, S, 384] @ [384, 1536] -> [B, S, 1536]
    const intermediate = hidden
        .matmul(layer.intermediate_weight.transpose())
        .add(layer.intermediate_bias)
        .gelu();

    // Down projection: [B, S, 1536] @ [1536, 384] -> [B, S, 384]
    return intermediate
        .matmul(layer.output_weight.transpose())
        .add(layer.output_bias);
}
```

### 3.4 Transformer Block
```zig
pub fn transformerBlock(
    hidden: Tensor(f32, .{B, S, 384}),
    layer: *const TransformerLayer,
    attention_mask: ?Tensor(f32, .{B, 1, 1, S}),
) Tensor(f32, .{B, S, 384}) {
    // Pre-norm attention
    const normed1 = layernorm(hidden, layer.attn_layernorm_weight, layer.attn_layernorm_bias);
    const attn_out = multiHeadAttention(normed1, layer, attention_mask);
    const residual1 = hidden.add(attn_out);

    // Pre-norm FFN
    const normed2 = layernorm(residual1, layer.output_layernorm_weight, layer.output_layernorm_bias);
    const ffn_out = feedForward(normed2, layer);
    return residual1.add(ffn_out);
}
```

### 3.5 Full Forward + Pooling
```zig
pub fn forward(
    self: *const ArcticEmbedXS,
    input_ids: Tensor(i32, .{B, S}),
    attention_mask: Tensor(f32, .{B, S}),
) Tensor(f32, .{B, 384}) {
    // Position IDs: 0, 1, 2, ..., S-1
    const position_ids = Tensor(i32, .{B, S}).arange();
    const token_type_ids = Tensor(i32, .{B, S}).zeros();

    // Embeddings
    var hidden = self.embeddings(input_ids, token_type_ids, position_ids);

    // Expand mask for attention: [B, S] -> [B, 1, 1, S]
    const attn_mask = attention_mask
        .unsqueeze(1)
        .unsqueeze(2)
        .sub(1.0)
        .mul(-1e9);  // 0 -> 0, 1 -> -inf

    // 12 transformer layers
    for (self.layers) |*layer| {
        hidden = transformerBlock(hidden, layer, attn_mask);
    }

    // Mean pooling over sequence (masked)
    const pooled = maskedMeanPool(hidden, attention_mask);

    // L2 normalize
    const norm = pooled.pow(2.0).sum(-1, true).sqrt();
    return pooled.div(norm);
}
```

---

## Phase 4: Testing

### 4.1 Reference Fixtures Generation (Python)
```python
# scripts/generate_fixtures.py
from transformers import AutoModel, AutoTokenizer
import torch
import json
import struct

model = AutoModel.from_pretrained("Snowflake/snowflake-arctic-embed-xs")
tok = AutoTokenizer.from_pretrained("Snowflake/snowflake-arctic-embed-xs")

# Test input
text = "The quick brown fox jumps over the lazy dog"
inputs = tok(text, return_tensors="pt", padding="max_length", max_length=32)

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

# Save fixtures
with open("test/arctic/fixtures/input_tokens.json", "w") as f:
    json.dump(inputs["input_ids"].tolist(), f)

with open("test/arctic/fixtures/attention_mask.json", "w") as f:
    json.dump(inputs["attention_mask"].tolist(), f)

# Save layer outputs
for i, hidden in enumerate(outputs.hidden_states):
    with open(f"test/arctic/fixtures/layer_{i}_output.bin", "wb") as f:
        f.write(hidden.numpy().astype("float32").tobytes())

# Save final embedding (pooled + normalized)
pooled = outputs.last_hidden_state.mean(dim=1)
normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
with open("test/arctic/fixtures/final_embedding.bin", "wb") as f:
    f.write(normalized.numpy().astype("float32").tobytes())
```

### 4.2 Test Structure
```
test/
├── arctic/
│   ├── ops_test.zig           # Unit tests for new ops
│   │   ├── test "batched matmul"
│   │   ├── test "softmax axis"
│   │   ├── test "layernorm"
│   │   └── test "gather embedding"
│   ├── forward_test.zig       # Integration tests
│   │   ├── test "single layer forward"
│   │   └── test "full model forward"
│   └── fixtures/
│       ├── input_tokens.json
│       ├── attention_mask.json
│       ├── layer_0_output.bin
│       ├── layer_6_output.bin
│       └── final_embedding.bin
```

---

## Implementation Order

### Week 1: Foundation
1. **Batched matmul** in shape.zig + matmul.zig + executor.zig
2. **Axis-aware transpose** (permutation support)
3. **Reshape expression** type wiring

### Week 2: Transformer Ops
4. **Softmax** op + kernel
5. **LayerNorm** op + kernel
6. **Gather** op + kernel (embedding lookup)

### Week 3: Model Loading
7. **SafeTensors parser**
8. **Arctic model struct** with weight mapping
9. **Basic forward pass** (single layer test)

### Week 4: Full Model
10. **12-layer forward**
11. **Mean pooling + L2 norm**
12. **End-to-end test** against reference

### Week 5: Polish
13. **Fused kernels** (optional perf)
14. **Example CLI**
15. **Benchmarks**

---

## Success Criteria

1. `zig build test` - all tests pass
2. Layer 0 output matches fixture within 1e-5
3. Final embedding cosine similarity > 0.9999 vs HuggingFace
4. Single binary < 10MB
5. Inference on sample: < 100ms (B=1, S=32)

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Batched matmul complexity | Start with loop over batches, optimize later |
| Shape algebra bugs | Extensive comptime tests |
| Numerical precision | Use f32 throughout, compare layer-by-layer |
| Memory usage | Use arena allocator for expression eval |
