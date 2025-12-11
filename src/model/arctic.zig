//! Arctic-embed-xs model implementation.
//!
//! Snowflake's arctic-embed-xs is a 22M parameter embedding model based on
//! MiniLM-L6-H384 (BERT architecture). It produces 384-dimensional embeddings
//! using CLS token pooling with L2 normalization.
//!
//! Architecture:
//! - hidden_size: 384
//! - num_hidden_layers: 6
//! - num_attention_heads: 12 (head_dim = 32)
//! - intermediate_size: 1536
//! - vocab_size: 30522
//! - max_position_embeddings: 512

const std = @import("std");
const safetensors = @import("../io/safetensors.zig");
const kernels = struct {
    const matmul = @import("../backend/cpu/kernels/matmul.zig");
    const softmax = @import("../backend/cpu/kernels/softmax.zig");
    const layernorm = @import("../backend/cpu/kernels/layernorm.zig");
    const transpose = @import("../backend/cpu/kernels/transpose.zig");
    const elementwise = @import("../backend/cpu/kernels/elementwise.zig");
};

/// Model configuration for arctic-embed-xs.
pub const Config = struct {
    hidden_size: usize = 384,
    num_hidden_layers: usize = 6,
    num_attention_heads: usize = 12,
    intermediate_size: usize = 1536,
    vocab_size: usize = 30522,
    max_position_embeddings: usize = 512,
    type_vocab_size: usize = 2,
    layer_norm_eps: f32 = 1e-12,

    pub fn headDim(self: Config) usize {
        return self.hidden_size / self.num_attention_heads;
    }
};

/// Default config for arctic-embed-xs.
pub const arctic_embed_xs_config = Config{};

/// Embedding layer weights.
pub const EmbeddingWeights = struct {
    word_embeddings: []const f32, // [vocab_size, hidden_size]
    position_embeddings: []const f32, // [max_position_embeddings, hidden_size]
    token_type_embeddings: []const f32, // [type_vocab_size, hidden_size]
    layer_norm_gamma: []const f32, // [hidden_size]
    layer_norm_beta: []const f32, // [hidden_size]
};

/// Self-attention weights for one layer.
pub const AttentionWeights = struct {
    query_weight: []const f32, // [hidden_size, hidden_size]
    query_bias: []const f32, // [hidden_size]
    key_weight: []const f32, // [hidden_size, hidden_size]
    key_bias: []const f32, // [hidden_size]
    value_weight: []const f32, // [hidden_size, hidden_size]
    value_bias: []const f32, // [hidden_size]
    output_weight: []const f32, // [hidden_size, hidden_size]
    output_bias: []const f32, // [hidden_size]
    output_ln_gamma: []const f32, // [hidden_size]
    output_ln_beta: []const f32, // [hidden_size]
};

/// Feed-forward (MLP) weights for one layer.
pub const FeedForwardWeights = struct {
    intermediate_weight: []const f32, // [hidden_size, intermediate_size]
    intermediate_bias: []const f32, // [intermediate_size]
    output_weight: []const f32, // [intermediate_size, hidden_size]
    output_bias: []const f32, // [hidden_size]
    output_ln_gamma: []const f32, // [hidden_size]
    output_ln_beta: []const f32, // [hidden_size]
};

/// Complete transformer layer weights.
pub const LayerWeights = struct {
    attention: AttentionWeights,
    ffn: FeedForwardWeights,
};

/// Complete model weights.
pub const ModelWeights = struct {
    config: Config,
    embeddings: EmbeddingWeights,
    layers: []LayerWeights,

    /// Load weights from a SafeTensors file.
    /// Handles both "bert." prefixed and non-prefixed weight names.
    pub fn fromSafeTensors(
        allocator: std.mem.Allocator,
        st: safetensors.SafeTensors,
        config: Config,
    ) !ModelWeights {
        // Detect if weights have "bert." prefix
        const has_bert_prefix = st.get("bert.embeddings.word_embeddings.weight") != null;

        // Allocate layer weights
        const layers = try allocator.alloc(LayerWeights, config.num_hidden_layers);
        errdefer allocator.free(layers);

        // Load embedding weights (handle both gamma/beta and weight/bias naming)
        const emb_weights = if (has_bert_prefix) EmbeddingWeights{
            .word_embeddings = getWeightData(st, "bert.embeddings.word_embeddings.weight"),
            .position_embeddings = getWeightData(st, "bert.embeddings.position_embeddings.weight"),
            .token_type_embeddings = getWeightData(st, "bert.embeddings.token_type_embeddings.weight"),
            .layer_norm_gamma = getWeightDataAlt(st, "bert.embeddings.LayerNorm.weight", "bert.embeddings.LayerNorm.gamma"),
            .layer_norm_beta = getWeightDataAlt(st, "bert.embeddings.LayerNorm.bias", "bert.embeddings.LayerNorm.beta"),
        } else EmbeddingWeights{
            .word_embeddings = getWeightData(st, "embeddings.word_embeddings.weight"),
            .position_embeddings = getWeightData(st, "embeddings.position_embeddings.weight"),
            .token_type_embeddings = getWeightData(st, "embeddings.token_type_embeddings.weight"),
            .layer_norm_gamma = getWeightDataAlt(st, "embeddings.LayerNorm.weight", "embeddings.LayerNorm.gamma"),
            .layer_norm_beta = getWeightDataAlt(st, "embeddings.LayerNorm.bias", "embeddings.LayerNorm.beta"),
        };

        // Load each transformer layer
        for (0..config.num_hidden_layers) |i| {
            layers[i] = loadLayerWeights(st, i, has_bert_prefix);
        }

        return .{
            .config = config,
            .embeddings = emb_weights,
            .layers = layers,
        };
    }

    pub fn deinit(self: *ModelWeights, allocator: std.mem.Allocator) void {
        allocator.free(self.layers);
    }
};

/// Helper to get f32 weight data from SafeTensors by name.
fn getWeightData(st: safetensors.SafeTensors, name: []const u8) []const f32 {
    const info = st.get(name) orelse {
        std.log.err("Missing weight: {s}", .{name});
        @panic("Missing required weight");
    };
    if (info.dtype != .F32) {
        std.log.err("Weight {s} has wrong dtype, expected F32", .{name});
        @panic("Wrong weight dtype");
    }
    return st.getData(f32, info);
}

/// Helper to try primary name, fall back to alternate (for weight/gamma, bias/beta).
fn getWeightDataAlt(st: safetensors.SafeTensors, primary: []const u8, alternate: []const u8) []const f32 {
    if (st.get(primary)) |info| {
        if (info.dtype != .F32) @panic("Wrong weight dtype");
        return st.getData(f32, info);
    }
    return getWeightData(st, alternate);
}

/// Load attention weights for a specific layer.
fn loadAttentionWeights(st: safetensors.SafeTensors, layer_idx: usize, has_bert_prefix: bool) AttentionWeights {
    var buf: [128]u8 = undefined;
    var alt_buf: [128]u8 = undefined;

    const base = if (has_bert_prefix) "bert.encoder.layer" else "encoder.layer";

    return .{
        .query_weight = getLayerWeightFmt(st, &buf, base, layer_idx, "attention.self.query.weight"),
        .query_bias = getLayerWeightFmt(st, &buf, base, layer_idx, "attention.self.query.bias"),
        .key_weight = getLayerWeightFmt(st, &buf, base, layer_idx, "attention.self.key.weight"),
        .key_bias = getLayerWeightFmt(st, &buf, base, layer_idx, "attention.self.key.bias"),
        .value_weight = getLayerWeightFmt(st, &buf, base, layer_idx, "attention.self.value.weight"),
        .value_bias = getLayerWeightFmt(st, &buf, base, layer_idx, "attention.self.value.bias"),
        .output_weight = getLayerWeightFmt(st, &buf, base, layer_idx, "attention.output.dense.weight"),
        .output_bias = getLayerWeightFmt(st, &buf, base, layer_idx, "attention.output.dense.bias"),
        .output_ln_gamma = getLayerWeightAlt(st, &buf, &alt_buf, base, layer_idx, "attention.output.LayerNorm.weight", "attention.output.LayerNorm.gamma"),
        .output_ln_beta = getLayerWeightAlt(st, &buf, &alt_buf, base, layer_idx, "attention.output.LayerNorm.bias", "attention.output.LayerNorm.beta"),
    };
}

/// Load feed-forward weights for a specific layer.
fn loadFeedForwardWeights(st: safetensors.SafeTensors, layer_idx: usize, has_bert_prefix: bool) FeedForwardWeights {
    var buf: [128]u8 = undefined;
    var alt_buf: [128]u8 = undefined;

    const base = if (has_bert_prefix) "bert.encoder.layer" else "encoder.layer";

    return .{
        .intermediate_weight = getLayerWeightFmt(st, &buf, base, layer_idx, "intermediate.dense.weight"),
        .intermediate_bias = getLayerWeightFmt(st, &buf, base, layer_idx, "intermediate.dense.bias"),
        .output_weight = getLayerWeightFmt(st, &buf, base, layer_idx, "output.dense.weight"),
        .output_bias = getLayerWeightFmt(st, &buf, base, layer_idx, "output.dense.bias"),
        .output_ln_gamma = getLayerWeightAlt(st, &buf, &alt_buf, base, layer_idx, "output.LayerNorm.weight", "output.LayerNorm.gamma"),
        .output_ln_beta = getLayerWeightAlt(st, &buf, &alt_buf, base, layer_idx, "output.LayerNorm.bias", "output.LayerNorm.beta"),
    };
}

/// Load all weights for a specific layer.
fn loadLayerWeights(st: safetensors.SafeTensors, layer_idx: usize, has_bert_prefix: bool) LayerWeights {
    return .{
        .attention = loadAttentionWeights(st, layer_idx, has_bert_prefix),
        .ffn = loadFeedForwardWeights(st, layer_idx, has_bert_prefix),
    };
}

/// Get a weight with layer-prefixed name.
fn getLayerWeightFmt(st: safetensors.SafeTensors, buf: *[128]u8, base: []const u8, layer_idx: usize, suffix: []const u8) []const f32 {
    const name = std.fmt.bufPrint(buf, "{s}.{d}.{s}", .{ base, layer_idx, suffix }) catch unreachable;
    return getWeightData(st, name);
}

/// Get a weight with fallback for gamma/beta naming.
fn getLayerWeightAlt(st: safetensors.SafeTensors, buf: *[128]u8, alt_buf: *[128]u8, base: []const u8, layer_idx: usize, primary_suffix: []const u8, alt_suffix: []const u8) []const f32 {
    const primary = std.fmt.bufPrint(buf, "{s}.{d}.{s}", .{ base, layer_idx, primary_suffix }) catch unreachable;
    const alternate = std.fmt.bufPrint(alt_buf, "{s}.{d}.{s}", .{ base, layer_idx, alt_suffix }) catch unreachable;
    return getWeightDataAlt(st, primary, alternate);
}

// ============================================================================
// Forward Pass (Runtime inference with variable sequence lengths)
// ============================================================================

/// Inference context - holds intermediate buffers for forward pass.
pub const InferenceContext = struct {
    allocator: std.mem.Allocator,
    config: Config,

    // Intermediate buffers (allocated once, reused)
    hidden: []f32, // [seq_len, hidden_size] - for transformer block intermediate
    qkv: []f32, // For Q, K, V projections
    attn_scores: []f32, // [num_heads, seq_len, seq_len]
    attn_out: []f32, // [seq_len, hidden_size]
    ffn_intermediate: []f32, // [seq_len, intermediate_size]
    tmp: []f32, // Scratch buffer for encoder layer chaining

    // Additional buffers for full forward pass
    encoder_input: []f32, // Input to encoder
    encoder_output: []f32, // Output from encoder
    layer_input: []f32, // For encoder layer ping-pong (separate from tmp!)

    max_seq_len: usize,

    pub fn init(allocator: std.mem.Allocator, config: Config, max_seq_len: usize) !InferenceContext {
        const h = config.hidden_size;
        const inter = config.intermediate_size;
        const heads = config.num_attention_heads;

        return .{
            .allocator = allocator,
            .config = config,
            .hidden = try allocator.alloc(f32, max_seq_len * h),
            .qkv = try allocator.alloc(f32, 3 * max_seq_len * h),
            .attn_scores = try allocator.alloc(f32, heads * max_seq_len * max_seq_len),
            .attn_out = try allocator.alloc(f32, max_seq_len * h),
            .ffn_intermediate = try allocator.alloc(f32, max_seq_len * inter),
            .tmp = try allocator.alloc(f32, max_seq_len * h),
            .encoder_input = try allocator.alloc(f32, max_seq_len * h),
            .encoder_output = try allocator.alloc(f32, max_seq_len * h),
            .layer_input = try allocator.alloc(f32, max_seq_len * h),
            .max_seq_len = max_seq_len,
        };
    }

    pub fn deinit(self: *InferenceContext) void {
        self.allocator.free(self.hidden);
        self.allocator.free(self.qkv);
        self.allocator.free(self.attn_scores);
        self.allocator.free(self.attn_out);
        self.allocator.free(self.ffn_intermediate);
        self.allocator.free(self.tmp);
        self.allocator.free(self.encoder_input);
        self.allocator.free(self.encoder_output);
        self.allocator.free(self.layer_input);
    }
};

/// Apply embeddings: word + position + token_type + LayerNorm.
/// Output: [seq_len, hidden_size]
pub fn embeddings(
    output: []f32,
    token_ids: []const u32,
    weights: EmbeddingWeights,
    config: Config,
) void {
    const seq_len = token_ids.len;
    const h = config.hidden_size;

    // Sum word embeddings + position embeddings + token_type embeddings
    for (token_ids, 0..) |tok_id, pos| {
        const out_start = pos * h;
        const word_start = tok_id * h;
        const pos_start = pos * h;

        for (0..h) |i| {
            output[out_start + i] =
                weights.word_embeddings[word_start + i] +
                weights.position_embeddings[pos_start + i] +
                weights.token_type_embeddings[i]; // Token type 0 for all tokens
        }
    }

    // Apply LayerNorm in-place
    kernels.layernorm.layerNormLastDim(
        f32,
        output[0 .. seq_len * h],
        weights.layer_norm_gamma,
        weights.layer_norm_beta,
        output[0 .. seq_len * h],
        2,
        .{ seq_len, h },
        config.layer_norm_eps,
    );
}

/// Linear projection: Y = XW^T + b (BERT weights are stored transposed)
fn linearProjection(
    output: []f32,
    input: []const f32,
    weight: []const f32, // [out_features, in_features] (transposed)
    bias: []const f32, // [out_features]
    seq_len: usize,
    in_features: usize,
    out_features: usize,
) void {
    // Matmul: [seq_len, in_features] @ [out_features, in_features]^T = [seq_len, out_features]
    // Weight is stored as [out_features, in_features], so we use transpose_b
    kernels.matmul.matmulTransposeB(f32, input, weight, output, seq_len, in_features, out_features);

    // Add bias
    for (0..seq_len) |s| {
        const start = s * out_features;
        for (0..out_features) |i| {
            output[start + i] += bias[i];
        }
    }
}

/// Self-attention forward pass for one layer.
/// Input/Output: [seq_len, hidden_size]
pub fn selfAttention(
    output: []f32,
    input: []const f32,
    attn: AttentionWeights,
    ctx: *InferenceContext,
    seq_len: usize,
) void {
    const h = ctx.config.hidden_size;
    const num_heads = ctx.config.num_attention_heads;
    const head_dim = ctx.config.headDim();
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    // Project Q, K, V
    const q = ctx.qkv[0 .. seq_len * h];
    const k = ctx.qkv[seq_len * h .. 2 * seq_len * h];
    const v = ctx.qkv[2 * seq_len * h .. 3 * seq_len * h];

    linearProjection(q, input, attn.query_weight, attn.query_bias, seq_len, h, h);
    linearProjection(k, input, attn.key_weight, attn.key_bias, seq_len, h, h);
    linearProjection(v, input, attn.value_weight, attn.value_bias, seq_len, h, h);

    // Multi-head attention: for each head
    for (0..num_heads) |head| {
        const head_offset = head * head_dim;
        const scores_offset = head * seq_len * seq_len;

        // Compute attention scores: Q_h @ K_h^T / sqrt(d_k)
        for (0..seq_len) |qi| {
            for (0..seq_len) |ki| {
                var dot: f32 = 0;
                for (0..head_dim) |d| {
                    const q_idx = qi * h + head_offset + d;
                    const k_idx = ki * h + head_offset + d;
                    dot += q[q_idx] * k[k_idx];
                }
                ctx.attn_scores[scores_offset + qi * seq_len + ki] = dot * scale;
            }
        }

        // Softmax over last dimension
        for (0..seq_len) |qi| {
            const row_start = scores_offset + qi * seq_len;
            const row = ctx.attn_scores[row_start..][0..seq_len];

            // Find max for numerical stability
            var max_val: f32 = row[0];
            for (row[1..]) |val| {
                if (val > max_val) max_val = val;
            }

            // Exp and sum
            var sum: f32 = 0;
            for (row) |*val| {
                val.* = @exp(val.* - max_val);
                sum += val.*;
            }

            // Normalize
            for (row) |*val| {
                val.* /= sum;
            }
        }

        // Apply attention: scores @ V_h -> attn_out
        for (0..seq_len) |qi| {
            for (0..head_dim) |d| {
                var sum: f32 = 0;
                for (0..seq_len) |vi| {
                    const score = ctx.attn_scores[scores_offset + qi * seq_len + vi];
                    const v_idx = vi * h + head_offset + d;
                    sum += score * v[v_idx];
                }
                ctx.attn_out[qi * h + head_offset + d] = sum;
            }
        }
    }

    // Output projection
    linearProjection(ctx.tmp[0 .. seq_len * h], ctx.attn_out[0 .. seq_len * h], attn.output_weight, attn.output_bias, seq_len, h, h);

    // Residual connection + LayerNorm
    for (0..seq_len * h) |i| {
        output[i] = input[i] + ctx.tmp[i];
    }
    kernels.layernorm.layerNormLastDim(
        f32,
        output[0 .. seq_len * h],
        attn.output_ln_gamma,
        attn.output_ln_beta,
        output[0 .. seq_len * h],
        2,
        .{ seq_len, h },
        ctx.config.layer_norm_eps,
    );
}

/// GELU activation (approximation used by BERT).
fn gelu(x: f32) f32 {
    // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const sqrt_2_over_pi: f32 = 0.7978845608;
    const coeff: f32 = 0.044715;
    const inner = sqrt_2_over_pi * (x + coeff * x * x * x);
    return 0.5 * x * (1.0 + std.math.tanh(inner));
}

/// Feed-forward network (MLP) forward pass.
/// Input/Output: [seq_len, hidden_size]
pub fn feedForward(
    output: []f32,
    input: []const f32,
    ffn: FeedForwardWeights,
    ctx: *InferenceContext,
    seq_len: usize,
) void {
    const h = ctx.config.hidden_size;
    const inter = ctx.config.intermediate_size;

    // Intermediate: GELU(XW1 + b1)
    linearProjection(ctx.ffn_intermediate[0 .. seq_len * inter], input, ffn.intermediate_weight, ffn.intermediate_bias, seq_len, h, inter);

    // Apply GELU
    for (ctx.ffn_intermediate[0 .. seq_len * inter]) |*val| {
        val.* = gelu(val.*);
    }

    // Output: hW2 + b2
    linearProjection(ctx.tmp[0 .. seq_len * h], ctx.ffn_intermediate[0 .. seq_len * inter], ffn.output_weight, ffn.output_bias, seq_len, inter, h);

    // Residual connection + LayerNorm
    for (0..seq_len * h) |i| {
        output[i] = input[i] + ctx.tmp[i];
    }
    kernels.layernorm.layerNormLastDim(
        f32,
        output[0 .. seq_len * h],
        ffn.output_ln_gamma,
        ffn.output_ln_beta,
        output[0 .. seq_len * h],
        2,
        .{ seq_len, h },
        ctx.config.layer_norm_eps,
    );
}

/// Single transformer block forward pass.
/// Performs self-attention followed by feed-forward network.
pub fn transformerBlock(
    output: []f32,
    input: []const f32,
    layer: LayerWeights,
    ctx: *InferenceContext,
    seq_len: usize,
) void {
    // Self-attention (output written to ctx.hidden)
    selfAttention(ctx.hidden[0 .. seq_len * ctx.config.hidden_size], input, layer.attention, ctx, seq_len);

    // Feed-forward (reads from ctx.hidden, writes to output)
    feedForward(output, ctx.hidden[0 .. seq_len * ctx.config.hidden_size], layer.ffn, ctx, seq_len);
}

/// Full encoder forward pass: all transformer layers.
/// Input: [seq_len, hidden_size], Output: [seq_len, hidden_size]
pub fn encoder(
    output: []f32,
    input: []const f32,
    weights: ModelWeights,
    ctx: *InferenceContext,
    seq_len: usize,
) void {
    const h = ctx.config.hidden_size;

    // First layer: input -> output
    transformerBlock(output, input, weights.layers[0], ctx, seq_len);

    // Remaining layers: output -> output (in-place with ping-pong)
    // Use layer_input buffer which is separate from tmp used inside transformerBlock
    for (weights.layers[1..]) |layer| {
        @memcpy(ctx.layer_input[0 .. seq_len * h], output[0 .. seq_len * h]);
        transformerBlock(output, ctx.layer_input[0 .. seq_len * h], layer, ctx, seq_len);
    }
}

/// L2 normalize a vector in-place.
pub fn l2Normalize(vec: []f32) void {
    var norm_sq: f32 = 0;
    for (vec) |v| {
        norm_sq += v * v;
    }

    const norm = @sqrt(norm_sq);
    if (norm > 1e-12) {
        for (vec) |*v| {
            v.* /= norm;
        }
    }
}

/// Complete arctic-embed forward pass.
/// Input: token_ids
/// Output: L2-normalized embedding vector [hidden_size]
///
/// Steps:
/// 1. Embeddings (word + position + token_type + LayerNorm)
/// 2. 6 transformer layers
/// 3. CLS token pooling (first token)
/// 4. L2 normalization
pub fn forward(
    output: []f32, // [hidden_size]
    token_ids: []const u32,
    weights: ModelWeights,
    ctx: *InferenceContext,
) void {
    const seq_len = token_ids.len;
    const h = ctx.config.hidden_size;

    // Use dedicated buffers that don't conflict with transformer internals
    const hidden_states = ctx.encoder_input[0 .. seq_len * h];
    const enc_output = ctx.encoder_output[0 .. seq_len * h];

    // Step 1: Embeddings
    embeddings(hidden_states, token_ids, weights.embeddings, ctx.config);

    // Step 2: Encoder (all transformer layers)
    encoder(enc_output, hidden_states, weights, ctx, seq_len);

    // Step 3: CLS pooling (extract first token)
    @memcpy(output[0..h], enc_output[0..h]);

    // Step 4: L2 normalization
    l2Normalize(output[0..h]);
}

/// Batch inference: process multiple sequences.
/// Sequences must be pre-padded to the same length.
pub fn forwardBatch(
    outputs: []f32, // [batch_size * hidden_size]
    token_ids_batch: []const []const u32,
    weights: ModelWeights,
    ctx: *InferenceContext,
) void {
    const h = ctx.config.hidden_size;

    for (token_ids_batch, 0..) |token_ids, batch_idx| {
        forward(
            outputs[batch_idx * h ..][0..h],
            token_ids,
            weights,
            ctx,
        );
    }
}

// ============================================================================
// Tests
// ============================================================================

test "config head dim" {
    const config = arctic_embed_xs_config;
    try std.testing.expectEqual(@as(usize, 32), config.headDim());
}

test "config defaults" {
    const config = Config{};
    try std.testing.expectEqual(@as(usize, 384), config.hidden_size);
    try std.testing.expectEqual(@as(usize, 6), config.num_hidden_layers);
    try std.testing.expectEqual(@as(usize, 12), config.num_attention_heads);
    try std.testing.expectEqual(@as(usize, 1536), config.intermediate_size);
}

test "gelu activation" {
    // GELU(0) = 0
    try std.testing.expectApproxEqAbs(@as(f32, 0), gelu(0), 1e-6);

    // GELU(-3) ≈ -0.00405 (very small negative)
    try std.testing.expect(gelu(-3) < 0);
    try std.testing.expect(gelu(-3) > -0.01);

    // GELU(3) ≈ 2.996 (close to x for large positive)
    try std.testing.expectApproxEqAbs(@as(f32, 3), gelu(3), 0.01);

    // GELU is monotonic
    try std.testing.expect(gelu(1) < gelu(2));
}

test "l2 normalize" {
    // Vector [3, 4] normalized is [0.6, 0.8]
    var vec = [_]f32{ 3, 4 };
    l2Normalize(&vec);
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), vec[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), vec[1], 1e-6);

    // Verify unit norm
    const norm = @sqrt(vec[0] * vec[0] + vec[1] * vec[1]);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), norm, 1e-6);
}

test "inference context init/deinit" {
    const config = arctic_embed_xs_config;
    var ctx = try InferenceContext.init(std.testing.allocator, config, 128);
    defer ctx.deinit();

    try std.testing.expectEqual(@as(usize, 128), ctx.max_seq_len);
    try std.testing.expectEqual(@as(usize, 128 * 384), ctx.hidden.len);
}
