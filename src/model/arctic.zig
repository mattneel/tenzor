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
const threading = @import("../backend/cpu/threading.zig");
const simd = @import("../backend/cpu/simd.zig");
const kernels = struct {
    const matmul = @import("../backend/cpu/kernels/matmul.zig");
    const matmul_w8a32 = @import("../backend/cpu/kernels/matmul_w8a32.zig");
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

pub const QuantRowI8 = struct {
    /// Quantized weight data stored row-major as [out, in].
    data: []const i8,
    /// Per-row symmetric scale (dequant: f32 = i8 * scale[row]).
    scale: []const f16,
    // zero-point intentionally omitted for symmetric v1
};

pub const MatWeight = union(enum) {
    f32: []const f32,
    q8: QuantRowI8,
};

/// Embedding layer weights.
pub const EmbeddingWeights = struct {
    word_embeddings: MatWeight, // [vocab_size, hidden_size]
    position_embeddings: MatWeight, // [max_position_embeddings, hidden_size]
    token_type_embeddings: MatWeight, // [type_vocab_size, hidden_size]
    layer_norm_gamma: []const f32, // [hidden_size]
    layer_norm_beta: []const f32, // [hidden_size]
};

/// Self-attention weights for one layer.
pub const AttentionWeights = struct {
    query_weight: MatWeight, // [hidden_size, hidden_size]
    query_bias: []const f32, // [hidden_size]
    key_weight: MatWeight, // [hidden_size, hidden_size]
    key_bias: []const f32, // [hidden_size]
    value_weight: MatWeight, // [hidden_size, hidden_size]
    value_bias: []const f32, // [hidden_size]
    output_weight: MatWeight, // [hidden_size, hidden_size]
    output_bias: []const f32, // [hidden_size]
    output_ln_gamma: []const f32, // [hidden_size]
    output_ln_beta: []const f32, // [hidden_size]
};

/// Feed-forward (MLP) weights for one layer.
pub const FeedForwardWeights = struct {
    intermediate_weight: MatWeight, // [hidden_size, intermediate_size]
    intermediate_bias: []const f32, // [intermediate_size]
    output_weight: MatWeight, // [intermediate_size, hidden_size]
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
    allocator: std.mem.Allocator,
    // Track all allocated weight buffers for cleanup
    allocated_f32: std.ArrayList([]const f32),
    allocated_f16: std.ArrayList([]const f16),
    allocated_i8: std.ArrayList([]const i8),

    /// Load weights from a SafeTensors file.
    /// Handles both "bert." prefixed and non-prefixed weight names.
    /// Copies weights to aligned memory (required for @embedFile sources).
    pub fn fromSafeTensors(
        allocator: std.mem.Allocator,
        st: safetensors.SafeTensors,
        config: Config,
    ) !ModelWeights {
        var allocated_f32: std.ArrayList([]const f32) = .empty;
        var allocated_f16: std.ArrayList([]const f16) = .empty;
        var allocated_i8: std.ArrayList([]const i8) = .empty;
        errdefer {
            for (allocated_f32.items) |w| allocator.free(w);
            for (allocated_f16.items) |w| allocator.free(w);
            for (allocated_i8.items) |w| allocator.free(w);
            allocated_f32.deinit(allocator);
            allocated_f16.deinit(allocator);
            allocated_i8.deinit(allocator);
        }

        // Detect if weights have "bert." prefix
        const has_bert_prefix = st.get("bert.embeddings.word_embeddings.weight") != null;

        // Allocate layer weights
        const layers = try allocator.alloc(LayerWeights, config.num_hidden_layers);
        errdefer allocator.free(layers);

        // Load embedding weights (handle both gamma/beta and weight/bias naming)
        const emb_weights = if (has_bert_prefix) EmbeddingWeights{
            .word_embeddings = try getMatWeightAlloc(allocator, st, "bert.embeddings.word_embeddings.weight", &allocated_f32, &allocated_f16, &allocated_i8),
            .position_embeddings = try getMatWeightAlloc(allocator, st, "bert.embeddings.position_embeddings.weight", &allocated_f32, &allocated_f16, &allocated_i8),
            .token_type_embeddings = try getMatWeightAlloc(allocator, st, "bert.embeddings.token_type_embeddings.weight", &allocated_f32, &allocated_f16, &allocated_i8),
            .layer_norm_gamma = try getF32AltAlloc(allocator, st, "bert.embeddings.LayerNorm.weight", "bert.embeddings.LayerNorm.gamma", &allocated_f32),
            .layer_norm_beta = try getF32AltAlloc(allocator, st, "bert.embeddings.LayerNorm.bias", "bert.embeddings.LayerNorm.beta", &allocated_f32),
        } else EmbeddingWeights{
            .word_embeddings = try getMatWeightAlloc(allocator, st, "embeddings.word_embeddings.weight", &allocated_f32, &allocated_f16, &allocated_i8),
            .position_embeddings = try getMatWeightAlloc(allocator, st, "embeddings.position_embeddings.weight", &allocated_f32, &allocated_f16, &allocated_i8),
            .token_type_embeddings = try getMatWeightAlloc(allocator, st, "embeddings.token_type_embeddings.weight", &allocated_f32, &allocated_f16, &allocated_i8),
            .layer_norm_gamma = try getF32AltAlloc(allocator, st, "embeddings.LayerNorm.weight", "embeddings.LayerNorm.gamma", &allocated_f32),
            .layer_norm_beta = try getF32AltAlloc(allocator, st, "embeddings.LayerNorm.bias", "embeddings.LayerNorm.beta", &allocated_f32),
        };

        // Load each transformer layer
        for (0..config.num_hidden_layers) |i| {
            layers[i] = try loadLayerWeightsAlloc(allocator, st, i, has_bert_prefix, &allocated_f32, &allocated_f16, &allocated_i8);
        }

        return .{
            .config = config,
            .embeddings = emb_weights,
            .layers = layers,
            .allocator = allocator,
            .allocated_f32 = allocated_f32,
            .allocated_f16 = allocated_f16,
            .allocated_i8 = allocated_i8,
        };
    }

    pub fn deinit(self: *ModelWeights, allocator: std.mem.Allocator) void {
        _ = allocator; // Use stored allocator
        // Free all allocated weight buffers
        for (self.allocated_f32.items) |w| self.allocator.free(w);
        for (self.allocated_f16.items) |w| self.allocator.free(w);
        for (self.allocated_i8.items) |w| self.allocator.free(w);
        self.allocated_f32.deinit(self.allocator);
        self.allocated_f16.deinit(self.allocator);
        self.allocated_i8.deinit(self.allocator);
        self.allocator.free(self.layers);
    }
};

fn trackAlloc(comptime T: type, allocator: std.mem.Allocator, st: safetensors.SafeTensors, info: safetensors.TensorInfo, data: []const T, allocated: *std.ArrayList([]const T)) !void {
    const byte_data = st.data[st.header_size + 8 + info.data_start ..][0..info.byteSize()];
    if (@intFromPtr(data.ptr) != @intFromPtr(byte_data.ptr)) {
        try allocated.append(allocator, data);
    }
}

fn getF32Alloc(
    allocator: std.mem.Allocator,
    st: safetensors.SafeTensors,
    name: []const u8,
    allocated_f32: *std.ArrayList([]const f32),
) ![]const f32 {
    const info = st.get(name) orelse {
        std.log.err("Missing weight: {s}", .{name});
        return error.MissingWeight;
    };
    if (info.dtype != .F32) {
        std.log.err("Weight {s} has wrong dtype, expected F32", .{name});
        return error.WrongDtype;
    }
    const data = try st.getDataAlloc(f32, info, allocator);
    try trackAlloc(f32, allocator, st, info, data, allocated_f32);
    return data;
}

fn getF16Alloc(
    allocator: std.mem.Allocator,
    st: safetensors.SafeTensors,
    name: []const u8,
    allocated_f16: *std.ArrayList([]const f16),
) ![]const f16 {
    const info = st.get(name) orelse {
        std.log.err("Missing weight: {s}", .{name});
        return error.MissingWeight;
    };
    if (info.dtype != .F16) {
        std.log.err("Weight {s} has wrong dtype, expected F16", .{name});
        return error.WrongDtype;
    }
    const data = try st.getDataAlloc(f16, info, allocator);
    try trackAlloc(f16, allocator, st, info, data, allocated_f16);
    return data;
}

fn getI8Alloc(
    allocator: std.mem.Allocator,
    st: safetensors.SafeTensors,
    name: []const u8,
    allocated_i8: *std.ArrayList([]const i8),
) ![]const i8 {
    const info = st.get(name) orelse {
        std.log.err("Missing weight: {s}", .{name});
        return error.MissingWeight;
    };
    if (info.dtype != .I8) {
        std.log.err("Weight {s} has wrong dtype, expected I8", .{name});
        return error.WrongDtype;
    }
    const data = try st.getDataAlloc(i8, info, allocator);
    try trackAlloc(i8, allocator, st, info, data, allocated_i8);
    return data;
}

fn getF32AltAlloc(
    allocator: std.mem.Allocator,
    st: safetensors.SafeTensors,
    primary: []const u8,
    alternate: []const u8,
    allocated_f32: *std.ArrayList([]const f32),
) ![]const f32 {
    if (st.get(primary)) |_| {
        return getF32Alloc(allocator, st, primary, allocated_f32);
    }
    return getF32Alloc(allocator, st, alternate, allocated_f32);
}

fn getMatWeightAlloc(
    allocator: std.mem.Allocator,
    st: safetensors.SafeTensors,
    name: []const u8,
    allocated_f32: *std.ArrayList([]const f32),
    allocated_f16: *std.ArrayList([]const f16),
    allocated_i8: *std.ArrayList([]const i8),
) !MatWeight {
    const info = st.get(name) orelse {
        std.log.err("Missing weight: {s}", .{name});
        return error.MissingWeight;
    };

    switch (info.dtype) {
        .F32 => return .{ .f32 = try getF32Alloc(allocator, st, name, allocated_f32) },
        .I8 => {
            const data = try getI8Alloc(allocator, st, name, allocated_i8);

            var scale_name_buf: [256]u8 = undefined;
            const scale_name = std.fmt.bufPrint(&scale_name_buf, "{s}.scale", .{name}) catch unreachable;
            const scale = try getF16Alloc(allocator, st, scale_name, allocated_f16);

            return .{ .q8 = .{ .data = data, .scale = scale } };
        },
        else => {
            std.log.err("Weight {s} has unsupported dtype {s}", .{ name, @tagName(info.dtype) });
            return error.UnsupportedDtype;
        },
    }
}

/// Load attention weights for a specific layer (allocating version).
fn loadAttentionWeightsAlloc(
    allocator: std.mem.Allocator,
    st: safetensors.SafeTensors,
    layer_idx: usize,
    has_bert_prefix: bool,
    allocated_f32: *std.ArrayList([]const f32),
    allocated_f16: *std.ArrayList([]const f16),
    allocated_i8: *std.ArrayList([]const i8),
) !AttentionWeights {
    var buf: [128]u8 = undefined;
    var alt_buf: [128]u8 = undefined;

    const base = if (has_bert_prefix) "bert.encoder.layer" else "encoder.layer";

    return .{
        .query_weight = try getLayerMatWeightFmtAlloc(allocator, st, &buf, base, layer_idx, "attention.self.query.weight", allocated_f32, allocated_f16, allocated_i8),
        .query_bias = try getLayerF32FmtAlloc(allocator, st, &buf, base, layer_idx, "attention.self.query.bias", allocated_f32),
        .key_weight = try getLayerMatWeightFmtAlloc(allocator, st, &buf, base, layer_idx, "attention.self.key.weight", allocated_f32, allocated_f16, allocated_i8),
        .key_bias = try getLayerF32FmtAlloc(allocator, st, &buf, base, layer_idx, "attention.self.key.bias", allocated_f32),
        .value_weight = try getLayerMatWeightFmtAlloc(allocator, st, &buf, base, layer_idx, "attention.self.value.weight", allocated_f32, allocated_f16, allocated_i8),
        .value_bias = try getLayerF32FmtAlloc(allocator, st, &buf, base, layer_idx, "attention.self.value.bias", allocated_f32),
        .output_weight = try getLayerMatWeightFmtAlloc(allocator, st, &buf, base, layer_idx, "attention.output.dense.weight", allocated_f32, allocated_f16, allocated_i8),
        .output_bias = try getLayerF32FmtAlloc(allocator, st, &buf, base, layer_idx, "attention.output.dense.bias", allocated_f32),
        .output_ln_gamma = try getLayerF32AltAlloc(allocator, st, &buf, &alt_buf, base, layer_idx, "attention.output.LayerNorm.weight", "attention.output.LayerNorm.gamma", allocated_f32),
        .output_ln_beta = try getLayerF32AltAlloc(allocator, st, &buf, &alt_buf, base, layer_idx, "attention.output.LayerNorm.bias", "attention.output.LayerNorm.beta", allocated_f32),
    };
}

/// Load feed-forward weights for a specific layer (allocating version).
fn loadFeedForwardWeightsAlloc(
    allocator: std.mem.Allocator,
    st: safetensors.SafeTensors,
    layer_idx: usize,
    has_bert_prefix: bool,
    allocated_f32: *std.ArrayList([]const f32),
    allocated_f16: *std.ArrayList([]const f16),
    allocated_i8: *std.ArrayList([]const i8),
) !FeedForwardWeights {
    var buf: [128]u8 = undefined;
    var alt_buf: [128]u8 = undefined;

    const base = if (has_bert_prefix) "bert.encoder.layer" else "encoder.layer";

    return .{
        .intermediate_weight = try getLayerMatWeightFmtAlloc(allocator, st, &buf, base, layer_idx, "intermediate.dense.weight", allocated_f32, allocated_f16, allocated_i8),
        .intermediate_bias = try getLayerF32FmtAlloc(allocator, st, &buf, base, layer_idx, "intermediate.dense.bias", allocated_f32),
        .output_weight = try getLayerMatWeightFmtAlloc(allocator, st, &buf, base, layer_idx, "output.dense.weight", allocated_f32, allocated_f16, allocated_i8),
        .output_bias = try getLayerF32FmtAlloc(allocator, st, &buf, base, layer_idx, "output.dense.bias", allocated_f32),
        .output_ln_gamma = try getLayerF32AltAlloc(allocator, st, &buf, &alt_buf, base, layer_idx, "output.LayerNorm.weight", "output.LayerNorm.gamma", allocated_f32),
        .output_ln_beta = try getLayerF32AltAlloc(allocator, st, &buf, &alt_buf, base, layer_idx, "output.LayerNorm.bias", "output.LayerNorm.beta", allocated_f32),
    };
}

/// Load all weights for a specific layer (allocating version).
fn loadLayerWeightsAlloc(
    allocator: std.mem.Allocator,
    st: safetensors.SafeTensors,
    layer_idx: usize,
    has_bert_prefix: bool,
    allocated_f32: *std.ArrayList([]const f32),
    allocated_f16: *std.ArrayList([]const f16),
    allocated_i8: *std.ArrayList([]const i8),
) !LayerWeights {
    return .{
        .attention = try loadAttentionWeightsAlloc(allocator, st, layer_idx, has_bert_prefix, allocated_f32, allocated_f16, allocated_i8),
        .ffn = try loadFeedForwardWeightsAlloc(allocator, st, layer_idx, has_bert_prefix, allocated_f32, allocated_f16, allocated_i8),
    };
}

fn getLayerF32FmtAlloc(
    allocator: std.mem.Allocator,
    st: safetensors.SafeTensors,
    buf: *[128]u8,
    base: []const u8,
    layer_idx: usize,
    suffix: []const u8,
    allocated_f32: *std.ArrayList([]const f32),
) ![]const f32 {
    const name = std.fmt.bufPrint(buf, "{s}.{d}.{s}", .{ base, layer_idx, suffix }) catch unreachable;
    return getF32Alloc(allocator, st, name, allocated_f32);
}

fn getLayerF32AltAlloc(
    allocator: std.mem.Allocator,
    st: safetensors.SafeTensors,
    buf: *[128]u8,
    alt_buf: *[128]u8,
    base: []const u8,
    layer_idx: usize,
    primary_suffix: []const u8,
    alt_suffix: []const u8,
    allocated_f32: *std.ArrayList([]const f32),
) ![]const f32 {
    const primary = std.fmt.bufPrint(buf, "{s}.{d}.{s}", .{ base, layer_idx, primary_suffix }) catch unreachable;
    const alternate = std.fmt.bufPrint(alt_buf, "{s}.{d}.{s}", .{ base, layer_idx, alt_suffix }) catch unreachable;
    return getF32AltAlloc(allocator, st, primary, alternate, allocated_f32);
}

fn getLayerMatWeightFmtAlloc(
    allocator: std.mem.Allocator,
    st: safetensors.SafeTensors,
    buf: *[128]u8,
    base: []const u8,
    layer_idx: usize,
    suffix: []const u8,
    allocated_f32: *std.ArrayList([]const f32),
    allocated_f16: *std.ArrayList([]const f16),
    allocated_i8: *std.ArrayList([]const i8),
) !MatWeight {
    const name = std.fmt.bufPrint(buf, "{s}.{d}.{s}", .{ base, layer_idx, suffix }) catch unreachable;
    return getMatWeightAlloc(allocator, st, name, allocated_f32, allocated_f16, allocated_i8);
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

    // Per-head attention scratch buffers for optimized matmul
    q_head: []f32, // [seq_len, head_dim] - contiguous Q for one head
    k_head: []f32, // [seq_len, head_dim] - contiguous K for one head
    v_head: []f32, // [seq_len, head_dim] - contiguous V for one head
    attn_out_head: []f32, // [seq_len, head_dim] - contiguous output for one head

    // Additional buffers for full forward pass
    encoder_input: []f32, // Input to encoder
    encoder_output: []f32, // Output from encoder
    layer_input: []f32, // For encoder layer ping-pong (separate from tmp!)

    max_seq_len: usize,

    // Thread pool for parallel execution (optional)
    pool: ?*threading.ThreadPool,

    pub fn init(allocator: std.mem.Allocator, config: Config, max_seq_len: usize) !InferenceContext {
        return initWithPool(allocator, config, max_seq_len, null);
    }

    pub fn initWithPool(allocator: std.mem.Allocator, config: Config, max_seq_len: usize, pool: ?*threading.ThreadPool) !InferenceContext {
        const h = config.hidden_size;
        const inter = config.intermediate_size;
        const heads = config.num_attention_heads;
        const head_dim = h / heads;

        return .{
            .allocator = allocator,
            .config = config,
            .hidden = try allocator.alloc(f32, max_seq_len * h),
            .qkv = try allocator.alloc(f32, 3 * max_seq_len * h),
            .attn_scores = try allocator.alloc(f32, heads * max_seq_len * max_seq_len),
            .attn_out = try allocator.alloc(f32, max_seq_len * h),
            .ffn_intermediate = try allocator.alloc(f32, max_seq_len * inter),
            .tmp = try allocator.alloc(f32, max_seq_len * h),
            .q_head = try allocator.alloc(f32, max_seq_len * head_dim),
            .k_head = try allocator.alloc(f32, max_seq_len * head_dim),
            .v_head = try allocator.alloc(f32, max_seq_len * head_dim),
            .attn_out_head = try allocator.alloc(f32, max_seq_len * head_dim),
            .encoder_input = try allocator.alloc(f32, max_seq_len * h),
            .encoder_output = try allocator.alloc(f32, max_seq_len * h),
            .layer_input = try allocator.alloc(f32, max_seq_len * h),
            .max_seq_len = max_seq_len,
            .pool = pool,
        };
    }

    pub fn deinit(self: *InferenceContext) void {
        self.allocator.free(self.hidden);
        self.allocator.free(self.qkv);
        self.allocator.free(self.attn_scores);
        self.allocator.free(self.attn_out);
        self.allocator.free(self.ffn_intermediate);
        self.allocator.free(self.tmp);
        self.allocator.free(self.q_head);
        self.allocator.free(self.k_head);
        self.allocator.free(self.v_head);
        self.allocator.free(self.attn_out_head);
        self.allocator.free(self.encoder_input);
        self.allocator.free(self.encoder_output);
        self.allocator.free(self.layer_input);
    }
};

fn writeMatWeightRow(dst: []f32, w: MatWeight, row: usize, cols: usize) void {
    switch (w) {
        .f32 => |data| {
            const base = row * cols;
            @memcpy(dst, data[base..][0..cols]);
        },
        .q8 => |q| {
            const base = row * cols;
            const s: f32 = @floatCast(q.scale[row]);

            const vec_len = simd.suggestVectorLength(f32);
            const VecF = @Vector(vec_len, f32);
            const VecI8 = @Vector(vec_len, i8);
            const s_vec: VecF = @splat(s);

            var i: usize = 0;
            while (i + vec_len <= cols) : (i += vec_len) {
                const v_i8: VecI8 = q.data[base + i ..][0..vec_len].*;
                const v_f32: VecF = @as(VecF, @floatFromInt(v_i8)) * s_vec;
                dst[i..][0..vec_len].* = v_f32;
            }
            while (i < cols) : (i += 1) {
                dst[i] = @as(f32, @floatFromInt(q.data[base + i])) * s;
            }
        },
    }
}

fn addMatWeightRow(dst: []f32, w: MatWeight, row: usize, cols: usize) void {
    switch (w) {
        .f32 => |data| {
            const base = row * cols;
            for (0..cols) |i| dst[i] += data[base + i];
        },
        .q8 => |q| {
            const base = row * cols;
            const s: f32 = @floatCast(q.scale[row]);

            const vec_len = simd.suggestVectorLength(f32);
            const VecF = @Vector(vec_len, f32);
            const VecI8 = @Vector(vec_len, i8);
            const s_vec: VecF = @splat(s);

            var i: usize = 0;
            while (i + vec_len <= cols) : (i += vec_len) {
                const v_i8: VecI8 = q.data[base + i ..][0..vec_len].*;
                const add_f32: VecF = @as(VecF, @floatFromInt(v_i8)) * s_vec;
                const cur: VecF = dst[i..][0..vec_len].*;
                dst[i..][0..vec_len].* = cur + add_f32;
            }
            while (i < cols) : (i += 1) {
                dst[i] += @as(f32, @floatFromInt(q.data[base + i])) * s;
            }
        },
    }
}

/// Apply embeddings: word + position + token_type + LayerNorm.
/// Output: [seq_len, hidden_size]
pub fn embeddingsImpl(
    output: []f32,
    token_ids: []const u32,
    weights: EmbeddingWeights,
    config: Config,
    pool: ?*threading.ThreadPool,
) void {
    const seq_len = token_ids.len;
    const h = config.hidden_size;

    for (token_ids, 0..) |tok_id, pos| {
        const out_row = output[pos * h ..][0..h];
        writeMatWeightRow(out_row, weights.word_embeddings, tok_id, h);
        addMatWeightRow(out_row, weights.position_embeddings, pos, h);
        addMatWeightRow(out_row, weights.token_type_embeddings, 0, h);
    }

    // Apply LayerNorm in-place (use parallel version if pool available)
    if (pool) |p| {
        kernels.layernorm.layerNormLastDimParallel(
            f32,
            output[0 .. seq_len * h],
            weights.layer_norm_gamma,
            weights.layer_norm_beta,
            output[0 .. seq_len * h],
            2,
            .{ seq_len, h },
            config.layer_norm_eps,
            p,
        );
    } else {
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
}

/// Apply embeddings: word + position + token_type + LayerNorm (sequential version).
/// Output: [seq_len, hidden_size]
pub fn embeddings(
    output: []f32,
    token_ids: []const u32,
    weights: EmbeddingWeights,
    config: Config,
) void {
    embeddingsImpl(output, token_ids, weights, config, null);
}

/// Linear projection: Y = XW^T + b (BERT weights are stored transposed)
fn linearProjection(
    output: []f32,
    input: []const f32,
    weight: MatWeight, // [out_features, in_features] (transposed)
    bias: []const f32, // [out_features]
    seq_len: usize,
    in_features: usize,
    out_features: usize,
) void {
    switch (weight) {
        .f32 => |w| {
            // Matmul: [seq_len, in_features] @ [out_features, in_features]^T = [seq_len, out_features]
            // Weight is stored as [out_features, in_features], so we use transpose_b
            kernels.matmul.matmulTransposeB(f32, input, w, output, seq_len, in_features, out_features);

            // Add bias
            for (0..seq_len) |s| {
                const start = s * out_features;
                for (0..out_features) |i| {
                    output[start + i] += bias[i];
                }
            }
        },
        .q8 => |q| {
            kernels.matmul_w8a32.matmulTransposeB_f32_i8_rowScale_bias(
                input,
                q.data,
                q.scale,
                bias,
                output,
                seq_len,
                in_features,
                out_features,
            );
        },
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

        // Extract Q_h, K_h, V_h to contiguous buffers for matmul
        for (0..seq_len) |s| {
            @memcpy(ctx.q_head[s * head_dim ..][0..head_dim], q[s * h + head_offset ..][0..head_dim]);
            @memcpy(ctx.k_head[s * head_dim ..][0..head_dim], k[s * h + head_offset ..][0..head_dim]);
            @memcpy(ctx.v_head[s * head_dim ..][0..head_dim], v[s * h + head_offset ..][0..head_dim]);
        }

        // Compute attention scores: Q_h @ K_h^T using matmul
        // [seq_len, head_dim] @ [seq_len, head_dim]^T = [seq_len, seq_len]
        const scores = ctx.attn_scores[scores_offset..][0 .. seq_len * seq_len];
        kernels.matmul.matmulTransposeB(f32, ctx.q_head[0 .. seq_len * head_dim], ctx.k_head[0 .. seq_len * head_dim], scores, seq_len, head_dim, seq_len);

        // Scale scores
        for (scores) |*s| {
            s.* *= scale;
        }

        // Softmax over last dimension
        for (0..seq_len) |qi| {
            const row = scores[qi * seq_len ..][0..seq_len];

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

        // Apply attention: scores @ V_h using matmul
        // [seq_len, seq_len] @ [seq_len, head_dim] = [seq_len, head_dim]
        kernels.matmul.matmulTiled(f32, scores, ctx.v_head[0 .. seq_len * head_dim], ctx.attn_out_head[0 .. seq_len * head_dim], seq_len, seq_len, head_dim);

        // Copy attn_out_head back to interleaved attn_out
        for (0..seq_len) |s| {
            @memcpy(ctx.attn_out[s * h + head_offset ..][0..head_dim], ctx.attn_out_head[s * head_dim ..][0..head_dim]);
        }
    }

    // Output projection
    linearProjection(ctx.tmp[0 .. seq_len * h], ctx.attn_out[0 .. seq_len * h], attn.output_weight, attn.output_bias, seq_len, h, h);

    // Residual connection + LayerNorm
    for (0..seq_len * h) |i| {
        output[i] = input[i] + ctx.tmp[i];
    }

    if (ctx.pool) |p| {
        kernels.layernorm.layerNormLastDimParallel(
            f32,
            output[0 .. seq_len * h],
            attn.output_ln_gamma,
            attn.output_ln_beta,
            output[0 .. seq_len * h],
            2,
            .{ seq_len, h },
            ctx.config.layer_norm_eps,
            p,
        );
    } else {
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

    if (ctx.pool) |p| {
        kernels.layernorm.layerNormLastDimParallel(
            f32,
            output[0 .. seq_len * h],
            ffn.output_ln_gamma,
            ffn.output_ln_beta,
            output[0 .. seq_len * h],
            2,
            .{ seq_len, h },
            ctx.config.layer_norm_eps,
            p,
        );
    } else {
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

    // Step 1: Embeddings (uses parallel layernorm if pool available)
    embeddingsImpl(hidden_states, token_ids, weights.embeddings, ctx.config, ctx.pool);

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

/// Batch inference context pool - holds multiple contexts for parallel batch processing.
pub const BatchContextPool = struct {
    contexts: []InferenceContext,
    allocator: std.mem.Allocator,

    pub fn init(
        allocator: std.mem.Allocator,
        config: Config,
        max_seq_len: usize,
        num_contexts: usize,
        pool: ?*threading.ThreadPool,
    ) !BatchContextPool {
        const contexts = try allocator.alloc(InferenceContext, num_contexts);
        errdefer allocator.free(contexts);

        for (contexts, 0..) |*ctx, i| {
            ctx.* = try InferenceContext.initWithPool(allocator, config, max_seq_len, pool);
            errdefer {
                for (contexts[0..i]) |*c| c.deinit();
            }
        }

        return .{
            .contexts = contexts,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *BatchContextPool) void {
        for (self.contexts) |*ctx| {
            ctx.deinit();
        }
        self.allocator.free(self.contexts);
    }
};

/// Parallel batch inference: process multiple sequences across threads.
/// Each thread uses its own InferenceContext from the pool.
/// Requires a BatchContextPool with at least as many contexts as threads.
pub fn forwardBatchParallel(
    outputs: []f32, // [batch_size * hidden_size]
    token_ids_batch: []const []const u32,
    weights: ModelWeights,
    ctx_pool: *BatchContextPool,
    pool: *threading.ThreadPool,
) void {
    const batch_size = token_ids_batch.len;
    if (batch_size == 0) return;

    const h = weights.config.hidden_size;

    // Atomic counter to assign unique context indices to each work batch
    var ctx_counter = std.atomic.Value(u32).init(0);

    const Context = struct {
        outputs: []f32,
        token_ids_batch: []const []const u32,
        weights: ModelWeights,
        ctx_pool: *BatchContextPool,
        ctx_counter: *std.atomic.Value(u32),
        h: usize,
    };

    const ctx = Context{
        .outputs = outputs,
        .token_ids_batch = token_ids_batch,
        .weights = weights,
        .ctx_pool = ctx_pool,
        .ctx_counter = &ctx_counter,
        .h = h,
    };

    pool.parallelForBatch(batch_size, ctx, struct {
        fn work(c: Context, start: usize, end: usize) void {
            // Get unique context for this work batch
            const ctx_idx = c.ctx_counter.fetchAdd(1, .monotonic) % c.ctx_pool.contexts.len;
            const local_ctx = &c.ctx_pool.contexts[ctx_idx];

            // Disable inner parallelism - we're already parallel at batch level
            const saved_pool = local_ctx.pool;
            local_ctx.pool = null;
            defer local_ctx.pool = saved_pool;

            for (start..end) |batch_idx| {
                forward(
                    c.outputs[batch_idx * c.h ..][0..c.h],
                    c.token_ids_batch[batch_idx],
                    c.weights,
                    local_ctx,
                );
            }
        }
    }.work);
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
