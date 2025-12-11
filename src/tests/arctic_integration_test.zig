//! Integration test for arctic-embed-xs model.
//!
//! Loads the model from safetensors, runs inference,
//! and compares against reference embeddings from HuggingFace.

const std = @import("std");
const arctic = @import("../model/arctic.zig");
const safetensors = @import("../io/safetensors.zig");

const fixtures_dir = "test_fixtures";

/// Read file completely in a loop (Zig 0.16 compatible).
fn readFileAll(file: std.fs.File, buffer: []u8) !void {
    var total_read: usize = 0;
    while (total_read < buffer.len) {
        const n = try file.read(buffer[total_read..]);
        if (n == 0) return error.UnexpectedEOF;
        total_read += n;
    }
}

/// Load token IDs from binary file.
fn loadTokens(allocator: std.mem.Allocator, path: []const u8) ![]u32 {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    const stat = try file.stat();
    const num_tokens = stat.size / 4;

    const tokens = try allocator.alloc(u32, num_tokens);
    errdefer allocator.free(tokens);

    const bytes = std.mem.sliceAsBytes(tokens);
    try readFileAll(file, bytes);

    return tokens;
}

/// Load expected embedding from binary file.
fn loadEmbedding(allocator: std.mem.Allocator, path: []const u8) ![]f32 {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    const stat = try file.stat();
    const num_floats = stat.size / 4;

    const embedding = try allocator.alloc(f32, num_floats);
    errdefer allocator.free(embedding);

    const bytes = std.mem.sliceAsBytes(embedding);
    try readFileAll(file, bytes);

    return embedding;
}

/// Compute cosine similarity between two vectors.
fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
    var dot: f32 = 0;
    var norm_a: f32 = 0;
    var norm_b: f32 = 0;

    for (a, b) |va, vb| {
        dot += va * vb;
        norm_a += va * va;
        norm_b += vb * vb;
    }

    const norm = @sqrt(norm_a) * @sqrt(norm_b);
    return if (norm > 1e-12) dot / norm else 0;
}

test "arctic embeddings match HuggingFace" {
    const allocator = std.testing.allocator;

    // Load model weights
    const model_path = fixtures_dir ++ "/model.safetensors";
    const load_result = safetensors.load(allocator, model_path) catch {
        return error.SkipZigTest;
    };
    var st = load_result.st;
    const data = load_result.data;
    defer st.deinit();
    defer allocator.free(data);

    // Load weights into model struct
    const config = arctic.arctic_embed_xs_config;
    var weights = try arctic.ModelWeights.fromSafeTensors(allocator, st, config);
    defer weights.deinit(allocator);

    // Load test tokens
    const tokens = try loadTokens(allocator, fixtures_dir ++ "/test_0_tokens.bin");
    defer allocator.free(tokens);

    // Load expected embeddings output
    const expected_emb = try loadEmbedding(allocator, fixtures_dir ++ "/test_0_embeddings_output.bin");
    defer allocator.free(expected_emb);

    // Allocate output
    const seq_len = tokens.len;
    const emb_output = try allocator.alloc(f32, seq_len * config.hidden_size);
    defer allocator.free(emb_output);

    // Run embeddings
    arctic.embeddings(emb_output, tokens, weights.embeddings, config);

    // Verify cosine similarity
    const cosine_sim = cosineSimilarity(emb_output, expected_emb);
    try std.testing.expect(cosine_sim > 0.99);
}

test "arctic layer 0 matches HuggingFace" {
    const allocator = std.testing.allocator;

    // Load model weights
    const model_path = fixtures_dir ++ "/model.safetensors";
    const load_result = safetensors.load(allocator, model_path) catch {
        return error.SkipZigTest;
    };
    var st = load_result.st;
    const data = load_result.data;
    defer st.deinit();
    defer allocator.free(data);

    // Load weights
    const config = arctic.arctic_embed_xs_config;
    var weights = try arctic.ModelWeights.fromSafeTensors(allocator, st, config);
    defer weights.deinit(allocator);

    // Load expected embeddings output (input to layer 0)
    const emb_output = try loadEmbedding(allocator, fixtures_dir ++ "/test_0_embeddings_output.bin");
    defer allocator.free(emb_output);

    // Load expected layer 0 output
    const expected_layer0 = try loadEmbedding(allocator, fixtures_dir ++ "/test_0_layer0_output.bin");
    defer allocator.free(expected_layer0);

    const seq_len: usize = 4;
    const h = config.hidden_size;

    // Initialize context
    var ctx = try arctic.InferenceContext.init(allocator, config, seq_len);
    defer ctx.deinit();

    // Run single transformer block
    const layer0_output = try allocator.alloc(f32, seq_len * h);
    defer allocator.free(layer0_output);

    arctic.transformerBlock(layer0_output, emb_output, weights.layers[0], &ctx, seq_len);

    // Verify cosine similarity
    const cosine_sim = cosineSimilarity(layer0_output, expected_layer0);
    try std.testing.expect(cosine_sim > 0.99);
}

test "arctic model inference matches HuggingFace reference" {
    const allocator = std.testing.allocator;

    // Load model weights
    const model_path = fixtures_dir ++ "/model.safetensors";
    const load_result = safetensors.load(allocator, model_path) catch {
        return error.SkipZigTest;
    };
    var st = load_result.st;
    const data = load_result.data;
    defer st.deinit();
    defer allocator.free(data);

    // Load weights into model struct
    const config = arctic.arctic_embed_xs_config;
    var weights = try arctic.ModelWeights.fromSafeTensors(allocator, st, config);
    defer weights.deinit(allocator);

    // Initialize inference context
    var ctx = try arctic.InferenceContext.init(allocator, config, 128);
    defer ctx.deinit();

    // Output buffer
    const output = try allocator.alloc(f32, config.hidden_size);
    defer allocator.free(output);

    // Test each fixture
    const test_cases = [_]struct { name: []const u8 }{
        .{ .name = "test_0" },
        .{ .name = "test_1" },
        .{ .name = "test_2" },
    };

    for (test_cases) |tc| {
        // Load test data
        var tokens_path_buf: [256]u8 = undefined;
        var embedding_path_buf: [256]u8 = undefined;

        const tokens_path = std.fmt.bufPrint(&tokens_path_buf, fixtures_dir ++ "/{s}_tokens.bin", .{tc.name}) catch unreachable;
        const embedding_path = std.fmt.bufPrint(&embedding_path_buf, fixtures_dir ++ "/{s}_embedding.bin", .{tc.name}) catch unreachable;

        const tokens = try loadTokens(allocator, tokens_path);
        defer allocator.free(tokens);

        const expected = try loadEmbedding(allocator, embedding_path);
        defer allocator.free(expected);

        // Run inference
        arctic.forward(output, tokens, weights, &ctx);

        // Verify cosine similarity (should be > 0.99 for correct implementation)
        const cosine_sim = cosineSimilarity(output, expected);
        try std.testing.expect(cosine_sim > 0.99);
    }
}
