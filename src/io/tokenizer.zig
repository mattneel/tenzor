//! WordPiece tokenizer for BERT-style models.
//!
//! Implements the WordPiece algorithm used by BERT/arctic-embed models:
//! 1. Basic tokenization (lowercase, split on whitespace/punctuation)
//! 2. WordPiece subword tokenization using vocab lookup

const std = @import("std");

pub const Tokenizer = struct {
    allocator: std.mem.Allocator,
    vocab: std.StringHashMap(u32),
    unk_id: u32 = 100,
    cls_id: u32 = 101,
    sep_id: u32 = 102,
    max_word_chars: usize = 200,

    pub fn init(allocator: std.mem.Allocator) Tokenizer {
        return .{
            .allocator = allocator,
            .vocab = std.StringHashMap(u32).init(allocator),
        };
    }

    pub fn deinit(self: *Tokenizer) void {
        // Free all the allocated keys
        var it = self.vocab.keyIterator();
        while (it.next()) |key| {
            self.allocator.free(key.*);
        }
        self.vocab.deinit();
    }

    /// Load vocabulary from vocab.txt file (one token per line).
    pub fn loadVocab(self: *Tokenizer, path: []const u8) !void {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        const stat = try file.stat();
        const data = try self.allocator.alloc(u8, stat.size);
        defer self.allocator.free(data);

        var total_read: usize = 0;
        while (total_read < stat.size) {
            const n = try file.read(data[total_read..]);
            if (n == 0) return error.UnexpectedEOF;
            total_read += n;
        }

        try self.loadVocabFromBytes(data);
    }

    /// Load vocabulary from bytes (one token per line).
    pub fn loadVocabFromBytes(self: *Tokenizer, data: []const u8) !void {
        var line_idx: u32 = 0;
        var lines = std.mem.splitScalar(u8, data, '\n');
        while (lines.next()) |line| {
            if (line.len == 0) {
                line_idx += 1;
                continue;
            }
            // Strip carriage return if present
            const token = if (line.len > 0 and line[line.len - 1] == '\r')
                line[0 .. line.len - 1]
            else
                line;

            if (token.len > 0) {
                const owned = try self.allocator.dupe(u8, token);
                try self.vocab.put(owned, line_idx);
            }
            line_idx += 1;
        }
    }

    /// Tokenize text into token IDs.
    /// Returns slice with [CLS] ... tokens ... [SEP]
    pub fn encode(
        self: *const Tokenizer,
        text: []const u8,
        output: []u32,
    ) !usize {
        std.debug.assert(output.len >= 2); // Need room for CLS and SEP

        var pos: usize = 0;
        output[pos] = self.cls_id;
        pos += 1;

        // Basic tokenization: split on whitespace and punctuation
        var i: usize = 0;
        while (i < text.len and pos < output.len - 1) {
            // Skip whitespace
            while (i < text.len and isWhitespace(text[i])) : (i += 1) {}
            if (i >= text.len) break;

            // Find word boundary
            const word_start = i;
            while (i < text.len and !isWhitespace(text[i]) and !isPunctuation(text[i])) : (i += 1) {}

            if (i > word_start) {
                // Tokenize word
                pos = try self.tokenizeWord(text[word_start..i], output, pos);
            }

            // Handle punctuation as separate token
            if (i < text.len and isPunctuation(text[i])) {
                if (pos < output.len - 1) {
                    const punct = text[i .. i + 1];
                    output[pos] = self.vocab.get(punct) orelse self.unk_id;
                    pos += 1;
                }
                i += 1;
            }
        }

        output[pos] = self.sep_id;
        pos += 1;

        return pos;
    }

    /// Tokenize text into token IDs WITHOUT [CLS]/[SEP] markers.
    /// Useful for chunking long documents where markers are added per-chunk.
    pub fn encodeRaw(
        self: *const Tokenizer,
        text: []const u8,
        output: *std.ArrayList(u32),
        ally: std.mem.Allocator,
    ) !void {
        var i: usize = 0;
        while (i < text.len) {
            // Skip whitespace
            while (i < text.len and isWhitespace(text[i])) : (i += 1) {}
            if (i >= text.len) break;

            // Find word boundary
            const word_start = i;
            while (i < text.len and !isWhitespace(text[i]) and !isPunctuation(text[i])) : (i += 1) {}

            if (i > word_start) {
                // Tokenize word into ArrayList
                try self.tokenizeWordAppend(text[word_start..i], output, ally);
            }

            // Handle punctuation as separate token
            if (i < text.len and isPunctuation(text[i])) {
                const punct = text[i .. i + 1];
                try output.append(ally, self.vocab.get(punct) orelse self.unk_id);
                i += 1;
            }
        }
    }

    /// WordPiece tokenization appending to ArrayList.
    fn tokenizeWordAppend(
        self: *const Tokenizer,
        word: []const u8,
        output: *std.ArrayList(u32),
        ally: std.mem.Allocator,
    ) !void {
        if (word.len > self.max_word_chars) {
            try output.append(ally, self.unk_id);
            return;
        }

        // Lowercase buffer
        var lower_buf: [256]u8 = undefined;
        const lower = toLower(word, &lower_buf);

        const start_len = output.items.len;
        var char_start: usize = 0;
        var is_first = true;

        while (char_start < lower.len) {
            var char_end = lower.len;
            var found = false;

            // Try to find longest matching subword
            while (char_end > char_start) {
                var subword_buf: [256]u8 = undefined;
                const subword = if (is_first)
                    lower[char_start..char_end]
                else blk: {
                    subword_buf[0] = '#';
                    subword_buf[1] = '#';
                    @memcpy(subword_buf[2 .. 2 + (char_end - char_start)], lower[char_start..char_end]);
                    break :blk subword_buf[0 .. 2 + (char_end - char_start)];
                };

                if (self.vocab.get(subword)) |token_id| {
                    try output.append(ally, token_id);
                    char_start = char_end;
                    is_first = false;
                    found = true;
                    break;
                }

                char_end -= 1;
            }

            if (!found) {
                // No subword found, use [UNK] for entire word
                output.shrinkRetainingCapacity(start_len);
                try output.append(ally, self.unk_id);
                return;
            }
        }
    }

    /// WordPiece tokenization for a single word.
    fn tokenizeWord(
        self: *const Tokenizer,
        word: []const u8,
        output: []u32,
        start_pos: usize,
    ) !usize {
        if (word.len > self.max_word_chars) {
            output[start_pos] = self.unk_id;
            return start_pos + 1;
        }

        // Lowercase buffer
        var lower_buf: [256]u8 = undefined;
        const lower = toLower(word, &lower_buf);

        var pos = start_pos;
        var char_start: usize = 0;
        var is_first = true;

        while (char_start < lower.len and pos < output.len - 1) {
            var char_end = lower.len;
            var found = false;

            // Try to find longest matching subword
            while (char_end > char_start) {
                var subword_buf: [256]u8 = undefined;
                const subword = if (is_first)
                    lower[char_start..char_end]
                else blk: {
                    subword_buf[0] = '#';
                    subword_buf[1] = '#';
                    @memcpy(subword_buf[2 .. 2 + (char_end - char_start)], lower[char_start..char_end]);
                    break :blk subword_buf[0 .. 2 + (char_end - char_start)];
                };

                if (self.vocab.get(subword)) |token_id| {
                    output[pos] = token_id;
                    pos += 1;
                    char_start = char_end;
                    is_first = false;
                    found = true;
                    break;
                }

                char_end -= 1;
            }

            if (!found) {
                // No subword found, use [UNK] for entire word
                output[start_pos] = self.unk_id;
                return start_pos + 1;
            }
        }

        return pos;
    }
};

fn isWhitespace(c: u8) bool {
    return c == ' ' or c == '\t' or c == '\n' or c == '\r';
}

fn isPunctuation(c: u8) bool {
    return switch (c) {
        '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~' => true,
        else => false,
    };
}

fn toLower(input: []const u8, buf: []u8) []const u8 {
    const len = @min(input.len, buf.len);
    for (input[0..len], buf[0..len]) |c, *out| {
        out.* = if (c >= 'A' and c <= 'Z') c + 32 else c;
    }
    return buf[0..len];
}

// ============================================================================
// Tests
// ============================================================================

test "tokenizer basic" {
    const allocator = std.testing.allocator;

    var tokenizer = Tokenizer.init(allocator);
    defer tokenizer.deinit();

    // Build a mini vocab
    const vocab_entries = [_]struct { []const u8, u32 }{
        .{ "[PAD]", 0 },
        .{ "[UNK]", 100 },
        .{ "[CLS]", 101 },
        .{ "[SEP]", 102 },
        .{ "hello", 7592 },
        .{ "world", 2088 },
        .{ "##ing", 2075 },
        .{ "test", 3231 },
        .{ "##s", 1055 },
    };

    for (vocab_entries) |entry| {
        const key = try allocator.dupe(u8, entry[0]);
        try tokenizer.vocab.put(key, entry[1]);
    }

    var output: [32]u32 = undefined;
    const len = try tokenizer.encode("hello world", &output);

    try std.testing.expectEqual(@as(u32, 101), output[0]); // [CLS]
    try std.testing.expectEqual(@as(u32, 7592), output[1]); // hello
    try std.testing.expectEqual(@as(u32, 2088), output[2]); // world
    try std.testing.expectEqual(@as(u32, 102), output[3]); // [SEP]
    try std.testing.expectEqual(@as(usize, 4), len);
}

test "tokenizer wordpiece split" {
    const allocator = std.testing.allocator;

    var tokenizer = Tokenizer.init(allocator);
    defer tokenizer.deinit();

    // Vocab where "testing" splits to "test" + "##ing"
    const vocab_entries = [_]struct { []const u8, u32 }{
        .{ "[UNK]", 100 },
        .{ "[CLS]", 101 },
        .{ "[SEP]", 102 },
        .{ "test", 3231 },
        .{ "##ing", 2075 },
    };

    for (vocab_entries) |entry| {
        const key = try allocator.dupe(u8, entry[0]);
        try tokenizer.vocab.put(key, entry[1]);
    }

    var output: [32]u32 = undefined;
    const len = try tokenizer.encode("testing", &output);

    try std.testing.expectEqual(@as(u32, 101), output[0]); // [CLS]
    try std.testing.expectEqual(@as(u32, 3231), output[1]); // test
    try std.testing.expectEqual(@as(u32, 2075), output[2]); // ##ing
    try std.testing.expectEqual(@as(u32, 102), output[3]); // [SEP]
    try std.testing.expectEqual(@as(usize, 4), len);
}

test "tokenizer unknown word" {
    const allocator = std.testing.allocator;

    var tokenizer = Tokenizer.init(allocator);
    defer tokenizer.deinit();

    // Minimal vocab - xyz won't be found
    const vocab_entries = [_]struct { []const u8, u32 }{
        .{ "[UNK]", 100 },
        .{ "[CLS]", 101 },
        .{ "[SEP]", 102 },
        .{ "hello", 7592 },
    };

    for (vocab_entries) |entry| {
        const key = try allocator.dupe(u8, entry[0]);
        try tokenizer.vocab.put(key, entry[1]);
    }

    var output: [32]u32 = undefined;
    const len = try tokenizer.encode("xyz", &output);

    try std.testing.expectEqual(@as(u32, 101), output[0]); // [CLS]
    try std.testing.expectEqual(@as(u32, 100), output[1]); // [UNK]
    try std.testing.expectEqual(@as(u32, 102), output[2]); // [SEP]
    try std.testing.expectEqual(@as(usize, 3), len);
}

test "tokenizer punctuation" {
    const allocator = std.testing.allocator;

    var tokenizer = Tokenizer.init(allocator);
    defer tokenizer.deinit();

    const vocab_entries = [_]struct { []const u8, u32 }{
        .{ "[UNK]", 100 },
        .{ "[CLS]", 101 },
        .{ "[SEP]", 102 },
        .{ "hello", 7592 },
        .{ ",", 1010 },
        .{ "world", 2088 },
    };

    for (vocab_entries) |entry| {
        const key = try allocator.dupe(u8, entry[0]);
        try tokenizer.vocab.put(key, entry[1]);
    }

    var output: [32]u32 = undefined;
    const len = try tokenizer.encode("hello, world", &output);

    try std.testing.expectEqual(@as(u32, 101), output[0]); // [CLS]
    try std.testing.expectEqual(@as(u32, 7592), output[1]); // hello
    try std.testing.expectEqual(@as(u32, 1010), output[2]); // ,
    try std.testing.expectEqual(@as(u32, 2088), output[3]); // world
    try std.testing.expectEqual(@as(u32, 102), output[4]); // [SEP]
    try std.testing.expectEqual(@as(usize, 5), len);
}

test "tokenizer lowercase" {
    const allocator = std.testing.allocator;

    var tokenizer = Tokenizer.init(allocator);
    defer tokenizer.deinit();

    const vocab_entries = [_]struct { []const u8, u32 }{
        .{ "[UNK]", 100 },
        .{ "[CLS]", 101 },
        .{ "[SEP]", 102 },
        .{ "hello", 7592 },
    };

    for (vocab_entries) |entry| {
        const key = try allocator.dupe(u8, entry[0]);
        try tokenizer.vocab.put(key, entry[1]);
    }

    var output: [32]u32 = undefined;
    const len = try tokenizer.encode("HELLO", &output);

    try std.testing.expectEqual(@as(u32, 101), output[0]); // [CLS]
    try std.testing.expectEqual(@as(u32, 7592), output[1]); // hello (lowercased)
    try std.testing.expectEqual(@as(u32, 102), output[2]); // [SEP]
    try std.testing.expectEqual(@as(usize, 3), len);
}
