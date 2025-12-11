//! Parity tests for WordPiece tokenizer against HuggingFace reference.
//!
//! Run: python scripts/parity/tokenizer.py
//! Then: zig build test

const std = @import("std");
const Tokenizer = @import("../io/tokenizer.zig").Tokenizer;

// Test cases from HuggingFace tokenizer (scripts/parity/tokenizer.py)
const TestCase = struct {
    text: []const u8,
    expected_ids: []const u32,
};

const test_cases = [_]TestCase{
    .{ .text = "Hello world", .expected_ids = &[_]u32{ 101, 7592, 2088, 102 } },
    .{ .text = "hello world", .expected_ids = &[_]u32{ 101, 7592, 2088, 102 } },
    .{ .text = "HELLO WORLD", .expected_ids = &[_]u32{ 101, 7592, 2088, 102 } },
    .{ .text = "Hello, world!", .expected_ids = &[_]u32{ 101, 7592, 1010, 2088, 999, 102 } },
    .{ .text = "The quick brown fox jumps over the lazy dog.", .expected_ids = &[_]u32{ 101, 1996, 4248, 2829, 4419, 14523, 2058, 1996, 13971, 3899, 1012, 102 } },
    .{ .text = "What is machine learning?", .expected_ids = &[_]u32{ 101, 2054, 2003, 3698, 4083, 1029, 102 } },
    .{ .text = "testing", .expected_ids = &[_]u32{ 101, 5604, 102 } },
    .{ .text = "embeddings", .expected_ids = &[_]u32{ 101, 7861, 8270, 4667, 2015, 102 } },
    .{ .text = "I love programming in Zig!", .expected_ids = &[_]u32{ 101, 1045, 2293, 4730, 1999, 1062, 8004, 999, 102 } },
    .{ .text = "", .expected_ids = &[_]u32{ 101, 102 } },
    .{ .text = "a", .expected_ids = &[_]u32{ 101, 1037, 102 } },
    .{ .text = "123", .expected_ids = &[_]u32{ 101, 13138, 102 } },
    .{ .text = "hello-world", .expected_ids = &[_]u32{ 101, 7592, 1011, 2088, 102 } },
    .{ .text = "don't", .expected_ids = &[_]u32{ 101, 2123, 1005, 1056, 102 } },
    .{ .text = "   spaces   ", .expected_ids = &[_]u32{ 101, 7258, 102 } },
};

test "tokenizer parity with HuggingFace" {
    const allocator = std.testing.allocator;

    var tokenizer = Tokenizer.init(allocator);
    defer tokenizer.deinit();

    // Load vocab
    tokenizer.loadVocab("models/arctic-embed-xs/vocab.txt") catch |err| {
        std.debug.print("Skipping parity test: vocab.txt not found ({s})\n", .{@errorName(err)});
        return;
    };

    var output: [512]u32 = undefined;

    for (test_cases) |tc| {
        const len = try tokenizer.encode(tc.text, &output);

        if (len != tc.expected_ids.len) {
            std.debug.print("\nText: \"{s}\"\n", .{tc.text});
            std.debug.print("Expected len: {d}, got: {d}\n", .{ tc.expected_ids.len, len });
            std.debug.print("Expected: ", .{});
            for (tc.expected_ids) |id| std.debug.print("{d} ", .{id});
            std.debug.print("\nGot:      ", .{});
            for (output[0..len]) |id| std.debug.print("{d} ", .{id});
            std.debug.print("\n", .{});
        }
        try std.testing.expectEqual(tc.expected_ids.len, len);

        for (tc.expected_ids, output[0..len]) |expected, actual| {
            if (expected != actual) {
                std.debug.print("\nText: \"{s}\"\n", .{tc.text});
                std.debug.print("Expected: ", .{});
                for (tc.expected_ids) |id| std.debug.print("{d} ", .{id});
                std.debug.print("\nGot:      ", .{});
                for (output[0..len]) |id| std.debug.print("{d} ", .{id});
                std.debug.print("\n", .{});
            }
            try std.testing.expectEqual(expected, actual);
        }
    }
}
