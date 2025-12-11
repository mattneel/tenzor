//! Tenzor benchmark suite entry point.
//!
//! Run with: zig build bench
//! Filter:   zig build bench -- --filter matmul

const std = @import("std");
const ziterion = @import("ziterion").ziterion;

// Import benchmark modules
const matmul_bench = @import("kernels/matmul.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    try ziterion.run(matmul_bench.benchmarks, .{
        .allocator = gpa.allocator(),
    });
}
