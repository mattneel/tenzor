//! Tenzor benchmark suite entry point.
//!
//! Run with: zig build bench

const std = @import("std");
const ziterion = @import("ziterion").ziterion;

// Kernel benchmarks
const matmul_bench = @import("kernels/matmul.zig");
const conv2d_bench = @import("kernels/conv2d.zig");
const elementwise_bench = @import("kernels/elementwise.zig");
const reduce_bench = @import("kernels/reduce.zig");
const softmax_bench = @import("kernels/softmax.zig");
const layernorm_bench = @import("kernels/layernorm.zig");

// Model benchmarks
const lenet_bench = @import("models/lenet.zig");

// Scaling benchmarks
const thread_scaling_bench = @import("scaling/thread_scaling.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Kernel benchmarks
    std.debug.print("\n=== Matrix Multiplication ===\n\n", .{});
    try ziterion.run(matmul_bench.benchmarks, .{ .allocator = allocator });

    std.debug.print("\n=== Convolution 2D ===\n\n", .{});
    try ziterion.run(conv2d_bench.benchmarks, .{ .allocator = allocator });

    std.debug.print("\n=== Elementwise ===\n\n", .{});
    try ziterion.run(elementwise_bench.benchmarks, .{ .allocator = allocator });

    std.debug.print("\n=== Reductions ===\n\n", .{});
    try ziterion.run(reduce_bench.benchmarks, .{ .allocator = allocator });

    std.debug.print("\n=== Softmax ===\n\n", .{});
    try ziterion.run(softmax_bench.benchmarks, .{ .allocator = allocator });

    std.debug.print("\n=== Layer Normalization ===\n\n", .{});
    try ziterion.run(layernorm_bench.benchmarks, .{ .allocator = allocator });

    // Model benchmarks
    std.debug.print("\n=== LeNet-5 Model ===\n\n", .{});
    try ziterion.run(lenet_bench.benchmarks, .{ .allocator = allocator });

    // Scaling benchmarks
    std.debug.print("\n=== Thread Scaling (512x512x512 matmul) ===\n\n", .{});
    try ziterion.run(thread_scaling_bench.benchmarks, .{ .allocator = allocator });
}
