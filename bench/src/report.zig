//! Benchmark HTML report generator.
//!
//! Runs all benchmarks and generates a self-contained HTML report.
//! Usage: zig build bench-report

const std = @import("std");
const ziterion = @import("ziterion").ziterion;
const html_report = @import("ziterion").html_report;

// Import all benchmark modules
const matmul_bench = @import("kernels/matmul.zig");
const conv2d_bench = @import("kernels/conv2d.zig");
const elementwise_bench = @import("kernels/elementwise.zig");
const reduce_bench = @import("kernels/reduce.zig");
const softmax_bench = @import("kernels/softmax.zig");
const layernorm_bench = @import("kernels/layernorm.zig");
const lenet_bench = @import("models/lenet.zig");
const thread_scaling_bench = @import("scaling/thread_scaling.zig");

/// All benchmarks as comptime tuples for inline iteration.
const all_benchmarks = .{
    // Matrix Multiplication
    .{ "Matrix Multiplication", "matmul_naive_64x64x64", matmul_bench.benchmarks.matmul_naive_64x64x64 },
    .{ "Matrix Multiplication", "matmul_tiled_64x64x64", matmul_bench.benchmarks.matmul_tiled_64x64x64 },
    .{ "Matrix Multiplication", "matmul_naive_256x256x256", matmul_bench.benchmarks.matmul_naive_256x256x256 },
    .{ "Matrix Multiplication", "matmul_tiled_256x256x256", matmul_bench.benchmarks.matmul_tiled_256x256x256 },
    .{ "Matrix Multiplication", "matmul_parallel_256x256x256", matmul_bench.benchmarks.matmul_parallel_256x256x256 },
    .{ "Matrix Multiplication", "matmul_tiled_512x512x512", matmul_bench.benchmarks.matmul_tiled_512x512x512 },
    .{ "Matrix Multiplication", "matmul_parallel_512x512x512", matmul_bench.benchmarks.matmul_parallel_512x512x512 },
    .{ "Matrix Multiplication", "matmul_tiled_384x512x384", matmul_bench.benchmarks.matmul_tiled_384x512x384 },
    .{ "Matrix Multiplication", "matmul_parallel_384x512x384", matmul_bench.benchmarks.matmul_parallel_384x512x384 },

    // Convolution 2D
    .{ "Convolution 2D", "conv2d_lenet1_b1", conv2d_bench.benchmarks.conv2d_lenet1_b1 },
    .{ "Convolution 2D", "conv2d_lenet1_b64", conv2d_bench.benchmarks.conv2d_lenet1_b64 },
    .{ "Convolution 2D", "conv2d_lenet2_b1", conv2d_bench.benchmarks.conv2d_lenet2_b1 },
    .{ "Convolution 2D", "conv2d_lenet2_b64", conv2d_bench.benchmarks.conv2d_lenet2_b64 },
    .{ "Convolution 2D", "conv2d_64x56x56_3x3", conv2d_bench.benchmarks.conv2d_64x56x56_3x3 },

    // Elementwise
    .{ "Elementwise", "relu_64K", elementwise_bench.benchmarks.relu_64K },
    .{ "Elementwise", "exp_64K", elementwise_bench.benchmarks.exp_64K },
    .{ "Elementwise", "tanh_64K", elementwise_bench.benchmarks.tanh_64K },
    .{ "Elementwise", "relu_1M", elementwise_bench.benchmarks.relu_1M },
    .{ "Elementwise", "exp_1M", elementwise_bench.benchmarks.exp_1M },
    .{ "Elementwise", "tanh_1M", elementwise_bench.benchmarks.tanh_1M },
    .{ "Elementwise", "add_64K", elementwise_bench.benchmarks.add_64K },
    .{ "Elementwise", "mul_64K", elementwise_bench.benchmarks.mul_64K },
    .{ "Elementwise", "add_1M", elementwise_bench.benchmarks.add_1M },
    .{ "Elementwise", "mul_1M", elementwise_bench.benchmarks.mul_1M },

    // Reductions
    .{ "Reductions", "sum_64K", reduce_bench.benchmarks.sum_64K },
    .{ "Reductions", "max_64K", reduce_bench.benchmarks.max_64K },
    .{ "Reductions", "mean_64K", reduce_bench.benchmarks.mean_64K },
    .{ "Reductions", "sum_1M", reduce_bench.benchmarks.sum_1M },
    .{ "Reductions", "max_1M", reduce_bench.benchmarks.max_1M },
    .{ "Reductions", "mean_1M", reduce_bench.benchmarks.mean_1M },
    .{ "Reductions", "argmax_64K", reduce_bench.benchmarks.argmax_64K },
    .{ "Reductions", "argmax_1M", reduce_bench.benchmarks.argmax_1M },

    // Softmax
    .{ "Softmax", "softmax_64x64", softmax_bench.benchmarks.softmax_64x64 },
    .{ "Softmax", "softmax_128x128", softmax_bench.benchmarks.softmax_128x128 },
    .{ "Softmax", "softmax_256x256", softmax_bench.benchmarks.softmax_256x256 },
    .{ "Softmax", "softmax_512x512", softmax_bench.benchmarks.softmax_512x512 },
    .{ "Softmax", "softmax_64x384", softmax_bench.benchmarks.softmax_64x384 },
    .{ "Softmax", "softmax_256x768", softmax_bench.benchmarks.softmax_256x768 },

    // Layer Normalization
    .{ "Layer Normalization", "layernorm_128x384", layernorm_bench.benchmarks.layernorm_128x384 },
    .{ "Layer Normalization", "layernorm_512x384", layernorm_bench.benchmarks.layernorm_512x384 },
    .{ "Layer Normalization", "layernorm_128x768", layernorm_bench.benchmarks.layernorm_128x768 },
    .{ "Layer Normalization", "layernorm_512x768", layernorm_bench.benchmarks.layernorm_512x768 },
    .{ "Layer Normalization", "layernorm_1x384", layernorm_bench.benchmarks.layernorm_1x384 },
    .{ "Layer Normalization", "layernorm_1x768", layernorm_bench.benchmarks.layernorm_1x768 },

    // LeNet Model
    .{ "LeNet-5 Model", "lenet_forward_b1", lenet_bench.benchmarks.lenet_forward_b1 },
    .{ "LeNet-5 Model", "lenet_forward_b16", lenet_bench.benchmarks.lenet_forward_b16 },
    .{ "LeNet-5 Model", "lenet_forward_b64", lenet_bench.benchmarks.lenet_forward_b64 },
    .{ "LeNet-5 Model", "lenet_train_b16", lenet_bench.benchmarks.lenet_train_b16 },
    .{ "LeNet-5 Model", "lenet_train_b64", lenet_bench.benchmarks.lenet_train_b64 },

    // Thread Scaling
    .{ "Thread Scaling", "scaling_matmul_t1", thread_scaling_bench.benchmarks.scaling_matmul_t1 },
    .{ "Thread Scaling", "scaling_matmul_t2", thread_scaling_bench.benchmarks.scaling_matmul_t2 },
    .{ "Thread Scaling", "scaling_matmul_t4", thread_scaling_bench.benchmarks.scaling_matmul_t4 },
    .{ "Thread Scaling", "scaling_matmul_t8", thread_scaling_bench.benchmarks.scaling_matmul_t8 },
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Parse output path from args
    var args = std.process.args();
    _ = args.skip(); // Skip program name
    const output_path = args.next() orelse "docs/src/benchmarks.html";

    const bench_count = all_benchmarks.len;
    std.debug.print("Running {d} benchmarks for HTML report...\n\n", .{bench_count});

    // Collect benchmark results
    var results: std.ArrayListUnmanaged(html_report.BenchmarkData) = .empty;
    defer {
        for (results.items) |item| {
            allocator.free(item.name);
        }
        results.deinit(allocator);
    }

    // Short config for report generation (faster)
    const config = ziterion.Config{
        .warmup_time_ns = 50 * std.time.ns_per_ms,
        .measurement_time_ns = 500 * std.time.ns_per_ms,
        .min_samples = 30,
        .max_samples = 500,
    };

    var current_category: []const u8 = "";

    // Use inline for to iterate at comptime
    inline for (all_benchmarks) |bench| {
        const category = bench[0];
        const name = bench[1];
        const func = bench[2];

        if (!std.mem.eql(u8, category, current_category)) {
            current_category = category;
            std.debug.print("\n=== {s} ===\n\n", .{current_category});
        }

        std.debug.print("  {s}...", .{name});

        // Run benchmark
        var state = ziterion.State.init(allocator);
        defer state.deinit();

        var result = ziterion.measure.measure(allocator, func, &state, config) catch |err| {
            std.debug.print(" error: {}\n", .{err});
            unreachable;
        };
        defer result.deinit();

        // Format and print result
        var time_buf: [32]u8 = undefined;
        const time_str = ziterion.formatNanoseconds(result.metrics.median, &time_buf);
        std.debug.print(" {s}\n", .{time_str});

        // Create display name with category prefix
        const display_name = try std.fmt.allocPrint(allocator, "[{s}] {s}", .{ category, name });

        try results.append(allocator, .{
            .name = display_name,
            .metrics = result.metrics,
            .samples = null, // Don't include raw samples to keep report small
        });
    }

    std.debug.print("\n\nGenerating HTML report to {s}...\n", .{output_path});

    // Generate HTML report
    const html = try html_report.generate(allocator, results.items, .{
        .title = "Tenzor Performance Benchmarks",
        .theme = .light,
    });
    defer allocator.free(html);

    // Write to file
    const file = try std.fs.cwd().createFile(output_path, .{});
    defer file.close();
    try file.writeAll(html);

    std.debug.print("Done! Report written to {s}\n", .{output_path});
}
