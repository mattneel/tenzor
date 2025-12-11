//! Fused kernel code generation.
//!
//! Generates optimized kernels that execute multiple operations
//! in a single pass, reducing memory traffic.

const std = @import("std");
const ops = @import("../ops/expr.zig");
const simd = @import("../backend/cpu/simd.zig");
const matmul_kernels = @import("../backend/cpu/kernels/matmul.zig");

const OpTag = ops.OpTag;

/// Generate a fused elementwise kernel for a chain of operations.
/// Returns a function that applies all operations in sequence.
pub fn FusedElementwiseKernel(comptime T: type, comptime operations: []const OpTag) type {
    return struct {
        /// Apply fused operations to input and write to output.
        pub fn apply(input: []const T, output: []T) void {
            const vec_len = simd.suggestVectorLength(T);
            var i: usize = 0;

            // SIMD path
            while (i + vec_len <= input.len) : (i += vec_len) {
                var v = simd.load(T, input[i..]);

                // Apply each operation in sequence
                inline for (operations) |op| {
                    v = applyUnaryVec(op, T, v);
                }

                simd.store(T, v, output[i..]);
            }

            // Scalar remainder
            while (i < input.len) : (i += 1) {
                var val = input[i];

                inline for (operations) |op| {
                    val = applyUnaryScalar(op, T, val);
                }

                output[i] = val;
            }
        }

        /// Apply fused operations in-place.
        pub fn applyInplace(data: []T) void {
            const vec_len = simd.suggestVectorLength(T);
            var i: usize = 0;

            // SIMD path
            while (i + vec_len <= data.len) : (i += vec_len) {
                var v = simd.load(T, data[i..]);

                inline for (operations) |op| {
                    v = applyUnaryVec(op, T, v);
                }

                simd.store(T, v, data[i..]);
            }

            // Scalar remainder
            while (i < data.len) : (i += 1) {
                var val = data[i];

                inline for (operations) |op| {
                    val = applyUnaryScalar(op, T, val);
                }

                data[i] = val;
            }
        }
    };
}

/// Generate a fused matmul + epilogue kernel.
/// Computes C = activation(A @ B + bias) in optimized form.
pub fn FusedMatmulEpilogueKernel(
    comptime T: type,
    comptime epilogue_ops: []const OpTag,
) type {
    return struct {
        /// Execute matmul with fused epilogue.
        /// bias can be null if there's no bias add.
        pub fn apply(
            a: []const T,
            b: []const T,
            c: []T,
            m: usize,
            k: usize,
            n: usize,
            bias: ?[]const T,
        ) void {
            // First compute the matmul
            matmul_kernels.matmulTiled(T, a, b, c, m, k, n);

            // Apply epilogue operations
            applyEpilogue(c, m, n, bias);
        }

        fn applyEpilogue(c: []T, m: usize, n: usize, bias: ?[]const T) void {
            // Process each row
            for (0..m) |i| {
                const row_start = i * n;
                const row_end = row_start + n;
                const row = c[row_start..row_end];

                var j: usize = 0;
                const vec_len = simd.suggestVectorLength(T);

                // SIMD path
                while (j + vec_len <= n) : (j += vec_len) {
                    var v = simd.load(T, row[j..]);

                    // Apply epilogue ops
                    inline for (epilogue_ops) |op| {
                        if (op == .add and bias != null) {
                            // Bias add - load bias vector
                            const bias_v = simd.load(T, bias.?[j..]);
                            v = simd.add(T, v, bias_v);
                        } else if (op != .add) {
                            // Activation
                            v = applyUnaryVec(op, T, v);
                        }
                    }

                    simd.store(T, v, row[j..]);
                }

                // Scalar remainder
                while (j < n) : (j += 1) {
                    var val = row[j];

                    inline for (epilogue_ops) |op| {
                        if (op == .add and bias != null) {
                            val += bias.?[j];
                        } else if (op != .add) {
                            val = applyUnaryScalar(op, T, val);
                        }
                    }

                    row[j] = val;
                }
            }
        }
    };
}

/// Generate a fused reduce kernel with pre-reduction elementwise ops.
/// Computes reduce(elementwise(input)) in a single pass.
pub fn FusedReduceKernel(
    comptime T: type,
    comptime reduce_op: OpTag,
    comptime pre_ops: []const OpTag,
) type {
    return struct {
        /// Apply fused elementwise + reduce.
        pub fn apply(input: []const T) T {
            if (input.len == 0) {
                return identityValue(reduce_op, T);
            }

            const vec_len = simd.suggestVectorLength(T);
            var i: usize = 0;

            // Initialize accumulator
            var acc_vec = simd.splat(T, identityValue(reduce_op, T));

            // SIMD loop - apply elementwise ops then reduce
            while (i + vec_len <= input.len) : (i += vec_len) {
                var v = simd.load(T, input[i..]);

                // Apply pre-ops
                inline for (pre_ops) |op| {
                    v = applyUnaryVec(op, T, v);
                }

                // Accumulate
                acc_vec = applyReduceVec(reduce_op, T, acc_vec, v);
            }

            // Reduce vector to scalar
            var acc = reduceVecToScalar(reduce_op, T, acc_vec);

            // Scalar remainder
            while (i < input.len) : (i += 1) {
                var val = input[i];

                // Apply pre-ops
                inline for (pre_ops) |op| {
                    val = applyUnaryScalar(op, T, val);
                }

                acc = applyReduceScalar(reduce_op, T, acc, val);
            }

            // Post-processing for mean
            if (reduce_op == .mean) {
                acc = acc / @as(T, @floatFromInt(input.len));
            }

            return acc;
        }
    };
}

// ============================================================================
// Helper functions
// ============================================================================

/// Apply unary operation to a SIMD vector.
fn applyUnaryVec(comptime op: OpTag, comptime T: type, v: simd.Vec(T)) simd.Vec(T) {
    return switch (op) {
        .neg => simd.neg(T, v),
        .abs => simd.abs(T, v),
        .exp => simd.exp(T, v),
        .log => simd.log(T, v),
        .sqrt => simd.sqrt(T, v),
        .rsqrt => simd.rsqrt(T, v),
        .sin => simd.sin(T, v),
        .cos => simd.cos(T, v),
        .tanh => simd.tanh(T, v),
        .sigmoid => simd.sigmoid(T, v),
        .relu => simd.relu(T, v),
        .gelu => simd.gelu(T, v),
        .silu => simd.silu(T, v),
        else => @compileError("Unsupported unary op for fusion: " ++ @tagName(op)),
    };
}

/// Apply unary operation to a scalar.
fn applyUnaryScalar(comptime op: OpTag, comptime T: type, val: T) T {
    return switch (op) {
        .neg => simd.scalar.neg(val),
        .abs => simd.scalar.abs(val),
        .exp => simd.scalar.exp(val),
        .log => simd.scalar.log(val),
        .sqrt => simd.scalar.sqrt(val),
        .rsqrt => simd.scalar.rsqrt(val),
        .sin => simd.scalar.sin(val),
        .cos => simd.scalar.cos(val),
        .tanh => simd.scalar.tanh(val),
        .sigmoid => simd.scalar.sigmoid(val),
        .relu => simd.scalar.relu(val),
        .gelu => simd.scalar.gelu(val),
        .silu => simd.scalar.silu(val),
        else => @compileError("Unsupported unary op for fusion: " ++ @tagName(op)),
    };
}

/// Identity value for a reduction operation.
fn identityValue(comptime op: OpTag, comptime T: type) T {
    return switch (op) {
        .sum, .mean => 0,
        .prod => 1,
        .reduce_max => -std.math.inf(T),
        .reduce_min => std.math.inf(T),
        else => @compileError("Unsupported reduction: " ++ @tagName(op)),
    };
}

/// Apply reduction operation to two vectors.
fn applyReduceVec(comptime op: OpTag, comptime T: type, acc: simd.Vec(T), v: simd.Vec(T)) simd.Vec(T) {
    return switch (op) {
        .sum, .mean => simd.add(T, acc, v),
        .prod => simd.mul(T, acc, v),
        .reduce_max => simd.max(T, acc, v),
        .reduce_min => simd.min(T, acc, v),
        else => @compileError("Unsupported reduction: " ++ @tagName(op)),
    };
}

/// Reduce a vector to a scalar.
fn reduceVecToScalar(comptime op: OpTag, comptime T: type, v: simd.Vec(T)) T {
    return switch (op) {
        .sum, .mean => simd.reduceAdd(T, v),
        .prod => simd.reduceMul(T, v),
        .reduce_max => simd.reduceMax(T, v),
        .reduce_min => simd.reduceMin(T, v),
        else => @compileError("Unsupported reduction: " ++ @tagName(op)),
    };
}

/// Apply reduction operation to two scalars.
fn applyReduceScalar(comptime op: OpTag, comptime T: type, acc: T, v: T) T {
    return switch (op) {
        .sum, .mean => acc + v,
        .prod => acc * v,
        .reduce_max => @max(acc, v),
        .reduce_min => @min(acc, v),
        else => @compileError("Unsupported reduction: " ++ @tagName(op)),
    };
}

// ============================================================================
// Tests
// ============================================================================

test "fused elementwise kernel" {
    // Test relu(exp(x))
    const Kernel = FusedElementwiseKernel(f32, &[_]OpTag{ .exp, .relu });

    const input = [_]f32{ -1, 0, 1, 2 };
    var output: [4]f32 = undefined;

    Kernel.apply(&input, &output);

    // exp(-1) ≈ 0.368, relu = 0.368
    // exp(0) = 1, relu = 1
    // exp(1) ≈ 2.718, relu = 2.718
    // exp(2) ≈ 7.389, relu = 7.389
    try std.testing.expectApproxEqAbs(@as(f32, 0.368), output[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), output[1], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 2.718), output[2], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 7.389), output[3], 0.01);
}

test "fused elementwise inplace" {
    const Kernel = FusedElementwiseKernel(f32, &[_]OpTag{.relu});

    var data = [_]f32{ -2, -1, 0, 1, 2 };
    Kernel.applyInplace(&data);

    try std.testing.expectEqual(@as(f32, 0), data[0]);
    try std.testing.expectEqual(@as(f32, 0), data[1]);
    try std.testing.expectEqual(@as(f32, 0), data[2]);
    try std.testing.expectEqual(@as(f32, 1), data[3]);
    try std.testing.expectEqual(@as(f32, 2), data[4]);
}

test "fused matmul epilogue" {
    const Kernel = FusedMatmulEpilogueKernel(f32, &[_]OpTag{ .add, .relu });

    // 2x2 @ 2x2 + bias with relu
    const a = [_]f32{ 1, 2, 3, 4 };
    const b = [_]f32{ 1, 0, 0, 1 }; // Identity
    const bias = [_]f32{ -10, 10 };
    var c: [4]f32 = undefined;

    Kernel.apply(&a, &b, &c, 2, 2, 2, &bias);

    // Result before relu: [[-9, 12], [-7, 14]]
    // After relu: [[0, 12], [0, 14]]
    try std.testing.expectEqual(@as(f32, 0), c[0]); // max(-9, 0)
    try std.testing.expectEqual(@as(f32, 12), c[1]); // 2 + 10
    try std.testing.expectEqual(@as(f32, 0), c[2]); // max(-7, 0)
    try std.testing.expectEqual(@as(f32, 14), c[3]); // 4 + 10
}

test "fused reduce kernel" {
    // Test sum(exp(x))
    const Kernel = FusedReduceKernel(f32, .sum, &[_]OpTag{.exp});

    const input = [_]f32{ 0, 0, 0, 0 }; // exp(0) = 1
    const result = Kernel.apply(&input);

    try std.testing.expectApproxEqAbs(@as(f32, 4.0), result, 0.001);
}

test "fused reduce with multiple pre-ops" {
    // Test sum(relu(x))
    const Kernel = FusedReduceKernel(f32, .sum, &[_]OpTag{.relu});

    const input = [_]f32{ -2, -1, 0, 1, 2 };
    const result = Kernel.apply(&input);

    // relu: [0, 0, 0, 1, 2], sum = 3
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), result, 0.001);
}
