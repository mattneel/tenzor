//! SIMD utilities for CPU backend.
//!
//! Provides portable SIMD abstractions using Zig's @Vector type.
//! Automatically selects optimal vector width for the target architecture.

const std = @import("std");

/// Get the suggested vector length for a type on the current architecture.
pub fn suggestVectorLength(comptime T: type) comptime_int {
    return std.simd.suggestVectorLength(T) orelse defaultVectorLength(T);
}

/// Default vector length when architecture-specific suggestion isn't available.
fn defaultVectorLength(comptime T: type) comptime_int {
    // Target 256 bits (32 bytes) which is common (AVX)
    const target_bytes = 32;
    const elem_size = @sizeOf(T);
    return @max(1, target_bytes / elem_size);
}

/// Vector type for a given element type.
pub fn Vec(comptime T: type) type {
    return @Vector(suggestVectorLength(T), T);
}

/// Create a vector with all elements set to the same value.
pub fn splat(comptime T: type, value: T) Vec(T) {
    return @splat(value);
}

/// Load a vector from a slice.
pub fn load(comptime T: type, slice: []const T) Vec(T) {
    const len = suggestVectorLength(T);
    return slice[0..len].*;
}

/// Store a vector to a slice.
pub fn store(comptime T: type, vec: Vec(T), slice: []T) void {
    const len = suggestVectorLength(T);
    slice[0..len].* = vec;
}

/// Reduce a vector to a single value using addition.
pub fn reduceAdd(comptime T: type, vec: Vec(T)) T {
    return @reduce(.Add, vec);
}

/// Reduce a vector to a single value using multiplication.
pub fn reduceMul(comptime T: type, vec: Vec(T)) T {
    return @reduce(.Mul, vec);
}

/// Reduce a vector to find the maximum value.
pub fn reduceMax(comptime T: type, vec: Vec(T)) T {
    return @reduce(.Max, vec);
}

/// Reduce a vector to find the minimum value.
pub fn reduceMin(comptime T: type, vec: Vec(T)) T {
    return @reduce(.Min, vec);
}

// ============================================================================
// Elementwise operations on vectors
// ============================================================================

/// Negate
pub fn neg(comptime T: type, a: Vec(T)) Vec(T) {
    return -a;
}

/// Absolute value
pub fn abs(comptime T: type, a: Vec(T)) Vec(T) {
    return @abs(a);
}

/// Add two vectors
pub fn add(comptime T: type, a: Vec(T), b: Vec(T)) Vec(T) {
    return a + b;
}

/// Subtract two vectors
pub fn sub(comptime T: type, a: Vec(T), b: Vec(T)) Vec(T) {
    return a - b;
}

/// Multiply two vectors
pub fn mul(comptime T: type, a: Vec(T), b: Vec(T)) Vec(T) {
    return a * b;
}

/// Divide two vectors
pub fn div(comptime T: type, a: Vec(T), b: Vec(T)) Vec(T) {
    return a / b;
}

/// Element-wise maximum
pub fn max(comptime T: type, a: Vec(T), b: Vec(T)) Vec(T) {
    return @max(a, b);
}

/// Element-wise minimum
pub fn min(comptime T: type, a: Vec(T), b: Vec(T)) Vec(T) {
    return @min(a, b);
}

/// Square root
pub fn sqrt(comptime T: type, a: Vec(T)) Vec(T) {
    return @sqrt(a);
}

/// Reciprocal square root (1/sqrt(x))
pub fn rsqrt(comptime T: type, a: Vec(T)) Vec(T) {
    const one: Vec(T) = @splat(1.0);
    return one / @sqrt(a);
}

/// Exponential
pub fn exp(comptime T: type, a: Vec(T)) Vec(T) {
    return @exp(a);
}

/// Natural logarithm
pub fn log(comptime T: type, a: Vec(T)) Vec(T) {
    return @log(a);
}

/// Sine
pub fn sin(comptime T: type, a: Vec(T)) Vec(T) {
    return @sin(a);
}

/// Cosine
pub fn cos(comptime T: type, a: Vec(T)) Vec(T) {
    return @cos(a);
}

/// Hyperbolic tangent
/// Uses numerically stable formula: tanh(x) = 2*sigmoid(2x) - 1
pub fn tanh(comptime T: type, a: Vec(T)) Vec(T) {
    const two: Vec(T) = @splat(2.0);
    const one: Vec(T) = @splat(1.0);
    return two * sigmoid(T, two * a) - one;
}

/// ReLU activation
pub fn relu(comptime T: type, a: Vec(T)) Vec(T) {
    const zero: Vec(T) = @splat(0.0);
    return @max(a, zero);
}

/// Leaky ReLU activation
pub fn leakyRelu(comptime T: type, a: Vec(T), alpha: T) Vec(T) {
    const zero: Vec(T) = @splat(0.0);
    const alpha_vec: Vec(T) = @splat(alpha);
    const mask = a > zero;
    return @select(T, mask, a, a * alpha_vec);
}

/// Sigmoid activation: 1 / (1 + exp(-x))
pub fn sigmoid(comptime T: type, a: Vec(T)) Vec(T) {
    const one: Vec(T) = @splat(1.0);
    return one / (one + @exp(-a));
}

/// GELU activation (approximate)
/// gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
pub fn gelu(comptime T: type, a: Vec(T)) Vec(T) {
    const half: Vec(T) = @splat(0.5);
    const one: Vec(T) = @splat(1.0);
    const sqrt_2_over_pi: Vec(T) = @splat(0.7978845608028654); // sqrt(2/π)
    const coeff: Vec(T) = @splat(0.044715);

    const x3 = a * a * a;
    const inner = sqrt_2_over_pi * (a + coeff * x3);
    return half * a * (one + tanh(T, inner));
}

/// SiLU (Swish) activation: x * sigmoid(x)
pub fn silu(comptime T: type, a: Vec(T)) Vec(T) {
    return a * sigmoid(T, a);
}

/// Softplus activation: log(1 + exp(x))
pub fn softplus(comptime T: type, a: Vec(T)) Vec(T) {
    const one: Vec(T) = @splat(1.0);
    return @log(one + @exp(a));
}

/// Floor
pub fn floor(comptime T: type, a: Vec(T)) Vec(T) {
    return @floor(a);
}

/// Ceiling
pub fn ceil(comptime T: type, a: Vec(T)) Vec(T) {
    return @ceil(a);
}

/// Round to nearest
pub fn round(comptime T: type, a: Vec(T)) Vec(T) {
    return @round(a);
}

// ============================================================================
// Scalar operations (for remainders and non-vectorized paths)
// ============================================================================

pub const scalar = struct {
    pub fn neg(x: anytype) @TypeOf(x) {
        return -x;
    }

    pub fn abs(x: anytype) @TypeOf(x) {
        return @abs(x);
    }

    pub fn add(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
        return a + b;
    }

    pub fn sub(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
        return a - b;
    }

    pub fn mul(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
        return a * b;
    }

    pub fn div(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
        return a / b;
    }

    pub fn max(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
        return @max(a, b);
    }

    pub fn min(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
        return @min(a, b);
    }

    pub fn sqrt(x: anytype) @TypeOf(x) {
        return @sqrt(x);
    }

    pub fn rsqrt(x: anytype) @TypeOf(x) {
        return 1.0 / @sqrt(x);
    }

    pub fn exp(x: anytype) @TypeOf(x) {
        return @exp(x);
    }

    pub fn log(x: anytype) @TypeOf(x) {
        return @log(x);
    }

    pub fn sin(x: anytype) @TypeOf(x) {
        return @sin(x);
    }

    pub fn cos(x: anytype) @TypeOf(x) {
        return @cos(x);
    }

    pub fn tanh(x: anytype) @TypeOf(x) {
        // tanh(x) = 2*sigmoid(2x) - 1 (numerically stable)
        return 2.0 * scalar.sigmoid(2.0 * x) - 1.0;
    }

    pub fn relu(x: anytype) @TypeOf(x) {
        return @max(x, 0.0);
    }

    pub fn leakyRelu(x: anytype, alpha: @TypeOf(x)) @TypeOf(x) {
        return if (x > 0) x else alpha * x;
    }

    pub fn sigmoid(x: anytype) @TypeOf(x) {
        return 1.0 / (1.0 + @exp(-x));
    }

    pub fn gelu(x: anytype) @TypeOf(x) {
        const T = @TypeOf(x);
        const sqrt_2_over_pi: T = 0.7978845608028654;
        const coeff: T = 0.044715;
        const x3 = x * x * x;
        const inner = sqrt_2_over_pi * (x + coeff * x3);
        return 0.5 * x * (1.0 + scalar.tanh(inner));
    }

    pub fn silu(x: anytype) @TypeOf(x) {
        return x * scalar.sigmoid(x);
    }

    pub fn softplus(x: anytype) @TypeOf(x) {
        return @log(1.0 + @exp(x));
    }

    pub fn floor(x: anytype) @TypeOf(x) {
        return @floor(x);
    }

    pub fn ceil(x: anytype) @TypeOf(x) {
        return @ceil(x);
    }

    pub fn round(x: anytype) @TypeOf(x) {
        return @round(x);
    }

    pub fn pow(base: anytype, exponent: @TypeOf(base)) @TypeOf(base) {
        return std.math.pow(@TypeOf(base), base, exponent);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "vector length suggestion" {
    const f32_len = suggestVectorLength(f32);
    try std.testing.expect(f32_len >= 4); // At least 128-bit
    try std.testing.expect(f32_len <= 16); // At most 512-bit

    const f64_len = suggestVectorLength(f64);
    try std.testing.expect(f64_len >= 2);
    try std.testing.expect(f64_len <= 8);
}

test "splat and reduce" {
    const vec = splat(f32, 1.0);
    const sum = reduceAdd(f32, vec);
    try std.testing.expectEqual(@as(f32, @floatFromInt(suggestVectorLength(f32))), sum);
}

test "basic arithmetic" {
    const a = splat(f32, 2.0);
    const b = splat(f32, 3.0);

    const sum = add(f32, a, b);
    try std.testing.expectEqual(@as(f32, 5.0), sum[0]);

    const product = mul(f32, a, b);
    try std.testing.expectEqual(@as(f32, 6.0), product[0]);

    const diff = sub(f32, b, a);
    try std.testing.expectEqual(@as(f32, 1.0), diff[0]);

    const quotient = div(f32, b, a);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), quotient[0], 1e-6);
}

test "relu" {
    const len = suggestVectorLength(f32);
    var data: [len]f32 = undefined;
    for (&data, 0..) |*d, i| {
        d.* = @as(f32, @floatFromInt(i)) - @as(f32, @floatFromInt(len / 2));
    }
    const vec: Vec(f32) = data;
    const result = relu(f32, vec);
    const result_arr: [len]f32 = result;

    for (result_arr, 0..) |v, i| {
        const expected = @max(data[i], 0.0);
        try std.testing.expectEqual(expected, v);
    }
}

test "sigmoid bounds" {
    const neg_large = splat(f32, -100.0);
    const pos_large = splat(f32, 100.0);
    const zero_vec = splat(f32, 0.0);

    const sig_neg = sigmoid(f32, neg_large);
    const sig_pos = sigmoid(f32, pos_large);
    const sig_zero = sigmoid(f32, zero_vec);

    // sigmoid(-100) ≈ 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), sig_neg[0], 1e-6);
    // sigmoid(100) ≈ 1
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sig_pos[0], 1e-6);
    // sigmoid(0) = 0.5
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), sig_zero[0], 1e-6);
}

test "tanh bounds" {
    const neg_large = splat(f32, -100.0);
    const pos_large = splat(f32, 100.0);
    const zero_vec = splat(f32, 0.0);

    const tanh_neg = tanh(f32, neg_large);
    const tanh_pos = tanh(f32, pos_large);
    const tanh_zero = tanh(f32, zero_vec);

    // tanh(-100) ≈ -1
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), tanh_neg[0], 1e-6);
    // tanh(100) ≈ 1
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), tanh_pos[0], 1e-6);
    // tanh(0) = 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), tanh_zero[0], 1e-6);
}

test "scalar operations" {
    try std.testing.expectEqual(@as(f32, 4.0), scalar.add(@as(f32, 2.0), @as(f32, 2.0)));
    try std.testing.expectEqual(@as(f32, 6.0), scalar.mul(@as(f32, 2.0), @as(f32, 3.0)));
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), scalar.relu(@as(f32, -5.0)), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), scalar.relu(@as(f32, 5.0)), 1e-6);
}
