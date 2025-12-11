//! Matrix multiplication CPU kernel with tiling optimization.
//!
//! Implements efficient matrix multiplication with cache-aware tiling
//! and SIMD vectorization where possible.

const std = @import("std");
const simd = @import("../simd.zig");

/// Tile sizes for cache optimization.
/// These are chosen to fit in L1 cache (~32KB) while maintaining efficiency.
const TILE_M: usize = 32;
const TILE_N: usize = 32;
const TILE_K: usize = 32;

/// Naive matrix multiplication: C = A @ B
/// A is M x K, B is K x N, C is M x N
pub fn matmulNaive(
    comptime T: type,
    a: []const T,
    b: []const T,
    c: []T,
    m: usize,
    k: usize,
    n: usize,
) void {
    // Initialize C to zero
    @memset(c, 0);

    for (0..m) |i| {
        for (0..k) |kk| {
            const a_ik = a[i * k + kk];
            for (0..n) |j| {
                c[i * n + j] += a_ik * b[kk * n + j];
            }
        }
    }
}

/// Tiled matrix multiplication: C = A @ B
/// Uses blocking for cache efficiency.
pub fn matmulTiled(
    comptime T: type,
    a: []const T,
    b: []const T,
    c: []T,
    m: usize,
    k: usize,
    n: usize,
) void {
    // Initialize C to zero
    @memset(c, 0);

    // Tile over M
    var ii: usize = 0;
    while (ii < m) : (ii += TILE_M) {
        const m_end = @min(ii + TILE_M, m);

        // Tile over N
        var jj: usize = 0;
        while (jj < n) : (jj += TILE_N) {
            const n_end = @min(jj + TILE_N, n);

            // Tile over K
            var kk: usize = 0;
            while (kk < k) : (kk += TILE_K) {
                const k_end = @min(kk + TILE_K, k);

                // Compute tile: C[ii:ii+TILE_M, jj:jj+TILE_N] += A[ii:ii+TILE_M, kk:kk+TILE_K] @ B[kk:kk+TILE_K, jj:jj+TILE_N]
                matmulTile(T, a, b, c, m, k, n, ii, m_end, jj, n_end, kk, k_end);
            }
        }
    }
}

/// Compute a single tile of the matrix multiplication.
fn matmulTile(
    comptime T: type,
    a: []const T,
    b: []const T,
    c: []T,
    m: usize,
    k: usize,
    n: usize,
    i_start: usize,
    i_end: usize,
    j_start: usize,
    j_end: usize,
    k_start: usize,
    k_end: usize,
) void {
    _ = m; // autofix
    for (i_start..i_end) |i| {
        for (k_start..k_end) |kk| {
            const a_ik = a[i * k + kk];

            // Inner loop - potentially vectorizable
            var j = j_start;

            // SIMD path
            const vec_len = simd.suggestVectorLength(T);
            while (j + vec_len <= j_end) : (j += vec_len) {
                const b_vec = simd.load(T, b[kk * n + j ..]);
                var c_vec = simd.load(T, c[i * n + j ..]);

                const a_vec = simd.splat(T, a_ik);
                c_vec = simd.add(T, c_vec, simd.mul(T, a_vec, b_vec));

                simd.store(T, c_vec, c[i * n + j ..]);
            }

            // Scalar remainder
            while (j < j_end) : (j += 1) {
                c[i * n + j] += a_ik * b[kk * n + j];
            }
        }
    }
}

/// Matrix multiplication with optional transpose flags.
/// C = op(A) @ op(B) where op is identity or transpose.
pub fn matmulGeneral(
    comptime T: type,
    a: []const T,
    b: []const T,
    c: []T,
    m: usize,
    k: usize,
    n: usize,
    trans_a: bool,
    trans_b: bool,
    lda: usize,
    ldb: usize,
    ldc: usize,
) void {
    // Initialize C to zero
    @memset(c, 0);

    for (0..m) |i| {
        for (0..k) |kk| {
            const a_idx = if (trans_a) kk * lda + i else i * lda + kk;
            const a_ik = a[a_idx];

            for (0..n) |j| {
                const b_idx = if (trans_b) j * ldb + kk else kk * ldb + j;
                c[i * ldc + j] += a_ik * b[b_idx];
            }
        }
    }
}

/// Vector-matrix multiplication: y = x @ A
/// x is a vector of length K, A is K x N, y is a vector of length N.
pub fn vecMatmul(
    comptime T: type,
    x: []const T,
    a: []const T,
    y: []T,
    k: usize,
    n: usize,
) void {
    @memset(y, 0);

    for (0..k) |kk| {
        const x_k = x[kk];

        var j: usize = 0;
        const vec_len = simd.suggestVectorLength(T);

        // SIMD path
        while (j + vec_len <= n) : (j += vec_len) {
            const a_vec = simd.load(T, a[kk * n + j ..]);
            var y_vec = simd.load(T, y[j..]);

            const x_vec = simd.splat(T, x_k);
            y_vec = simd.add(T, y_vec, simd.mul(T, x_vec, a_vec));

            simd.store(T, y_vec, y[j..]);
        }

        // Scalar remainder
        while (j < n) : (j += 1) {
            y[j] += x_k * a[kk * n + j];
        }
    }
}

/// Matrix-vector multiplication: y = A @ x
/// A is M x K, x is a vector of length K, y is a vector of length M.
pub fn matVecmul(
    comptime T: type,
    a: []const T,
    x: []const T,
    y: []T,
    m: usize,
    k: usize,
) void {
    for (0..m) |i| {
        var sum: T = 0;

        var j: usize = 0;
        const vec_len = simd.suggestVectorLength(T);

        // SIMD path - accumulate partial sums
        while (j + vec_len <= k) : (j += vec_len) {
            const a_vec = simd.load(T, a[i * k + j ..]);
            const x_vec = simd.load(T, x[j..]);
            sum += simd.reduceAdd(T, simd.mul(T, a_vec, x_vec));
        }

        // Scalar remainder
        while (j < k) : (j += 1) {
            sum += a[i * k + j] * x[j];
        }

        y[i] = sum;
    }
}

/// Dot product of two vectors.
pub fn dotProduct(comptime T: type, a: []const T, b: []const T) T {
    std.debug.assert(a.len == b.len);

    var sum: T = 0;
    var i: usize = 0;
    const vec_len = simd.suggestVectorLength(T);

    // SIMD path
    while (i + vec_len <= a.len) : (i += vec_len) {
        const a_vec = simd.load(T, a[i..]);
        const b_vec = simd.load(T, b[i..]);
        sum += simd.reduceAdd(T, simd.mul(T, a_vec, b_vec));
    }

    // Scalar remainder
    while (i < a.len) : (i += 1) {
        sum += a[i] * b[i];
    }

    return sum;
}

/// Outer product: C = a * b^T
/// a is a vector of length M, b is a vector of length N, C is M x N.
pub fn outerProduct(
    comptime T: type,
    a: []const T,
    b: []const T,
    c: []T,
    m: usize,
    n: usize,
) void {
    for (0..m) |i| {
        const a_i = a[i];

        var j: usize = 0;
        const vec_len = simd.suggestVectorLength(T);

        // SIMD path
        while (j + vec_len <= n) : (j += vec_len) {
            const b_vec = simd.load(T, b[j..]);
            const a_vec = simd.splat(T, a_i);
            const result = simd.mul(T, a_vec, b_vec);
            simd.store(T, result, c[i * n + j ..]);
        }

        // Scalar remainder
        while (j < n) : (j += 1) {
            c[i * n + j] = a_i * b[j];
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

test "matmulNaive 2x2" {
    // A = [[1, 2], [3, 4]]
    // B = [[5, 6], [7, 8]]
    // C = A @ B = [[19, 22], [43, 50]]
    const a = [_]f32{ 1, 2, 3, 4 };
    const b = [_]f32{ 5, 6, 7, 8 };
    var c: [4]f32 = undefined;

    matmulNaive(f32, &a, &b, &c, 2, 2, 2);

    try std.testing.expectApproxEqAbs(@as(f32, 19), c[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 22), c[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 43), c[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 50), c[3], 1e-6);
}

test "matmulNaive 3x3" {
    // Identity matrix test
    const identity = [_]f32{ 1, 0, 0, 0, 1, 0, 0, 0, 1 };
    const a = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    var c: [9]f32 = undefined;

    matmulNaive(f32, &a, &identity, &c, 3, 3, 3);

    for (a, c) |av, cv| {
        try std.testing.expectApproxEqAbs(av, cv, 1e-6);
    }
}

test "matmulTiled vs naive" {
    // Random-ish matrix multiplication, compare tiled vs naive
    const m = 64;
    const k = 48;
    const n = 32;

    var a: [m * k]f32 = undefined;
    var b: [k * n]f32 = undefined;
    var c_naive: [m * n]f32 = undefined;
    var c_tiled: [m * n]f32 = undefined;

    // Initialize with simple pattern
    for (&a, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 10)) / 10.0;
    for (&b, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i + 3) % 10)) / 10.0;

    matmulNaive(f32, &a, &b, &c_naive, m, k, n);
    matmulTiled(f32, &a, &b, &c_tiled, m, k, n);

    for (c_naive, c_tiled) |naive, tiled| {
        try std.testing.expectApproxEqAbs(naive, tiled, 1e-5);
    }
}

test "vecMatmul" {
    // x = [1, 2], A = [[1, 2, 3], [4, 5, 6]]
    // y = x @ A = [9, 12, 15]
    const x = [_]f32{ 1, 2 };
    const a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var y: [3]f32 = undefined;

    vecMatmul(f32, &x, &a, &y, 2, 3);

    try std.testing.expectApproxEqAbs(@as(f32, 9), y[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 12), y[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 15), y[2], 1e-6);
}

test "matVecmul" {
    // A = [[1, 2], [3, 4], [5, 6]], x = [1, 2]
    // y = A @ x = [5, 11, 17]
    const a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const x = [_]f32{ 1, 2 };
    var y: [3]f32 = undefined;

    matVecmul(f32, &a, &x, &y, 3, 2);

    try std.testing.expectApproxEqAbs(@as(f32, 5), y[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 11), y[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 17), y[2], 1e-6);
}

test "dotProduct" {
    const a = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const b = [_]f32{ 1, 1, 1, 1, 1, 1, 1, 1 };

    const result = dotProduct(f32, &a, &b);
    try std.testing.expectApproxEqAbs(@as(f32, 36), result, 1e-6); // 1+2+3+4+5+6+7+8 = 36
}

test "outerProduct" {
    // a = [1, 2], b = [3, 4, 5]
    // C = a * b^T = [[3, 4, 5], [6, 8, 10]]
    const a = [_]f32{ 1, 2 };
    const b = [_]f32{ 3, 4, 5 };
    var c: [6]f32 = undefined;

    outerProduct(f32, &a, &b, &c, 2, 3);

    const expected = [_]f32{ 3, 4, 5, 6, 8, 10 };
    for (c, expected) |cv, ev| {
        try std.testing.expectApproxEqAbs(ev, cv, 1e-6);
    }
}

test "matmulNaive non-square" {
    // A is 2x3, B is 3x4, C is 2x4
    const a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const b = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    var c: [8]f32 = undefined;

    matmulNaive(f32, &a, &b, &c, 2, 3, 4);

    // Row 0: [1,2,3] @ B = [1+10+27, 2+12+30, 3+14+33, 4+16+36] = [38, 44, 50, 56]
    // Row 1: [4,5,6] @ B = [4+25+54, 8+30+60, 12+35+66, 16+40+72] = [83, 98, 113, 128]
    try std.testing.expectApproxEqAbs(@as(f32, 38), c[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 44), c[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 50), c[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 56), c[3], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 83), c[4], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 98), c[5], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 113), c[6], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 128), c[7], 1e-5);
}
