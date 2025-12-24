//! CBLAS-backed matrix multiplication kernels.
//!
//! Provides high-performance GEMM via system BLAS (OpenBLAS, MKL, etc).
//! Only available on native builds - WASM falls back to pure Zig.

const std = @import("std");
const builtin = @import("builtin");
const options = @import("tenzor_options");

// CBLAS is optional; when disabled we fall back to pure Zig kernels.
pub const has_cblas = options.enable_blas and builtin.os.tag != .freestanding and builtin.cpu.arch != .wasm32;

// Import CBLAS bindings when available
const cblas = if (has_cblas) @import("cblas") else struct {};

/// CBLAS matrix layout (c_uint for enum)
const CblasRowMajor: c_uint = 101;
const CblasColMajor: c_uint = 102;

/// CBLAS transpose options (c_uint for enum)
const CblasNoTrans: c_uint = 111;
const CblasTrans: c_uint = 112;
const CblasConjTrans: c_uint = 113;

/// GEMM: C = alpha * A @ B + beta * C
/// Uses CBLAS sgemm/dgemm for optimal performance.
pub fn gemm(
    comptime T: type,
    a: []const T,
    b: []const T,
    c: []T,
    m: usize,
    k: usize,
    n: usize,
    alpha: T,
    beta: T,
    trans_a: bool,
    trans_b: bool,
) void {
    if (comptime !has_cblas) {
        @compileError("CBLAS not available on this target");
    }

    const m_i: c_int = @intCast(m);
    const n_i: c_int = @intCast(n);
    const k_i: c_int = @intCast(k);

    // Leading dimensions for row-major layout
    const lda: c_int = if (trans_a) m_i else k_i;
    const ldb: c_int = if (trans_b) k_i else n_i;
    const ldc: c_int = n_i;

    const trans_a_flag: c_uint = if (trans_a) CblasTrans else CblasNoTrans;
    const trans_b_flag: c_uint = if (trans_b) CblasTrans else CblasNoTrans;

    if (T == f32) {
        cblas.cblas_sgemm(
            CblasRowMajor,
            trans_a_flag,
            trans_b_flag,
            m_i,
            n_i,
            k_i,
            alpha,
            a.ptr,
            lda,
            b.ptr,
            ldb,
            beta,
            c.ptr,
            ldc,
        );
    } else if (T == f64) {
        cblas.cblas_dgemm(
            CblasRowMajor,
            trans_a_flag,
            trans_b_flag,
            m_i,
            n_i,
            k_i,
            alpha,
            a.ptr,
            lda,
            b.ptr,
            ldb,
            beta,
            c.ptr,
            ldc,
        );
    } else {
        @compileError("CBLAS GEMM only supports f32 and f64");
    }
}

/// Simple matmul: C = A @ B (alpha=1, beta=0)
pub fn matmul(
    comptime T: type,
    a: []const T,
    b: []const T,
    c: []T,
    m: usize,
    k: usize,
    n: usize,
) void {
    gemm(T, a, b, c, m, k, n, 1.0, 0.0, false, false);
}

/// Matrix-vector multiply: y = A @ x
pub fn gemv(
    comptime T: type,
    a: []const T,
    x: []const T,
    y: []T,
    m: usize,
    n: usize,
    alpha: T,
    beta: T,
    trans: bool,
) void {
    if (comptime !has_cblas) {
        @compileError("CBLAS not available on this target");
    }

    const m_i: c_int = @intCast(m);
    const n_i: c_int = @intCast(n);
    const lda: c_int = n_i;
    const trans_flag: c_uint = if (trans) CblasTrans else CblasNoTrans;

    if (T == f32) {
        cblas.cblas_sgemv(
            CblasRowMajor,
            trans_flag,
            m_i,
            n_i,
            alpha,
            a.ptr,
            lda,
            x.ptr,
            1, // incx
            beta,
            y.ptr,
            1, // incy
        );
    } else if (T == f64) {
        cblas.cblas_dgemv(
            CblasRowMajor,
            trans_flag,
            m_i,
            n_i,
            alpha,
            a.ptr,
            lda,
            x.ptr,
            1,
            beta,
            y.ptr,
            1,
        );
    } else {
        @compileError("CBLAS GEMV only supports f32 and f64");
    }
}

/// Matrix-vector multiply: y = A @ x (simple form)
pub fn matVecmul(
    comptime T: type,
    a: []const T,
    x: []const T,
    y: []T,
    m: usize,
    k: usize,
) void {
    @memset(y, 0);
    gemv(T, a, x, y, m, k, 1.0, 0.0, false);
}

/// Vector-matrix multiply: y = x @ A
pub fn vecMatmul(
    comptime T: type,
    x: []const T,
    a: []const T,
    y: []T,
    k: usize,
    n: usize,
) void {
    // x @ A = (A^T @ x^T)^T, but since x and y are vectors, this simplifies
    // to using transposed GEMV
    @memset(y, 0);
    gemv(T, a, x, y, k, n, 1.0, 0.0, true);
}

/// Dot product: result = a . b
pub fn dot(
    comptime T: type,
    a: []const T,
    b: []const T,
) T {
    if (comptime !has_cblas) {
        @compileError("CBLAS not available on this target");
    }

    const n: c_int = @intCast(a.len);

    if (T == f32) {
        return cblas.cblas_sdot(n, a.ptr, 1, b.ptr, 1);
    } else if (T == f64) {
        return cblas.cblas_ddot(n, a.ptr, 1, b.ptr, 1);
    } else {
        @compileError("CBLAS dot only supports f32 and f64");
    }
}

/// Batched matrix multiplication using CBLAS.
/// C[b] = A[b] @ B[b] for each batch.
pub fn batchedMatmul(
    comptime T: type,
    a: []const T,
    b: []const T,
    c: []T,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) void {
    const a_batch_stride = m * k;
    const b_batch_stride = k * n;
    const c_batch_stride = m * n;

    for (0..batch) |batch_idx| {
        const a_offset = batch_idx * a_batch_stride;
        const b_offset = batch_idx * b_batch_stride;
        const c_offset = batch_idx * c_batch_stride;

        matmul(
            T,
            a[a_offset..][0..a_batch_stride],
            b[b_offset..][0..b_batch_stride],
            c[c_offset..][0..c_batch_stride],
            m,
            k,
            n,
        );
    }
}

/// Batched matmul with broadcast: A is [batch, M, K], B is [K, N]
pub fn batchedMatmulBroadcastB(
    comptime T: type,
    a: []const T,
    b: []const T,
    c: []T,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) void {
    const a_batch_stride = m * k;
    const c_batch_stride = m * n;

    for (0..batch) |batch_idx| {
        const a_offset = batch_idx * a_batch_stride;
        const c_offset = batch_idx * c_batch_stride;

        matmul(
            T,
            a[a_offset..][0..a_batch_stride],
            b[0 .. k * n],
            c[c_offset..][0..c_batch_stride],
            m,
            k,
            n,
        );
    }
}

// ============================================================================
// Tests
// ============================================================================

test "blas gemm 2x2" {
    if (!has_cblas) return error.SkipZigTest;

    const a = [_]f32{ 1, 2, 3, 4 };
    const b = [_]f32{ 5, 6, 7, 8 };
    var c: [4]f32 = .{ 0, 0, 0, 0 };

    matmul(f32, &a, &b, &c, 2, 2, 2);

    // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
    try std.testing.expectApproxEqAbs(@as(f32, 19), c[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 22), c[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 43), c[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 50), c[3], 1e-5);
}

test "blas gemm larger" {
    if (!has_cblas) return error.SkipZigTest;

    // 64x48 @ 48x32
    const m = 64;
    const k = 48;
    const n = 32;

    var a: [m * k]f32 = undefined;
    var b: [k * n]f32 = undefined;
    var c: [m * n]f32 = undefined;

    // Initialize with simple pattern
    for (&a, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 10)) / 10.0;
    for (&b, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i + 3) % 10)) / 10.0;

    matmul(f32, &a, &b, &c, m, k, n);

    // Just check it ran without crashing and produced non-zero output
    var sum: f32 = 0;
    for (c) |v| sum += v;
    try std.testing.expect(sum != 0);
}

test "blas dot product" {
    if (!has_cblas) return error.SkipZigTest;

    const a = [_]f32{ 1, 2, 3, 4 };
    const b = [_]f32{ 1, 1, 1, 1 };

    const result = dot(f32, &a, &b);
    try std.testing.expectApproxEqAbs(@as(f32, 10), result, 1e-5);
}

test "blas matvec" {
    if (!has_cblas) return error.SkipZigTest;

    // A = [[1,2],[3,4],[5,6]], x = [1, 2]
    // y = A @ x = [5, 11, 17]
    const a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const x = [_]f32{ 1, 2 };
    var y: [3]f32 = undefined;

    matVecmul(f32, &a, &x, &y, 3, 2);

    try std.testing.expectApproxEqAbs(@as(f32, 5), y[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 11), y[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 17), y[2], 1e-5);
}
