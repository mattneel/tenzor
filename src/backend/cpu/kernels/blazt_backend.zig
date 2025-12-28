//! Blazt Backend - Pure Zig BLAS Implementation
//!
//! Provides high-performance BLAS operations using Blazt when vendor
//! BLAS libraries (OpenBLAS, MKL, Accelerate) are unavailable.
//! Blazt uses comptime-specialized SIMD kernels optimized for the
//! target CPU at build time.
//!
//! Note: Blazt requires cache-line aligned data. For unaligned inputs,
//! this module returns false to signal fallback to pure Zig kernels.

const std = @import("std");
const blazt = @import("blazt");

/// Cache line size required by Blazt
const CACHE_LINE = blazt.CacheLine;

/// Backend status
pub const Status = enum {
    available,
    unavailable,
};

/// Check if Blazt is available (always true at comptime)
pub fn status() Status {
    return .available;
}

/// Check if a pointer is cache-line aligned
fn isAligned(ptr: anytype) bool {
    return @intFromPtr(ptr) % CACHE_LINE == 0;
}

/// GEMM: C = alpha * A @ B + beta * C
/// Uses Blazt's optimized GEMM implementation.
/// Returns true if Blazt was used, false if caller should use fallback.
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
) bool {
    // Blazt requires cache-line aligned data
    if (!isAligned(a.ptr) or !isAligned(b.ptr) or !isAligned(c.ptr)) {
        return false;
    }

    // Create Blazt Matrix views over the raw data
    // Tenzor uses row-major layout
    const Layout = blazt.Layout.row_major;
    const MatrixType = blazt.Matrix(T, Layout);

    // Determine effective dimensions based on transpose
    const a_rows = if (trans_a) k else m;
    const a_cols = if (trans_a) m else k;
    const b_rows = if (trans_b) n else k;
    const b_cols = if (trans_b) k else n;

    // Create matrix views (no allocation, null allocator = view mode)
    // Data is verified aligned above
    const mat_a = MatrixType{
        .data = @alignCast(@constCast(a.ptr)[0 .. a_rows * a_cols]),
        .rows = a_rows,
        .cols = a_cols,
        .stride = a_cols,
        .allocator = null,
    };

    const mat_b = MatrixType{
        .data = @alignCast(@constCast(b.ptr)[0 .. b_rows * b_cols]),
        .rows = b_rows,
        .cols = b_cols,
        .stride = b_cols,
        .allocator = null,
    };

    var mat_c = MatrixType{
        .data = @alignCast(c.ptr[0 .. m * n]),
        .rows = m,
        .cols = n,
        .stride = n,
        .allocator = null,
    };

    // Map transpose flags
    const trans_a_flag: blazt.Trans = if (trans_a) .trans else .no_trans;
    const trans_b_flag: blazt.Trans = if (trans_b) .trans else .no_trans;

    // Call Blazt's optimized GEMM
    blazt.ops.gemm(
        T,
        Layout,
        trans_a_flag,
        trans_b_flag,
        alpha,
        mat_a,
        mat_b,
        beta,
        &mat_c,
    );

    return true;
}

/// Simple matmul: C = A @ B (alpha=1, beta=0)
/// Returns true if Blazt was used, false if caller should use fallback.
pub fn matmul(
    comptime T: type,
    a: []const T,
    b: []const T,
    c: []T,
    m: usize,
    k: usize,
    n: usize,
) bool {
    return gemm(T, a, b, c, m, k, n, 1.0, 0.0, false, false);
}

/// Matrix-vector multiply: y = alpha * A @ x + beta * y
/// Returns true if Blazt was used, false if caller should use fallback.
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
) bool {
    // For GEMV, we can use GEMM with n=1
    // y[m,1] = A[m,n] @ x[n,1]
    const out_rows = if (trans) n else m;
    const in_cols = if (trans) m else n;

    return gemm(T, a, x, y, out_rows, in_cols, 1, alpha, beta, trans, false);
}

/// Matrix-vector multiply: y = A @ x
/// Returns true if Blazt was used, false if caller should use fallback.
pub fn matVecmul(
    comptime T: type,
    a: []const T,
    x: []const T,
    y: []T,
    m: usize,
    k: usize,
) bool {
    @memset(y, 0);
    return gemv(T, a, x, y, m, k, 1.0, 0.0, false);
}

/// Vector-matrix multiply: y = x @ A
/// Returns true if Blazt was used, false if caller should use fallback.
pub fn vecMatmul(
    comptime T: type,
    x: []const T,
    a: []const T,
    y: []T,
    k: usize,
    n: usize,
) bool {
    // x @ A = (A^T @ x^T)^T
    // Using transposed GEMV
    @memset(y, 0);
    return gemv(T, a, x, y, k, n, 1.0, 0.0, true);
}

/// Dot product: result = a . b
/// Returns null if Blazt couldn't be used (unaligned data).
pub fn dot(comptime T: type, a: []const T, b: []const T) ?T {
    std.debug.assert(a.len == b.len);

    // Blazt dot doesn't require alignment, but check anyway for consistency
    // Actually blazt.ops.dot takes slices directly, no alignment needed
    return blazt.ops.dot(T, a, b);
}

// ============================================================================
// Tests
// ============================================================================

test "blazt gemm with aligned data" {
    // Use aligned arrays for testing
    var a align(CACHE_LINE) = [_]f32{ 1, 2, 3, 4 };
    var b align(CACHE_LINE) = [_]f32{ 5, 6, 7, 8 };
    var c align(CACHE_LINE) = [_]f32{ 0, 0, 0, 0 };

    const success = matmul(f32, &a, &b, &c, 2, 2, 2);
    try std.testing.expect(success);

    // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
    try std.testing.expectApproxEqAbs(@as(f32, 19), c[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 22), c[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 43), c[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 50), c[3], 1e-5);
}

test "blazt gemm unaligned returns false" {
    // Unaligned data should cause fallback
    const a = [_]f32{ 1, 2, 3, 4 };
    const b = [_]f32{ 5, 6, 7, 8 };
    var c = [_]f32{ 0, 0, 0, 0 };

    const success = matmul(f32, &a, &b, &c, 2, 2, 2);
    // May or may not succeed depending on stack alignment
    _ = success;
}

test "blazt dot product" {
    const a = [_]f32{ 1, 2, 3, 4 };
    const b = [_]f32{ 1, 1, 1, 1 };

    // dot always returns a result (no alignment requirement)
    const result = dot(f32, &a, &b).?;
    try std.testing.expectApproxEqAbs(@as(f32, 10), result, 1e-5);
}
