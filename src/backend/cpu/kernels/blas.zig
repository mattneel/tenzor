//! CBLAS-backed matrix multiplication kernels.
//!
//! Provides high-performance GEMM via system BLAS (OpenBLAS, MKL, etc).
//! Only available on native builds - WASM falls back to pure Zig.

const std = @import("std");
const builtin = @import("builtin");
const options = @import("tenzor_options");

// CBLAS is optional; when unavailable at runtime we fall back to pure Zig kernels.
pub const can_try_cblas = options.enable_blas and builtin.os.tag != .freestanding and builtin.cpu.arch != .wasm32;

pub const Status = enum {
    /// Build-time disabled (`-Dblas=false`).
    disabled,
    /// Platform/target unsupported (WASM/freestanding).
    unsupported,
    /// Enabled, but no compatible BLAS found at runtime.
    unavailable,
    /// Loaded from `TENZOR_BLAS_LIB`.
    custom,
    /// macOS Accelerate.framework.
    accelerate,
    /// OpenBLAS (or compatible).
    openblas,
    /// Intel MKL runtime.
    mkl,
    /// Generic CBLAS.
    cblas,
    /// Generic BLAS (with CBLAS symbols).
    blas,
};

pub fn status() Status {
    return Runtime.status();
}

pub fn statusString(s: Status) []const u8 {
    return switch (s) {
        .disabled => "disabled",
        .unsupported => "unsupported",
        .unavailable => "unavailable",
        .custom => "custom",
        .accelerate => "accelerate",
        .openblas => "openblas",
        .mkl => "mkl",
        .cblas => "cblas",
        .blas => "blas",
    };
}

pub fn statusName() []const u8 {
    return statusString(status());
}

const CblasSgemmFn = *const fn (
    layout: c_int,
    trans_a: c_int,
    trans_b: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: f32,
    a: [*]const f32,
    lda: c_int,
    b: [*]const f32,
    ldb: c_int,
    beta: f32,
    c: [*]f32,
    ldc: c_int,
) callconv(.c) void;

const CblasDgemmFn = *const fn (
    layout: c_int,
    trans_a: c_int,
    trans_b: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: f64,
    a: [*]const f64,
    lda: c_int,
    b: [*]const f64,
    ldb: c_int,
    beta: f64,
    c: [*]f64,
    ldc: c_int,
) callconv(.c) void;

const CblasSgemvFn = *const fn (
    layout: c_int,
    trans: c_int,
    m: c_int,
    n: c_int,
    alpha: f32,
    a: [*]const f32,
    lda: c_int,
    x: [*]const f32,
    incx: c_int,
    beta: f32,
    y: [*]f32,
    incy: c_int,
) callconv(.c) void;

const CblasDgemvFn = *const fn (
    layout: c_int,
    trans: c_int,
    m: c_int,
    n: c_int,
    alpha: f64,
    a: [*]const f64,
    lda: c_int,
    x: [*]const f64,
    incx: c_int,
    beta: f64,
    y: [*]f64,
    incy: c_int,
) callconv(.c) void;

const CblasSdotFn = *const fn (n: c_int, a: [*]const f32, inc_a: c_int, b: [*]const f32, inc_b: c_int) callconv(.c) f32;
const CblasDdotFn = *const fn (n: c_int, a: [*]const f64, inc_a: c_int, b: [*]const f64, inc_b: c_int) callconv(.c) f64;

const CblasApi = struct {
    sgemm: CblasSgemmFn,
    dgemm: CblasDgemmFn,
    sgemv: CblasSgemvFn,
    dgemv: CblasDgemvFn,
    sdot: CblasSdotFn,
    ddot: CblasDdotFn,
};

pub fn has_cblas() bool {
    return Runtime.hasCblas();
}

fn api() ?*const CblasApi {
    return Runtime.apiPtr();
}

const Runtime = if (can_try_cblas) struct {
    const Loaded = struct {
        lib: std.DynLib,
        api: CblasApi,
        status: Status,
    };

    var once = std.once(init);
    var loaded: bool = false;
    var lib: std.DynLib = undefined;
    var api_state: CblasApi = undefined;
    var status_state: Status = .unavailable;
    var init_err: ?anyerror = null;

    fn init() void {
        const loaded_result = loadCblas() catch |err| {
            init_err = err;
            return;
        };

        lib = loaded_result.lib;
        api_state = loaded_result.api;
        status_state = loaded_result.status;
        loaded = true;
    }

    pub fn hasCblas() bool {
        once.call();
        return loaded;
    }

    pub fn status() Status {
        once.call();
        if (!loaded) return .unavailable;
        return status_state;
    }

    pub fn apiPtr() ?*const CblasApi {
        if (!hasCblas()) return null;
        return &api_state;
    }

    fn loadFromPath(path: []const u8, loaded_status: Status) !Loaded {
        var dynlib = try std.DynLib.open(path);
        errdefer dynlib.close();

        const api_loaded = CblasApi{
            .sgemm = dynlib.lookup(CblasSgemmFn, "cblas_sgemm") orelse return error.MissingSymbol,
            .dgemm = dynlib.lookup(CblasDgemmFn, "cblas_dgemm") orelse return error.MissingSymbol,
            .sgemv = dynlib.lookup(CblasSgemvFn, "cblas_sgemv") orelse return error.MissingSymbol,
            .dgemv = dynlib.lookup(CblasDgemvFn, "cblas_dgemv") orelse return error.MissingSymbol,
            .sdot = dynlib.lookup(CblasSdotFn, "cblas_sdot") orelse return error.MissingSymbol,
            .ddot = dynlib.lookup(CblasDdotFn, "cblas_ddot") orelse return error.MissingSymbol,
        };

        return .{ .lib = dynlib, .api = api_loaded, .status = loaded_status };
    }

    fn loadCblas() !Loaded {
        if (std.process.getEnvVarOwned(std.heap.page_allocator, "TENZOR_BLAS_LIB")) |path| {
            defer std.heap.page_allocator.free(path);
            if (loadFromPath(path, .custom)) |loaded_result| return loaded_result else |_| {}
        } else |_| {}

        const Candidate = struct { path: []const u8, status: Status };

        const candidates = switch (builtin.os.tag) {
            .linux => &[_]Candidate{
                .{ .path = "libopenblas.so.0", .status = .openblas },
                .{ .path = "libopenblas.so", .status = .openblas },
                .{ .path = "libopenblas64_.so.0", .status = .openblas },
                .{ .path = "libopenblas64_.so", .status = .openblas },
                .{ .path = "libopenblas64.so.0", .status = .openblas },
                .{ .path = "libopenblas64.so", .status = .openblas },
                .{ .path = "libopenblasp.so.0", .status = .openblas },
                .{ .path = "libopenblasp.so", .status = .openblas },
                .{ .path = "libcblas.so.3", .status = .cblas },
                .{ .path = "libcblas.so", .status = .cblas },
                .{ .path = "libblas.so.3", .status = .blas },
                .{ .path = "libblas.so", .status = .blas },
                .{ .path = "libmkl_rt.so", .status = .mkl },
            },
            .macos => &[_]Candidate{
                .{ .path = "/System/Library/Frameworks/Accelerate.framework/Accelerate", .status = .accelerate },
                .{ .path = "/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate", .status = .accelerate },
                .{ .path = "libopenblas.dylib", .status = .openblas },
                .{ .path = "libcblas.dylib", .status = .cblas },
                .{ .path = "libblas.dylib", .status = .blas },
            },
            .windows => &[_]Candidate{
                .{ .path = "openblas.dll", .status = .openblas },
                .{ .path = "libopenblas.dll", .status = .openblas },
                .{ .path = "openblas64_.dll", .status = .openblas },
                .{ .path = "libopenblas64_.dll", .status = .openblas },
                .{ .path = "mkl_rt.dll", .status = .mkl },
            },
            else => &[_]Candidate{},
        };

        for (candidates) |cand| {
            if (loadFromPath(cand.path, cand.status)) |loaded_result| return loaded_result else |_| {}
        }

        return error.BlasUnavailable;
    }
} else struct {
    pub fn hasCblas() bool {
        return false;
    }

    pub fn status() Status {
        if (!options.enable_blas) return .disabled;
        return .unsupported;
    }

    pub fn apiPtr() ?*const CblasApi {
        return null;
    }
};

/// CBLAS matrix layout (c_uint for enum)
const CblasRowMajor: c_int = 101;
const CblasColMajor: c_int = 102;

/// CBLAS transpose options (c_uint for enum)
const CblasNoTrans: c_int = 111;
const CblasTrans: c_int = 112;
const CblasConjTrans: c_int = 113;

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
) bool {
    const cblas = api() orelse return false;

    const m_i: c_int = @intCast(m);
    const n_i: c_int = @intCast(n);
    const k_i: c_int = @intCast(k);

    // Leading dimensions for row-major layout
    const lda: c_int = if (trans_a) m_i else k_i;
    const ldb: c_int = if (trans_b) k_i else n_i;
    const ldc: c_int = n_i;

    const trans_a_flag: c_int = if (trans_a) CblasTrans else CblasNoTrans;
    const trans_b_flag: c_int = if (trans_b) CblasTrans else CblasNoTrans;

    if (T == f32) {
        cblas.sgemm(
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
        return true;
    } else if (T == f64) {
        cblas.dgemm(
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
        return true;
    } else {
        return false;
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
) bool {
    return gemm(T, a, b, c, m, k, n, 1.0, 0.0, false, false);
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
) bool {
    const cblas = api() orelse return false;

    const m_i: c_int = @intCast(m);
    const n_i: c_int = @intCast(n);
    const lda: c_int = n_i;
    const trans_flag: c_int = if (trans) CblasTrans else CblasNoTrans;

    if (T == f32) {
        cblas.sgemv(
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
        return true;
    } else if (T == f64) {
        cblas.dgemv(
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
        return true;
    } else {
        return false;
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
) bool {
    @memset(y, 0);
    return gemv(T, a, x, y, m, k, 1.0, 0.0, false);
}

/// Vector-matrix multiply: y = x @ A
pub fn vecMatmul(
    comptime T: type,
    x: []const T,
    a: []const T,
    y: []T,
    k: usize,
    n: usize,
) bool {
    // x @ A = (A^T @ x^T)^T, but since x and y are vectors, this simplifies
    // to using transposed GEMV
    @memset(y, 0);
    return gemv(T, a, x, y, k, n, 1.0, 0.0, true);
}

/// Dot product: result = a . b
pub fn dot(
    comptime T: type,
    a: []const T,
    b: []const T,
) ?T {
    const cblas = api() orelse return null;

    const n: c_int = @intCast(a.len);

    if (T == f32) {
        return cblas.sdot(n, a.ptr, 1, b.ptr, 1);
    } else if (T == f64) {
        return cblas.ddot(n, a.ptr, 1, b.ptr, 1);
    } else {
        return null;
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
    if (!has_cblas()) return error.SkipZigTest;

    const a = [_]f32{ 1, 2, 3, 4 };
    const b = [_]f32{ 5, 6, 7, 8 };
    var c: [4]f32 = .{ 0, 0, 0, 0 };

    _ = matmul(f32, &a, &b, &c, 2, 2, 2);

    // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
    try std.testing.expectApproxEqAbs(@as(f32, 19), c[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 22), c[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 43), c[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 50), c[3], 1e-5);
}

test "blas gemm larger" {
    if (!has_cblas()) return error.SkipZigTest;

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

    _ = matmul(f32, &a, &b, &c, m, k, n);

    // Just check it ran without crashing and produced non-zero output
    var sum: f32 = 0;
    for (c) |v| sum += v;
    try std.testing.expect(sum != 0);
}

test "blas dot product" {
    if (!has_cblas()) return error.SkipZigTest;

    const a = [_]f32{ 1, 2, 3, 4 };
    const b = [_]f32{ 1, 1, 1, 1 };

    const result = dot(f32, &a, &b).?;
    try std.testing.expectApproxEqAbs(@as(f32, 10), result, 1e-5);
}

test "blas matvec" {
    if (!has_cblas()) return error.SkipZigTest;

    // A = [[1,2],[3,4],[5,6]], x = [1, 2]
    // y = A @ x = [5, 11, 17]
    const a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const x = [_]f32{ 1, 2 };
    var y: [3]f32 = undefined;

    _ = matVecmul(f32, &a, &x, &y, 3, 2);

    try std.testing.expectApproxEqAbs(@as(f32, 5), y[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 11), y[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 17), y[2], 1e-5);
}
