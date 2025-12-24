//! Weight-only int8 matmul kernels (W8A32 -> f32).
//!
//! Designed for models with f32 activations and per-output-channel symmetric int8 weights.
//! For a weight matrix Wq (i8) with per-row scale S (f16), the effective weight is:
//!   W ≈ Wq * S[row]
//!
//! This file provides kernels that compute:
//!   C = A @ W^T + bias
//! where A is f32 and Wq is stored as [out, in] (transposed relative to standard matmul).

const std = @import("std");
const simd = @import("../simd.zig");

pub fn matmulTransposeB_f32_i8_rowScale_bias(
    a: []const f32,
    wq: []const i8, // [out, in]
    scale: []const f16, // [out]
    bias: []const f32, // [out]
    c: []f32,
    m: usize,
    k: usize,
    n: usize,
) void {
    if (m == 0 or n == 0) return;
    std.debug.assert(a.len >= m * k);
    std.debug.assert(wq.len >= n * k);
    std.debug.assert(scale.len >= n);
    std.debug.assert(bias.len >= n);
    std.debug.assert(c.len >= m * n);

    const vec_len = simd.suggestVectorLength(f32);
    const VecF = @Vector(vec_len, f32);
    const VecI8 = @Vector(vec_len, i8);

    for (0..m) |i| {
        const a_row = a[i * k ..][0..k];
        for (0..n) |j| {
            const w_row = wq[j * k ..][0..k];

            var acc: VecF = @splat(0.0);
            var kk: usize = 0;
            while (kk + vec_len <= k) : (kk += vec_len) {
                const a_vec: VecF = a_row[kk..][0..vec_len].*;
                const w_i8: VecI8 = w_row[kk..][0..vec_len].*;
                const w_f32: VecF = @floatFromInt(w_i8);
                acc += a_vec * w_f32;
            }

            var sum: f32 = @reduce(.Add, acc);
            while (kk < k) : (kk += 1) {
                sum += a_row[kk] * @as(f32, @floatFromInt(w_row[kk]));
            }

            const s: f32 = @floatCast(scale[j]);
            c[i * n + j] = sum * s + bias[j];
        }
    }
}
