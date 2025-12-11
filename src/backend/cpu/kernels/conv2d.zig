//! 2D Convolution CPU kernel for CNN training.
//!
//! Implements forward and backward passes for 2D convolution:
//! - Forward: Y = conv2d(X, W) + b
//! - Backward: dX = conv2d_transpose(dY, W), dW = conv2d(X^T, dY), db = sum(dY)
//!
//! Layout: NHWC (batch, height, width, channels) - CPU/SIMD friendly

const std = @import("std");

/// Compute output dimension for convolution.
pub fn convOutputSize(
    input_size: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) usize {
    return (input_size + 2 * padding - kernel_size) / stride + 1;
}

/// 2D Convolution forward pass.
///
/// Input layout:  [N, H, W, C_in]  (NHWC)
/// Weight layout: [C_out, kH, kW, C_in]
/// Bias layout:   [C_out] (optional)
/// Output layout: [N, H_out, W_out, C_out]
pub fn conv2dForward(
    comptime T: type,
    input: []const T,
    weight: []const T,
    bias: ?[]const T,
    output: []T,
    batch: usize,
    in_h: usize,
    in_w: usize,
    in_c: usize,
    out_c: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) void {
    const out_h = convOutputSize(in_h, kernel_h, stride_h, pad_h);
    const out_w = convOutputSize(in_w, kernel_w, stride_w, pad_w);

    // Initialize output with bias (or zeros)
    if (bias) |b| {
        for (0..batch) |n| {
            for (0..out_h) |oh| {
                for (0..out_w) |ow| {
                    for (0..out_c) |oc| {
                        const out_idx = ((n * out_h + oh) * out_w + ow) * out_c + oc;
                        output[out_idx] = b[oc];
                    }
                }
            }
        }
    } else {
        @memset(output, 0);
    }

    // Convolution: accumulate weighted sums
    for (0..batch) |n| {
        for (0..out_h) |oh| {
            for (0..out_w) |ow| {
                // Input position (top-left of receptive field)
                const ih_start = oh * stride_h;
                const iw_start = ow * stride_w;

                for (0..out_c) |oc| {
                    var sum: T = 0;

                    for (0..kernel_h) |kh| {
                        for (0..kernel_w) |kw| {
                            // Handle padding
                            const ih = ih_start + kh;
                            const iw = iw_start + kw;

                            // Check bounds (for padding)
                            if (ih >= pad_h and ih < in_h + pad_h and
                                iw >= pad_w and iw < in_w + pad_w)
                            {
                                const ih_actual = ih - pad_h;
                                const iw_actual = iw - pad_w;

                                for (0..in_c) |ic| {
                                    const in_idx = ((n * in_h + ih_actual) * in_w + iw_actual) * in_c + ic;
                                    const w_idx = ((oc * kernel_h + kh) * kernel_w + kw) * in_c + ic;
                                    sum += input[in_idx] * weight[w_idx];
                                }
                            }
                        }
                    }

                    const out_idx = ((n * out_h + oh) * out_w + ow) * out_c + oc;
                    output[out_idx] += sum;
                }
            }
        }
    }
}

/// 2D Convolution backward pass for input gradient.
///
/// Computes dL/dX given dL/dY (grad_output) and weights.
/// This is essentially a "full" convolution with rotated kernel.
pub fn conv2dBackwardInput(
    comptime T: type,
    grad_output: []const T,
    weight: []const T,
    grad_input: []T,
    batch: usize,
    in_h: usize,
    in_w: usize,
    in_c: usize,
    out_c: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) void {
    const out_h = convOutputSize(in_h, kernel_h, stride_h, pad_h);
    const out_w = convOutputSize(in_w, kernel_w, stride_w, pad_w);

    // Initialize grad_input to zeros
    @memset(grad_input, 0);

    // For each output gradient, scatter back to input
    for (0..batch) |n| {
        for (0..out_h) |oh| {
            for (0..out_w) |ow| {
                const ih_start = oh * stride_h;
                const iw_start = ow * stride_w;

                for (0..out_c) |oc| {
                    const grad_out_idx = ((n * out_h + oh) * out_w + ow) * out_c + oc;
                    const grad_out_val = grad_output[grad_out_idx];

                    for (0..kernel_h) |kh| {
                        for (0..kernel_w) |kw| {
                            const ih = ih_start + kh;
                            const iw = iw_start + kw;

                            if (ih >= pad_h and ih < in_h + pad_h and
                                iw >= pad_w and iw < in_w + pad_w)
                            {
                                const ih_actual = ih - pad_h;
                                const iw_actual = iw - pad_w;

                                for (0..in_c) |ic| {
                                    const in_idx = ((n * in_h + ih_actual) * in_w + iw_actual) * in_c + ic;
                                    const w_idx = ((oc * kernel_h + kh) * kernel_w + kw) * in_c + ic;
                                    grad_input[in_idx] += grad_out_val * weight[w_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// 2D Convolution backward pass for weight gradient.
///
/// Computes dL/dW and dL/db given dL/dY (grad_output) and input.
pub fn conv2dBackwardWeight(
    comptime T: type,
    input: []const T,
    grad_output: []const T,
    grad_weight: []T,
    grad_bias: ?[]T,
    batch: usize,
    in_h: usize,
    in_w: usize,
    in_c: usize,
    out_c: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) void {
    const out_h = convOutputSize(in_h, kernel_h, stride_h, pad_h);
    const out_w = convOutputSize(in_w, kernel_w, stride_w, pad_w);

    // Initialize gradients to zeros
    @memset(grad_weight, 0);
    if (grad_bias) |gb| {
        @memset(gb, 0);
    }

    // Accumulate weight gradients
    for (0..batch) |n| {
        for (0..out_h) |oh| {
            for (0..out_w) |ow| {
                const ih_start = oh * stride_h;
                const iw_start = ow * stride_w;

                for (0..out_c) |oc| {
                    const grad_out_idx = ((n * out_h + oh) * out_w + ow) * out_c + oc;
                    const grad_out_val = grad_output[grad_out_idx];

                    // Accumulate bias gradient
                    if (grad_bias) |gb| {
                        gb[oc] += grad_out_val;
                    }

                    // Accumulate weight gradients
                    for (0..kernel_h) |kh| {
                        for (0..kernel_w) |kw| {
                            const ih = ih_start + kh;
                            const iw = iw_start + kw;

                            if (ih >= pad_h and ih < in_h + pad_h and
                                iw >= pad_w and iw < in_w + pad_w)
                            {
                                const ih_actual = ih - pad_h;
                                const iw_actual = iw - pad_w;

                                for (0..in_c) |ic| {
                                    const in_idx = ((n * in_h + ih_actual) * in_w + iw_actual) * in_c + ic;
                                    const w_idx = ((oc * kernel_h + kh) * kernel_w + kw) * in_c + ic;
                                    grad_weight[w_idx] += grad_out_val * input[in_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

test "conv2d output size calculation" {
    // 28x28 input, 5x5 kernel, stride 1, no padding -> 24x24
    try std.testing.expectEqual(@as(usize, 24), convOutputSize(28, 5, 1, 0));

    // 24x24 input, 2x2 kernel, stride 2, no padding -> 12x12
    try std.testing.expectEqual(@as(usize, 12), convOutputSize(24, 2, 2, 0));

    // 28x28 input, 5x5 kernel, stride 1, padding 2 -> 28x28 (same)
    try std.testing.expectEqual(@as(usize, 28), convOutputSize(28, 5, 1, 2));
}

test "conv2d forward basic" {
    // Simple 1x3x3x1 input, 1x2x2x1 kernel, stride 1, no padding
    // Expected output: 1x2x2x1
    const input = [_]f32{
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    };
    const weight = [_]f32{
        1, 0,
        0, 1,
    };

    var output: [4]f32 = undefined;

    conv2dForward(
        f32,
        &input,
        &weight,
        null,
        &output,
        1, // batch
        3,
        3,
        1, // in_h, in_w, in_c
        1,
        2,
        2, // out_c, kernel_h, kernel_w
        1,
        1,
        0,
        0, // stride_h, stride_w, pad_h, pad_w
    );

    // output[0,0] = 1*1 + 2*0 + 4*0 + 5*1 = 6
    // output[0,1] = 2*1 + 3*0 + 5*0 + 6*1 = 8
    // output[1,0] = 4*1 + 5*0 + 7*0 + 8*1 = 12
    // output[1,1] = 5*1 + 6*0 + 8*0 + 9*1 = 14
    try std.testing.expectApproxEqAbs(@as(f32, 6), output[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 8), output[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 12), output[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 14), output[3], 1e-6);
}

test "conv2d forward with bias" {
    const input = [_]f32{
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    };
    const weight = [_]f32{
        1, 0,
        0, 1,
    };
    const bias = [_]f32{10};

    var output: [4]f32 = undefined;

    conv2dForward(
        f32,
        &input,
        &weight,
        &bias,
        &output,
        1,
        3,
        3,
        1,
        1,
        2,
        2,
        1,
        1,
        0,
        0,
    );

    // Same as before but +10 for bias
    try std.testing.expectApproxEqAbs(@as(f32, 16), output[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 18), output[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 22), output[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 24), output[3], 1e-6);
}

test "conv2d backward weight gradient" {
    // Simple case: verify gradient accumulation
    const input = [_]f32{
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    };
    const grad_output = [_]f32{ 1, 1, 1, 1 };

    var grad_weight: [4]f32 = undefined;
    var grad_bias: [1]f32 = undefined;

    conv2dBackwardWeight(
        f32,
        &input,
        &grad_output,
        &grad_weight,
        &grad_bias,
        1,
        3,
        3,
        1,
        1,
        2,
        2,
        1,
        1,
        0,
        0,
    );

    // grad_weight[0,0] = sum of top-left corners weighted by grad_output
    // = 1*1 + 2*1 + 4*1 + 5*1 = 12
    try std.testing.expectApproxEqAbs(@as(f32, 12), grad_weight[0], 1e-6);

    // grad_bias = sum(grad_output) = 4
    try std.testing.expectApproxEqAbs(@as(f32, 4), grad_bias[0], 1e-6);
}

test "conv2d gradient check" {
    // Numerical gradient check using finite differences
    const eps: f32 = 1e-4;
    const tol: f32 = 1e-2; // Finite differences have ~O(eps) error

    const input = [_]f32{
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    };
    var weight = [_]f32{
        0.1, 0.2,
        0.3, 0.4,
    };

    // Forward pass
    var output: [4]f32 = undefined;
    conv2dForward(f32, &input, &weight, null, &output, 1, 3, 3, 1, 1, 2, 2, 1, 1, 0, 0);

    // Assume loss = sum(output), so grad_output = 1s
    const grad_output = [_]f32{ 1, 1, 1, 1 };

    // Backward pass
    var grad_weight: [4]f32 = undefined;
    conv2dBackwardWeight(f32, &input, &grad_output, &grad_weight, null, 1, 3, 3, 1, 1, 2, 2, 1, 1, 0, 0);

    // Numerical gradient for each weight
    for (0..4) |i| {
        // f(w + eps)
        weight[i] += eps;
        var out_plus: [4]f32 = undefined;
        conv2dForward(f32, &input, &weight, null, &out_plus, 1, 3, 3, 1, 1, 2, 2, 1, 1, 0, 0);
        var loss_plus: f32 = 0;
        for (out_plus) |v| loss_plus += v;

        // f(w - eps)
        weight[i] -= 2 * eps;
        var out_minus: [4]f32 = undefined;
        conv2dForward(f32, &input, &weight, null, &out_minus, 1, 3, 3, 1, 1, 2, 2, 1, 1, 0, 0);
        var loss_minus: f32 = 0;
        for (out_minus) |v| loss_minus += v;

        // Restore
        weight[i] += eps;

        // Numerical gradient
        const numerical_grad = (loss_plus - loss_minus) / (2 * eps);
        try std.testing.expectApproxEqAbs(numerical_grad, grad_weight[i], tol);
    }
}
