//! 2D Convolution CPU kernel for CNN training.
//!
//! Implements forward and backward passes for 2D convolution:
//! - Forward: Y = conv2d(X, W) + b
//! - Backward: dX = conv2d_transpose(dY, W), dW = conv2d(X^T, dY), db = sum(dY)
//!
//! Layout: NHWC (batch, height, width, channels) - CPU/SIMD friendly
//!
//! Parallel versions available: conv2dForwardParallel, etc.

const std = @import("std");
const threading = @import("../threading.zig");

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
// Parallel versions (using ThreadPool)
// ============================================================================

/// Parallel 2D Convolution forward pass.
/// Parallelizes over the batch dimension.
pub fn conv2dForwardParallel(
    comptime T: type,
    pool: *threading.ThreadPool,
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
    const in_batch_size = in_h * in_w * in_c;
    const out_batch_size = out_h * out_w * out_c;

    const Context = struct {
        input: []const T,
        weight: []const T,
        bias: ?[]const T,
        output: []T,
        in_h: usize,
        in_w: usize,
        in_c: usize,
        out_c: usize,
        out_h: usize,
        out_w: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
        in_batch_size: usize,
        out_batch_size: usize,
    };

    const ctx = Context{
        .input = input,
        .weight = weight,
        .bias = bias,
        .output = output,
        .in_h = in_h,
        .in_w = in_w,
        .in_c = in_c,
        .out_c = out_c,
        .out_h = out_h,
        .out_w = out_w,
        .kernel_h = kernel_h,
        .kernel_w = kernel_w,
        .stride_h = stride_h,
        .stride_w = stride_w,
        .pad_h = pad_h,
        .pad_w = pad_w,
        .in_batch_size = in_batch_size,
        .out_batch_size = out_batch_size,
    };

    pool.parallelForBatch(batch, ctx, struct {
        fn work(c: Context, start: usize, end: usize) void {
            for (start..end) |n| {
                conv2dForwardSingle(
                    T,
                    c.input[n * c.in_batch_size ..][0..c.in_batch_size],
                    c.weight,
                    c.bias,
                    c.output[n * c.out_batch_size ..][0..c.out_batch_size],
                    c.in_h,
                    c.in_w,
                    c.in_c,
                    c.out_c,
                    c.out_h,
                    c.out_w,
                    c.kernel_h,
                    c.kernel_w,
                    c.stride_h,
                    c.stride_w,
                    c.pad_h,
                    c.pad_w,
                );
            }
        }
    }.work);
}

/// Single-sample forward pass (helper for parallel version).
fn conv2dForwardSingle(
    comptime T: type,
    input: []const T,
    weight: []const T,
    bias: ?[]const T,
    output: []T,
    in_h: usize,
    in_w: usize,
    in_c: usize,
    out_c: usize,
    out_h: usize,
    out_w: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) void {
    // Initialize output with bias (or zeros)
    if (bias) |b| {
        for (0..out_h) |oh| {
            for (0..out_w) |ow| {
                for (0..out_c) |oc| {
                    const out_idx = (oh * out_w + ow) * out_c + oc;
                    output[out_idx] = b[oc];
                }
            }
        }
    } else {
        @memset(output, 0);
    }

    // Convolution
    for (0..out_h) |oh| {
        for (0..out_w) |ow| {
            const ih_start = oh * stride_h;
            const iw_start = ow * stride_w;

            for (0..out_c) |oc| {
                var sum: T = 0;

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
                                const in_idx = (ih_actual * in_w + iw_actual) * in_c + ic;
                                const w_idx = ((oc * kernel_h + kh) * kernel_w + kw) * in_c + ic;
                                sum += input[in_idx] * weight[w_idx];
                            }
                        }
                    }
                }

                const out_idx = (oh * out_w + ow) * out_c + oc;
                output[out_idx] += sum;
            }
        }
    }
}

/// Parallel backward pass for input gradient.
pub fn conv2dBackwardInputParallel(
    comptime T: type,
    pool: *threading.ThreadPool,
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
    const in_batch_size = in_h * in_w * in_c;
    const out_batch_size = out_h * out_w * out_c;

    const Context = struct {
        grad_output: []const T,
        weight: []const T,
        grad_input: []T,
        in_h: usize,
        in_w: usize,
        in_c: usize,
        out_c: usize,
        out_h: usize,
        out_w: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
        in_batch_size: usize,
        out_batch_size: usize,
    };

    const ctx = Context{
        .grad_output = grad_output,
        .weight = weight,
        .grad_input = grad_input,
        .in_h = in_h,
        .in_w = in_w,
        .in_c = in_c,
        .out_c = out_c,
        .out_h = out_h,
        .out_w = out_w,
        .kernel_h = kernel_h,
        .kernel_w = kernel_w,
        .stride_h = stride_h,
        .stride_w = stride_w,
        .pad_h = pad_h,
        .pad_w = pad_w,
        .in_batch_size = in_batch_size,
        .out_batch_size = out_batch_size,
    };

    pool.parallelForBatch(batch, ctx, struct {
        fn work(c: Context, start: usize, end: usize) void {
            for (start..end) |n| {
                conv2dBackwardInputSingle(
                    T,
                    c.grad_output[n * c.out_batch_size ..][0..c.out_batch_size],
                    c.weight,
                    c.grad_input[n * c.in_batch_size ..][0..c.in_batch_size],
                    c.in_h,
                    c.in_w,
                    c.in_c,
                    c.out_c,
                    c.out_h,
                    c.out_w,
                    c.kernel_h,
                    c.kernel_w,
                    c.stride_h,
                    c.stride_w,
                    c.pad_h,
                    c.pad_w,
                );
            }
        }
    }.work);
}

/// Single-sample backward input (helper for parallel version).
fn conv2dBackwardInputSingle(
    comptime T: type,
    grad_output: []const T,
    weight: []const T,
    grad_input: []T,
    in_h: usize,
    in_w: usize,
    in_c: usize,
    out_c: usize,
    out_h: usize,
    out_w: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) void {
    @memset(grad_input, 0);

    for (0..out_h) |oh| {
        for (0..out_w) |ow| {
            const ih_start = oh * stride_h;
            const iw_start = ow * stride_w;

            for (0..out_c) |oc| {
                const grad_out_idx = (oh * out_w + ow) * out_c + oc;
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
                                const in_idx = (ih_actual * in_w + iw_actual) * in_c + ic;
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

/// Parallel backward pass for weight gradient.
/// Note: Weight gradients are accumulated across batches, so we parallelize
/// over batches and use thread-local accumulators then reduce.
pub fn conv2dBackwardWeightParallel(
    comptime T: type,
    pool: *threading.ThreadPool,
    allocator: std.mem.Allocator,
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
) !void {
    const out_h = convOutputSize(in_h, kernel_h, stride_h, pad_h);
    const out_w = convOutputSize(in_w, kernel_w, stride_w, pad_w);
    const in_batch_size = in_h * in_w * in_c;
    const out_batch_size = out_h * out_w * out_c;
    const weight_size = out_c * kernel_h * kernel_w * in_c;

    // For small batches, use sequential to avoid allocation overhead
    if (batch < pool.getThreadCount() * 2) {
        conv2dBackwardWeight(
            T,
            input,
            grad_output,
            grad_weight,
            grad_bias,
            batch,
            in_h,
            in_w,
            in_c,
            out_c,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
        );
        return;
    }

    // Allocate per-thread accumulators
    const num_threads = pool.getThreadCount();
    const thread_grad_weights = try allocator.alloc([]T, num_threads);
    defer allocator.free(thread_grad_weights);

    const thread_grad_biases: ?[][]T = if (grad_bias != null)
        try allocator.alloc([]T, num_threads)
    else
        null;
    defer if (thread_grad_biases) |tgb| allocator.free(tgb);

    // Initialize thread-local buffers
    for (0..num_threads) |t| {
        thread_grad_weights[t] = try allocator.alloc(T, weight_size);
        @memset(thread_grad_weights[t], 0);
        if (thread_grad_biases) |tgb| {
            tgb[t] = try allocator.alloc(T, out_c);
            @memset(tgb[t], 0);
        }
    }
    defer for (0..num_threads) |t| {
        allocator.free(thread_grad_weights[t]);
        if (thread_grad_biases) |tgb| allocator.free(tgb[t]);
    };

    const Context = struct {
        input: []const T,
        grad_output: []const T,
        thread_grad_weights: [][]T,
        thread_grad_biases: ?[][]T,
        in_h: usize,
        in_w: usize,
        in_c: usize,
        out_c: usize,
        out_h: usize,
        out_w: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
        in_batch_size: usize,
        out_batch_size: usize,
        num_threads: usize,
        batch: usize,
    };

    const ctx = Context{
        .input = input,
        .grad_output = grad_output,
        .thread_grad_weights = thread_grad_weights,
        .thread_grad_biases = thread_grad_biases,
        .in_h = in_h,
        .in_w = in_w,
        .in_c = in_c,
        .out_c = out_c,
        .out_h = out_h,
        .out_w = out_w,
        .kernel_h = kernel_h,
        .kernel_w = kernel_w,
        .stride_h = stride_h,
        .stride_w = stride_w,
        .pad_h = pad_h,
        .pad_w = pad_w,
        .in_batch_size = in_batch_size,
        .out_batch_size = out_batch_size,
        .num_threads = num_threads,
        .batch = batch,
    };

    // Each thread processes a range of batches
    pool.parallelForBatch(num_threads, ctx, struct {
        fn work(c: Context, start: usize, end: usize) void {
            _ = end;
            const thread_id = start;
            const batches_per_thread = (c.batch + c.num_threads - 1) / c.num_threads;
            const batch_start = thread_id * batches_per_thread;
            const batch_end = @min(batch_start + batches_per_thread, c.batch);

            const my_grad_weight = c.thread_grad_weights[thread_id];
            const my_grad_bias: ?[]T = if (c.thread_grad_biases) |tgb| tgb[thread_id] else null;

            for (batch_start..batch_end) |n| {
                conv2dBackwardWeightSingle(
                    T,
                    c.input[n * c.in_batch_size ..][0..c.in_batch_size],
                    c.grad_output[n * c.out_batch_size ..][0..c.out_batch_size],
                    my_grad_weight,
                    my_grad_bias,
                    c.in_h,
                    c.in_w,
                    c.in_c,
                    c.out_c,
                    c.out_h,
                    c.out_w,
                    c.kernel_h,
                    c.kernel_w,
                    c.stride_h,
                    c.stride_w,
                    c.pad_h,
                    c.pad_w,
                );
            }
        }
    }.work);

    // Reduce thread-local gradients into output
    @memset(grad_weight, 0);
    if (grad_bias) |gb| @memset(gb, 0);

    for (0..num_threads) |t| {
        for (grad_weight, thread_grad_weights[t]) |*gw, tgw| {
            gw.* += tgw;
        }
        if (grad_bias) |gb| {
            if (thread_grad_biases) |tgb| {
                for (gb, tgb[t]) |*b, tb| {
                    b.* += tb;
                }
            }
        }
    }
}

/// Single-sample backward weight (helper for parallel version).
fn conv2dBackwardWeightSingle(
    comptime T: type,
    input: []const T,
    grad_output: []const T,
    grad_weight: []T,
    grad_bias: ?[]T,
    in_h: usize,
    in_w: usize,
    in_c: usize,
    out_c: usize,
    out_h: usize,
    out_w: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) void {
    for (0..out_h) |oh| {
        for (0..out_w) |ow| {
            const ih_start = oh * stride_h;
            const iw_start = ow * stride_w;

            for (0..out_c) |oc| {
                const grad_out_idx = (oh * out_w + ow) * out_c + oc;
                const grad_out_val = grad_output[grad_out_idx];

                if (grad_bias) |gb| {
                    gb[oc] += grad_out_val;
                }

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
                                const in_idx = (ih_actual * in_w + iw_actual) * in_c + ic;
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
