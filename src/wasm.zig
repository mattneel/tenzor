//! WASM module for LeNet-5 inference in the browser.
//!
//! Exports:
//!   - init(): Initialize model weights
//!   - predict(pixels: [*]f32) u8: Run inference on 28x28 grayscale image
//!
//! Build: zig build-exe src/wasm.zig -target wasm32-freestanding -O ReleaseSmall

const std = @import("std");

// ============================================================================
// Memory Management (freestanding)
// ============================================================================

// 1MB fixed buffer for all allocations
var buffer: [1024 * 1024]u8 = undefined;
var fba = std.heap.FixedBufferAllocator.init(&buffer);

// ============================================================================
// Model Weights (178KB total)
// ============================================================================

const Weights = struct {
    // Conv1: [6, 1, 5, 5] = 150 params
    conv1_weight: [6 * 1 * 5 * 5]f32 = undefined,
    conv1_bias: [6]f32 = undefined,

    // Conv2: [16, 6, 5, 5] = 2400 params
    conv2_weight: [16 * 6 * 5 * 5]f32 = undefined,
    conv2_bias: [16]f32 = undefined,

    // FC1: [120, 256] = 30720 params
    fc1_weight: [120 * 256]f32 = undefined,
    fc1_bias: [120]f32 = undefined,

    // FC2: [84, 120] = 10080 params
    fc2_weight: [84 * 120]f32 = undefined,
    fc2_bias: [84]f32 = undefined,

    // FC3: [10, 84] = 840 params
    fc3_weight: [10 * 84]f32 = undefined,
    fc3_bias: [10]f32 = undefined,
};

var weights: Weights = .{};
var initialized: bool = false;

// ============================================================================
// Activation Buffers
// ============================================================================

// Input: 1x28x28
var input: [1 * 28 * 28]f32 = undefined;

// After conv1 + relu: 6x24x24
var conv1_out: [6 * 24 * 24]f32 = undefined;

// After pool1: 6x12x12
var pool1_out: [6 * 12 * 12]f32 = undefined;

// After conv2 + relu: 16x8x8
var conv2_out: [16 * 8 * 8]f32 = undefined;

// After pool2: 16x4x4 = 256
var pool2_out: [16 * 4 * 4]f32 = undefined;

// FC outputs
var fc1_out: [120]f32 = undefined;
var fc2_out: [84]f32 = undefined;
var fc3_out: [10]f32 = undefined;

// ============================================================================
// Exported Functions
// ============================================================================

/// Initialize model - called once with pointer to weight data
export fn init(weight_data: [*]const u8, len: usize) void {
    _ = len;
    const data: [*]const f32 = @ptrCast(@alignCast(weight_data));
    var offset: usize = 0;

    // Conv1
    @memcpy(&weights.conv1_weight, data[offset..][0..150]);
    offset += 150;
    @memcpy(&weights.conv1_bias, data[offset..][0..6]);
    offset += 6;

    // Conv2
    @memcpy(&weights.conv2_weight, data[offset..][0..2400]);
    offset += 2400;
    @memcpy(&weights.conv2_bias, data[offset..][0..16]);
    offset += 16;

    // FC1
    @memcpy(&weights.fc1_weight, data[offset..][0..30720]);
    offset += 30720;
    @memcpy(&weights.fc1_bias, data[offset..][0..120]);
    offset += 120;

    // FC2
    @memcpy(&weights.fc2_weight, data[offset..][0..10080]);
    offset += 10080;
    @memcpy(&weights.fc2_bias, data[offset..][0..84]);
    offset += 84;

    // FC3
    @memcpy(&weights.fc3_weight, data[offset..][0..840]);
    offset += 840;
    @memcpy(&weights.fc3_bias, data[offset..][0..10]);

    initialized = true;
}

/// Get pointer to input buffer (for JS to write pixel data)
export fn getInputPtr() [*]f32 {
    return &input;
}

/// Run inference, return predicted digit 0-9
export fn predict() u8 {
    if (!initialized) return 255;

    // Conv1: 1x28x28 -> 6x24x24
    conv2d(&input, &weights.conv1_weight, &weights.conv1_bias, &conv1_out, 1, 6, 28, 28, 5);
    relu(&conv1_out);

    // Pool1: 6x24x24 -> 6x12x12
    maxpool2d(&conv1_out, &pool1_out, 6, 24, 24);

    // Conv2: 6x12x12 -> 16x8x8
    conv2d(&pool1_out, &weights.conv2_weight, &weights.conv2_bias, &conv2_out, 6, 16, 12, 12, 5);
    relu(&conv2_out);

    // Pool2: 16x8x8 -> 16x4x4
    maxpool2d(&conv2_out, &pool2_out, 16, 8, 8);

    // FC1: 256 -> 120
    linear(&pool2_out, &weights.fc1_weight, &weights.fc1_bias, &fc1_out, 256, 120);
    relu(&fc1_out);

    // FC2: 120 -> 84
    linear(&fc1_out, &weights.fc2_weight, &weights.fc2_bias, &fc2_out, 120, 84);
    relu(&fc2_out);

    // FC3: 84 -> 10
    linear(&fc2_out, &weights.fc3_weight, &weights.fc3_bias, &fc3_out, 84, 10);

    // Argmax
    return argmax(&fc3_out);
}

/// Get confidence for predicted class (0-100)
export fn getConfidence() f32 {
    // Softmax and return max probability
    var max_val: f32 = fc3_out[0];
    for (fc3_out[1..]) |v| {
        if (v > max_val) max_val = v;
    }

    var sum: f32 = 0;
    var probs: [10]f32 = undefined;
    for (fc3_out, 0..) |v, i| {
        probs[i] = @exp(v - max_val);
        sum += probs[i];
    }

    const pred = argmax(&fc3_out);
    return (probs[pred] / sum) * 100.0;
}

// ============================================================================
// Layer Implementations
// ============================================================================

fn conv2d(
    inp: []const f32,
    weight: []const f32,
    bias: []const f32,
    out: []f32,
    in_c: usize,
    out_c: usize,
    in_h: usize,
    in_w: usize,
    k: usize,
) void {
    const out_h = in_h - k + 1;
    const out_w = in_w - k + 1;

    // Input layout: [H, W, C] (HWC for single image)
    // Weight layout: [C_out, kH, kW, C_in]
    // Output layout: [H, W, C_out]

    for (0..out_h) |oh| {
        for (0..out_w) |ow| {
            for (0..out_c) |oc| {
                var sum: f32 = bias[oc];

                for (0..k) |kh| {
                    for (0..k) |kw| {
                        const ih = oh + kh;
                        const iw = ow + kw;

                        for (0..in_c) |ic| {
                            // HWC input layout
                            const inp_idx = (ih * in_w + iw) * in_c + ic;
                            // Weight layout: [out_c, kh, kw, in_c]
                            const w_idx = ((oc * k + kh) * k + kw) * in_c + ic;
                            sum += inp[inp_idx] * weight[w_idx];
                        }
                    }
                }

                // HWC output layout
                out[(oh * out_w + ow) * out_c + oc] = sum;
            }
        }
    }
}

fn maxpool2d(inp: []const f32, out: []f32, channels: usize, in_h: usize, in_w: usize) void {
    const out_h = in_h / 2;
    const out_w = in_w / 2;

    // HWC layout
    for (0..out_h) |oh| {
        for (0..out_w) |ow| {
            const ih = oh * 2;
            const iw = ow * 2;

            for (0..channels) |c| {
                // Sample 2x2 window in HWC layout
                const v00 = inp[(ih * in_w + iw) * channels + c];
                const v01 = inp[(ih * in_w + iw + 1) * channels + c];
                const v10 = inp[((ih + 1) * in_w + iw) * channels + c];
                const v11 = inp[((ih + 1) * in_w + iw + 1) * channels + c];

                out[(oh * out_w + ow) * channels + c] = @max(@max(v00, v01), @max(v10, v11));
            }
        }
    }
}

fn linear(inp: []const f32, weight: []const f32, bias: []const f32, out: []f32, in_features: usize, out_features: usize) void {
    for (0..out_features) |o| {
        var sum: f32 = bias[o];
        for (0..in_features) |i| {
            sum += inp[i] * weight[o * in_features + i];
        }
        out[o] = sum;
    }
}

fn relu(data: []f32) void {
    for (data) |*v| {
        if (v.* < 0) v.* = 0;
    }
}

fn argmax(data: []const f32) u8 {
    var max_idx: u8 = 0;
    var max_val: f32 = data[0];
    for (data[1..], 1..) |v, i| {
        if (v > max_val) {
            max_val = v;
            max_idx = @intCast(i);
        }
    }
    return max_idx;
}

// ============================================================================
// Panic handler for freestanding
// ============================================================================

pub fn panic(msg: []const u8, _: ?*std.builtin.StackTrace, _: ?usize) noreturn {
    _ = msg;
    while (true) {}
}
