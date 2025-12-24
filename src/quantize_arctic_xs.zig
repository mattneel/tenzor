const std = @import("std");
const safetensors = @import("io/safetensors.zig");

fn usage() void {
    std.debug.print(
        \\Quantize Arctic-embed-xs SafeTensors weights to symmetric per-row int8.
        \\
        \\Usage:
        \\  zig run src/quantize_arctic_xs.zig -- <input.safetensors> <output.safetensors>
        \\
        \\Output convention:
        \\  <name>          -> I8  (same name, 2D weights only)
        \\  <name>.scale    -> F16 (shape [out])
        \\
        \\Notes:
        \\  - Only quantizes large 2D tensors ending with ".weight" (excluding LayerNorm).
        \\  - Biases and norms remain F32.
        \\
    , .{});
}

fn shouldQuantize(name: []const u8, info: safetensors.TensorInfo) bool {
    if (info.dtype != .F32) return false;
    if (info.shape.len != 2) return false;
    if (!std.mem.endsWith(u8, name, ".weight")) return false;
    if (std.mem.indexOf(u8, name, "LayerNorm") != null) return false;

    const rows = info.shape[0];
    const cols = info.shape[1];
    if (rows == 0 or cols == 0) return false;

    const numel = rows * cols;
    return numel >= 4096;
}

fn readF32LE(bytes: []const u8, idx: usize) f32 {
    const off = idx * 4;
    const bits = std.mem.readInt(u32, bytes[off..][0..4], .little);
    return @bitCast(bits);
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena.deinit();
    const allocator = arena.allocator();

    var args = std.process.args();
    _ = args.next();
    const input_path = args.next() orelse {
        usage();
        return error.InvalidArgs;
    };
    const output_path = args.next() orelse {
        usage();
        return error.InvalidArgs;
    };
    if (args.next() != null) {
        usage();
        return error.InvalidArgs;
    }

    var loaded = try safetensors.load(allocator, input_path);
    defer loaded.st.deinit();

    var out_tensors: std.ArrayList(safetensors.TensorToWrite) = .empty;
    defer out_tensors.deinit(allocator);

    var quantized_count: usize = 0;
    var quantized_bytes: usize = 0;

    for (loaded.st.tensors) |info| {
        const name = info.name;
        const raw = loaded.st.data[loaded.st.header_size + 8 + info.data_start ..][0..info.byteSize()];

        if (!shouldQuantize(name, info)) {
            try out_tensors.append(allocator, .{
                .name = name,
                .dtype = info.dtype,
                .shape = info.shape,
                .data = raw,
            });
            continue;
        }

        const rows = info.shape[0];
        const cols = info.shape[1];

        const q = try allocator.alloc(i8, rows * cols);
        const scale = try allocator.alloc(f16, rows);

        for (0..rows) |r| {
            var max_abs: f32 = 0;
            const row_base = r * cols;
            for (0..cols) |c| {
                const v = readF32LE(raw, row_base + c);
                const a = @abs(v);
                if (a > max_abs) max_abs = a;
            }

            const s_f32: f32 = if (max_abs == 0) 1 else max_abs / 127.0;
            const s_f16: f16 = @floatCast(s_f32);
            scale[r] = s_f16;

            const inv_s: f32 = 1.0 / @as(f32, @floatCast(s_f16));
            for (0..cols) |c| {
                const v = readF32LE(raw, row_base + c);
                const qf: f32 = @round(v * inv_s);
                const qi: i32 = @intFromFloat(qf);
                const clamped: i32 = @min(127, @max(-127, qi));
                q[row_base + c] = @intCast(clamped);
            }
        }

        try out_tensors.append(allocator, .{
            .name = name,
            .dtype = .I8,
            .shape = info.shape,
            .data = std.mem.sliceAsBytes(q),
        });

        const scale_name = try std.fmt.allocPrint(allocator, "{s}.scale", .{name});
        const scale_shape = try allocator.alloc(usize, 1);
        scale_shape[0] = rows;

        try out_tensors.append(allocator, .{
            .name = scale_name,
            .dtype = .F16,
            .shape = scale_shape,
            .data = std.mem.sliceAsBytes(scale),
        });

        quantized_count += 1;
        quantized_bytes += q.len;
    }

    try safetensors.writeFile(allocator, output_path, out_tensors.items);

    std.debug.print(
        "Wrote {s} ({d} weights quantized, ~{d} MB int8)\n",
        .{ output_path, quantized_count, quantized_bytes / (1024 * 1024) },
    );
}
