//! ONNX Graph Executor
//!
//! Executes ONNX computation graphs using tenzor's optimized kernels.
//! Tensors have runtime-determined shapes (unlike tenzor's comptime shapes).
//!
//! NOTE: This is a work-in-progress implementation. Many operations are incomplete
//! or have known issues with shape handling, broadcasting, and edge cases.
//! Not yet suitable for production use.

const std = @import("std");
const builtin = @import("builtin");

/// Debug assertions for catching bugs during development
const debug = struct {
    /// Assert tensor shape is valid (all dimensions positive, product matches numel)
    fn assertValidTensor(tensor: RuntimeTensor) void {
        if (builtin.mode == .Debug) {
            var expected_numel: usize = 1;
            for (tensor.shape) |dim| {
                std.debug.assert(dim >= 0); // No negative dimensions
                expected_numel *= @intCast(dim);
            }
            std.debug.assert(tensor.numel == expected_numel);
            std.debug.assert(tensor.data.len >= tensor.numel * tensor.dtype.byteSize());
        }
    }

    /// Assert axis is valid for given number of dimensions
    fn assertValidAxis(axis: i64, ndim: usize) void {
        if (builtin.mode == .Debug) {
            const ndim_i64: i64 = @intCast(ndim);
            std.debug.assert(axis >= -ndim_i64 and axis < ndim_i64);
        }
    }

    /// Assert index is within bounds
    fn assertInBounds(index: usize, len: usize) void {
        if (builtin.mode == .Debug) {
            std.debug.assert(index < len);
        }
    }

    /// Assert shapes are broadcastable
    fn assertBroadcastable(lhs: []const i64, rhs: []const i64) void {
        if (builtin.mode == .Debug) {
            var out_buf: [8]i64 = undefined;
            const result = Executor.computeBroadcastShape(lhs, rhs, &out_buf);
            std.debug.assert(result != null); // Shapes must be broadcastable
        }
    }
};
const Allocator = std.mem.Allocator;
const graph_mod = @import("graph.zig");
const Graph = graph_mod.Graph;
const Node = graph_mod.Node;
const OpType = graph_mod.OpType;
const DType = graph_mod.DType;
const WeightData = graph_mod.WeightData;

// Import tenzor's optimized kernels
const tenzor = @import("../root.zig");
const TenzorOpTag = tenzor.ops.expr.OpTag;
const tenzor_elementwise = tenzor.backend.cpu.kernels.elementwise;
const tenzor_matmul = tenzor.backend.cpu.kernels.matmul;
const tenzor_reduce = tenzor.backend.cpu.kernels.reduce;
const tenzor_gather = tenzor.backend.cpu.kernels.gather;
const tenzor_softmax = tenzor.backend.cpu.kernels.softmax;
const tenzor_layernorm = tenzor.backend.cpu.kernels.layernorm;
const tenzor_transpose = tenzor.backend.cpu.kernels.transpose;

// Wrapper to use tenzor kernels with ONNX runtime types
const kernels = struct {
    const elementwise = struct {
        // Map to tenzor's OpTag for kernel dispatch
        const OpTag = enum {
            add,
            sub,
            mul,
            div,
            max,
            min,
            neg,
            abs,
            exp,
            log,
            sqrt,
            pow,
            sin,
            cos,
            ceil,
            floor,
            round,
            relu,
            leaky_relu,
            elu,
            sigmoid,
            tanh,
            gelu,
            silu,
            softplus,
            erf,
        };

        fn unaryOp(comptime op: OpTag, comptime T: type, input: []const T, output: []T) void {
            for (input, 0..) |x, i| {
                output[i] = switch (op) {
                    .neg => -x,
                    .abs => @abs(x),
                    .exp => @exp(x),
                    .log => @log(x),
                    .sqrt => @sqrt(x),
                    .sin => @sin(x),
                    .cos => @cos(x),
                    .ceil => @ceil(x),
                    .floor => @floor(x),
                    .round => @round(x),
                    .relu => @max(x, 0),
                    .leaky_relu => if (x > 0) x else x * 0.01, // default alpha=0.01
                    .elu => if (x > 0) x else @exp(x) - 1,
                    .sigmoid => 1.0 / (1.0 + @exp(-x)),
                    .tanh => std.math.tanh(x),
                    .gelu => x * 0.5 * (1.0 + std.math.tanh(0.7978845608 * (x + 0.044715 * x * x * x))),
                    .silu => x / (1.0 + @exp(-x)),
                    .softplus => @log(1.0 + @exp(x)),
                    // erf approximation using Horner's form
                    .erf => blk: {
                        const a1: f32 = 0.254829592;
                        const a2: f32 = -0.284496736;
                        const a3: f32 = 1.421413741;
                        const a4: f32 = -1.453152027;
                        const a5: f32 = 1.061405429;
                        const p: f32 = 0.3275911;
                        const sign: f32 = if (x < 0) -1 else 1;
                        const ax = @abs(x);
                        const t = 1.0 / (1.0 + p * ax);
                        const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * @exp(-ax * ax);
                        break :blk sign * y;
                    },
                    else => x,
                };
            }
        }

        fn divTyped(comptime T: type, a: T, b: T) T {
            if (@typeInfo(T) == .int) {
                return @divTrunc(a, b);
            } else {
                return a / b;
            }
        }

        fn binaryOp(comptime op: OpTag, comptime T: type, lhs: []const T, rhs: []const T, output: []T) void {
            for (lhs, rhs, 0..) |a, b, i| {
                output[i] = switch (op) {
                    .add => a + b,
                    .sub => a - b,
                    .mul => a * b,
                    .div => divTyped(T, a, b),
                    .max => @max(a, b),
                    .min => @min(a, b),
                    else => a,
                };
            }
        }

        fn binaryOpScalarRhs(comptime op: OpTag, comptime T: type, lhs: []const T, rhs_scalar: T, output: []T) void {
            for (lhs, 0..) |a, i| {
                output[i] = switch (op) {
                    .add => a + rhs_scalar,
                    .sub => a - rhs_scalar,
                    .mul => a * rhs_scalar,
                    .div => divTyped(T, a, rhs_scalar),
                    .max => @max(a, rhs_scalar),
                    .min => @min(a, rhs_scalar),
                    else => a,
                };
            }
        }

        fn binaryOpScalarLhs(comptime op: OpTag, comptime T: type, lhs_scalar: T, rhs: []const T, output: []T) void {
            for (rhs, 0..) |b, i| {
                output[i] = switch (op) {
                    .add => lhs_scalar + b,
                    .sub => lhs_scalar - b,
                    .mul => lhs_scalar * b,
                    .div => divTyped(T, lhs_scalar, b),
                    .max => @max(lhs_scalar, b),
                    .min => @min(lhs_scalar, b),
                    else => b,
                };
            }
        }
    };

    // Delegate to tenzor's optimized SIMD/BLAS matmul kernels
    const matmul = struct {
        fn matmulTiled(comptime T: type, a: []const T, b: []const T, c: []T, m: usize, k: usize, n: usize) void {
            tenzor_matmul.matmulTiled(T, a, b, c, m, k, n);
        }

        fn batchedMatmul(comptime T: type, a: []const T, b: []const T, c: []T, batch: usize, m: usize, k: usize, n: usize) void {
            tenzor_matmul.batchedMatmul(T, a, b, c, batch, m, k, n);
        }

        fn batchedMatmulBroadcastB(comptime T: type, a: []const T, b: []const T, c: []T, batch: usize, m: usize, k: usize, n: usize) void {
            tenzor_matmul.batchedMatmulBroadcastB(T, a, b, c, batch, m, k, n);
        }
    };

    const transpose = struct {
        fn transpose(comptime T: type, input: []const T, output: []T, in_shape: []const usize, out_shape: []const usize, perm: []const usize) void {
            _ = out_shape;
            const ndim = in_shape.len;

            // Calculate strides
            var in_strides: [8]usize = undefined;
            var out_strides: [8]usize = undefined;
            in_strides[ndim - 1] = 1;
            out_strides[ndim - 1] = 1;

            var i: usize = ndim - 1;
            while (i > 0) : (i -= 1) {
                in_strides[i - 1] = in_strides[i] * in_shape[i];
                out_strides[i - 1] = out_strides[i] * in_shape[perm[i]];
            }

            // Compute permuted strides
            var perm_strides: [8]usize = undefined;
            for (0..ndim) |pi| {
                perm_strides[pi] = in_strides[perm[pi]];
            }

            // Perform transpose
            const numel = @min(input.len, output.len);
            for (0..numel) |out_idx| {
                var in_idx: usize = 0;
                var remaining = out_idx;
                for (0..ndim) |d| {
                    const coord = remaining / out_strides[d];
                    remaining = remaining % out_strides[d];
                    in_idx += coord * perm_strides[d];
                }
                if (in_idx < input.len and out_idx < output.len) {
                    output[out_idx] = input[in_idx];
                }
            }
        }
    };
};

/// Runtime tensor with dynamically-sized shape
pub const RuntimeTensor = struct {
    /// Raw data buffer
    data: []u8,
    /// Data type
    dtype: DType,
    /// Shape dimensions
    shape: []const i64,
    /// Number of elements
    numel: usize,
    /// Allocator used for this tensor
    allocator: Allocator,
    /// Whether we own the data (vs borrowed from weights)
    owns_data: bool,
    /// Alignment of data buffer (for proper freeing)
    data_alignment: u8 = 1,

    /// Create a new tensor with allocated storage
    pub fn alloc(allocator: Allocator, dtype: DType, shape: []const i64) !RuntimeTensor {
        var numel: usize = 1;
        for (shape) |dim| {
            if (dim < 0) return error.InvalidShape;
            numel *= @intCast(dim);
        }

        const byte_size = numel * dtype.byteSize();
        const alignment: u8 = @intCast(dtype.byteSize());

        // Allocate with proper alignment for dtype
        const data: []u8 = blk: {
            if (byte_size == 0) {
                break :blk try allocator.alloc(u8, 0);
            }
            switch (alignment) {
                8 => {
                    const ptr = try allocator.alloc(u64, (byte_size + 7) / 8);
                    break :blk @as([*]u8, @ptrCast(ptr.ptr))[0..byte_size];
                },
                4 => {
                    const ptr = try allocator.alloc(u32, (byte_size + 3) / 4);
                    break :blk @as([*]u8, @ptrCast(ptr.ptr))[0..byte_size];
                },
                2 => {
                    const ptr = try allocator.alloc(u16, (byte_size + 1) / 2);
                    break :blk @as([*]u8, @ptrCast(ptr.ptr))[0..byte_size];
                },
                else => {
                    break :blk try allocator.alloc(u8, byte_size);
                },
            }
        };
        if (byte_size > 0) @memset(data, 0);

        const owned_shape = try allocator.alloc(i64, shape.len);
        @memcpy(owned_shape, shape);

        return .{
            .data = data,
            .dtype = dtype,
            .shape = owned_shape,
            .numel = numel,
            .allocator = allocator,
            .owns_data = true,
            .data_alignment = alignment,
        };
    }

    /// Create a tensor from weight data (copies to ensure alignment)
    pub fn fromWeightData(allocator: Allocator, weight: WeightData) !RuntimeTensor {
        var numel: usize = 1;
        for (weight.shape) |dim| {
            if (dim <= 0) return error.InvalidShape;
            numel *= @intCast(dim);
        }

        // Copy shape
        const owned_shape = try allocator.alloc(i64, weight.shape.len);
        @memcpy(owned_shape, weight.shape);

        // Copy data - use allocRaw to ensure proper alignment for the dtype
        // alloc(u8) has alignment 1, but we need proper alignment for typed access
        const byte_size = weight.dtype.byteSize();
        const owned_data = switch (byte_size) {
            8 => blk: {
                const aligned = try allocator.alloc(u64, (weight.data.len + 7) / 8);
                const as_bytes: []u8 = @as([*]u8, @ptrCast(aligned.ptr))[0..weight.data.len];
                @memcpy(as_bytes, weight.data);
                break :blk as_bytes;
            },
            4 => blk: {
                const aligned = try allocator.alloc(u32, (weight.data.len + 3) / 4);
                const as_bytes: []u8 = @as([*]u8, @ptrCast(aligned.ptr))[0..weight.data.len];
                @memcpy(as_bytes, weight.data);
                break :blk as_bytes;
            },
            2 => blk: {
                const aligned = try allocator.alloc(u16, (weight.data.len + 1) / 2);
                const as_bytes: []u8 = @as([*]u8, @ptrCast(aligned.ptr))[0..weight.data.len];
                @memcpy(as_bytes, weight.data);
                break :blk as_bytes;
            },
            else => blk: {
                const data = try allocator.alloc(u8, weight.data.len);
                @memcpy(data, weight.data);
                break :blk data;
            },
        };

        return .{
            .data = owned_data,
            .dtype = weight.dtype,
            .shape = owned_shape,
            .numel = numel,
            .allocator = allocator,
            .owns_data = true,
            .data_alignment = @intCast(byte_size),
        };
    }

    /// Create from typed slice (copies data)
    pub fn fromSlice(allocator: Allocator, comptime T: type, data: []const T, shape: []const i64) !RuntimeTensor {
        const dtype = DType.fromZigType(T) orelse return error.UnsupportedType;

        var numel: usize = 1;
        for (shape) |dim| {
            if (dim < 0) return error.InvalidShape;
            numel *= @intCast(dim);
        }

        if (numel != data.len) return error.ShapeMismatch;

        const byte_size = numel * dtype.byteSize();
        const alignment: u8 = @intCast(@sizeOf(T));

        // Allocate with proper alignment for type T
        const buffer: []u8 = blk: {
            if (byte_size == 0) {
                break :blk try allocator.alloc(u8, 0);
            }
            switch (alignment) {
                8 => {
                    const ptr = try allocator.alloc(u64, (byte_size + 7) / 8);
                    break :blk @as([*]u8, @ptrCast(ptr.ptr))[0..byte_size];
                },
                4 => {
                    const ptr = try allocator.alloc(u32, (byte_size + 3) / 4);
                    break :blk @as([*]u8, @ptrCast(ptr.ptr))[0..byte_size];
                },
                2 => {
                    const ptr = try allocator.alloc(u16, (byte_size + 1) / 2);
                    break :blk @as([*]u8, @ptrCast(ptr.ptr))[0..byte_size];
                },
                else => {
                    break :blk try allocator.alloc(u8, byte_size);
                },
            }
        };
        if (byte_size > 0) {
            const src_bytes: []const u8 = @as([*]const u8, @ptrCast(data.ptr))[0..byte_size];
            @memcpy(buffer, src_bytes);
        }

        const owned_shape = try allocator.alloc(i64, shape.len);
        @memcpy(owned_shape, shape);

        return .{
            .data = buffer,
            .dtype = dtype,
            .shape = owned_shape,
            .numel = numel,
            .allocator = allocator,
            .owns_data = true,
            .data_alignment = alignment,
        };
    }

    pub fn deinit(self: *RuntimeTensor) void {
        if (self.owns_data) {
            // Empty data was allocated with u8 regardless of data_alignment
            if (self.data.len == 0) {
                self.allocator.free(self.data);
            } else {
                // Free with correct alignment based on how it was allocated
                switch (self.data_alignment) {
                    8 => {
                        const ptr: [*]u64 = @ptrCast(@alignCast(self.data.ptr));
                        const aligned_len = (self.data.len + 7) / 8;
                        self.allocator.free(ptr[0..aligned_len]);
                    },
                    4 => {
                        const ptr: [*]u32 = @ptrCast(@alignCast(self.data.ptr));
                        const aligned_len = (self.data.len + 3) / 4;
                        self.allocator.free(ptr[0..aligned_len]);
                    },
                    2 => {
                        const ptr: [*]u16 = @ptrCast(@alignCast(self.data.ptr));
                        const aligned_len = (self.data.len + 1) / 2;
                        self.allocator.free(ptr[0..aligned_len]);
                    },
                    else => self.allocator.free(self.data),
                }
            }
        }
        self.allocator.free(@constCast(self.shape));
        self.* = undefined;
    }

    /// Get data as typed slice
    pub fn asSlice(self: *const RuntimeTensor, comptime T: type) ?[]T {
        const expected_dtype = DType.fromZigType(T) orelse return null;
        // Allow u8 to match bool_ (both are 1 byte)
        const matches = (self.dtype == expected_dtype) or
            (T == u8 and self.dtype == .bool_);
        if (!matches) return null;

        // Return empty slice for empty tensors without checking alignment
        if (self.numel == 0) {
            return @as([*]T, undefined)[0..0];
        }

        const ptr: [*]T = @ptrCast(@alignCast(self.data.ptr));
        return ptr[0..self.numel];
    }

    /// Get data as const typed slice
    pub fn asConstSlice(self: *const RuntimeTensor, comptime T: type) ?[]const T {
        const expected_dtype = DType.fromZigType(T) orelse return null;
        // Allow u8 to match bool_ (both are 1 byte)
        const matches = (self.dtype == expected_dtype) or
            (T == u8 and self.dtype == .bool_);
        if (!matches) return null;

        // Return empty slice for empty tensors without checking alignment
        if (self.numel == 0) {
            return @as([*]const T, undefined)[0..0];
        }

        const ptr: [*]const T = @ptrCast(@alignCast(self.data.ptr));
        return ptr[0..self.numel];
    }

    /// Get number of dimensions
    pub fn ndim(self: *const RuntimeTensor) usize {
        return self.shape.len;
    }

    /// Clone this tensor
    pub fn clone(self: *const RuntimeTensor) !RuntimeTensor {
        const byte_size = self.data.len;
        const alignment = self.data_alignment;

        // Allocate with proper alignment based on original tensor
        const new_data: []u8 = blk: {
            if (byte_size == 0) {
                break :blk try self.allocator.alloc(u8, 0);
            }
            switch (alignment) {
                8 => {
                    const ptr = try self.allocator.alloc(u64, (byte_size + 7) / 8);
                    break :blk @as([*]u8, @ptrCast(ptr.ptr))[0..byte_size];
                },
                4 => {
                    const ptr = try self.allocator.alloc(u32, (byte_size + 3) / 4);
                    break :blk @as([*]u8, @ptrCast(ptr.ptr))[0..byte_size];
                },
                2 => {
                    const ptr = try self.allocator.alloc(u16, (byte_size + 1) / 2);
                    break :blk @as([*]u8, @ptrCast(ptr.ptr))[0..byte_size];
                },
                else => {
                    break :blk try self.allocator.alloc(u8, byte_size);
                },
            }
        };
        @memcpy(new_data, self.data);

        const new_shape = try self.allocator.alloc(i64, self.shape.len);
        @memcpy(new_shape, self.shape);

        return .{
            .data = new_data,
            .dtype = self.dtype,
            .shape = new_shape,
            .numel = self.numel,
            .allocator = self.allocator,
            .owns_data = true,
            .data_alignment = alignment,
        };
    }
};

/// ONNX Graph Executor
pub const Executor = struct {
    /// The graph being executed
    graph: *const Graph,
    /// Allocator for tensor buffers
    allocator: Allocator,
    /// Tensor buffers indexed by graph tensor ID
    buffers: []?RuntimeTensor,

    pub fn init(allocator: Allocator, graph: *const Graph) !Executor {
        const num_tensors = graph.tensors.items.len;
        const buffers = try allocator.alloc(?RuntimeTensor, num_tensors);
        @memset(buffers, null);

        return .{
            .graph = graph,
            .allocator = allocator,
            .buffers = buffers,
        };
    }

    pub fn deinit(self: *Executor) void {
        for (self.buffers) |*maybe_tensor| {
            if (maybe_tensor.*) |*tensor| {
                tensor.deinit();
            }
        }
        self.allocator.free(self.buffers);
        self.* = undefined;
    }

    /// Load weights from the graph into buffers
    pub fn loadWeights(self: *Executor) !void {
        var it = self.graph.weights.iterator();
        while (it.next()) |entry| {
            const name = entry.key_ptr.*;
            const weight = entry.value_ptr.*;

            // Skip weights with no data (external data not loaded)
            if (weight.data.len == 0) continue;

            if (self.graph.getTensorIndex(name)) |idx| {
                self.buffers[idx] = try RuntimeTensor.fromWeightData(self.allocator, weight);
            }
        }
    }

    /// Load external weights from files
    /// base_dir: Directory containing the .onnx file (external paths are relative to this)
    pub fn loadExternalWeights(self: *Executor, base_dir: []const u8) !void {
        var it = self.graph.weights.iterator();
        while (it.next()) |entry| {
            const name = entry.key_ptr.*;
            const weight = entry.value_ptr.*;

            // Only process weights with external data
            if (!weight.isExternal()) continue;

            const location = weight.external_location orelse continue;

            // Get tensor index
            const idx = self.graph.getTensorIndex(name) orelse continue;

            // Skip weights with no shape info
            if (weight.shape.len == 0) continue;

            // Calculate expected size from shape if not specified
            var numel: usize = 1;
            for (weight.shape) |dim| {
                if (dim <= 0) return error.InvalidShape;
                numel *= @intCast(dim);
            }
            const expected_size = numel * weight.dtype.byteSize();
            const read_length = if (weight.external_length > 0) weight.external_length else expected_size;

            // Build full path (base_dir + location)
            var path_buf: [4096]u8 = undefined;
            const full_path = std.fmt.bufPrint(&path_buf, "{s}/{s}", .{ base_dir, location }) catch return error.PathTooLong;

            // Open the external file
            const file = std.fs.cwd().openFile(full_path, .{}) catch |err| {
                std.debug.print("Failed to open external weight file '{s}': {}\n", .{ full_path, err });
                return error.ExternalDataNotFound;
            };
            defer file.close();

            // Seek to offset if specified
            if (weight.external_offset > 0) {
                file.seekTo(weight.external_offset) catch return error.ExternalDataSeekFailed;
            }

            // Allocate aligned buffer for the data
            const byte_size = weight.dtype.byteSize();
            const owned_data = switch (byte_size) {
                8 => blk: {
                    const aligned = try self.allocator.alloc(u64, (read_length + 7) / 8);
                    break :blk @as([*]u8, @ptrCast(aligned.ptr))[0..read_length];
                },
                4 => blk: {
                    const aligned = try self.allocator.alloc(u32, (read_length + 3) / 4);
                    break :blk @as([*]u8, @ptrCast(aligned.ptr))[0..read_length];
                },
                2 => blk: {
                    const aligned = try self.allocator.alloc(u16, (read_length + 1) / 2);
                    break :blk @as([*]u8, @ptrCast(aligned.ptr))[0..read_length];
                },
                else => try self.allocator.alloc(u8, read_length),
            };
            errdefer self.allocator.free(owned_data);

            // Read the data
            var total_read: usize = 0;
            while (total_read < read_length) {
                const bytes_read = file.read(owned_data[total_read..]) catch return error.ExternalDataReadFailed;
                if (bytes_read == 0) break;
                total_read += bytes_read;
            }

            if (total_read != read_length) {
                return error.ExternalDataIncomplete;
            }

            // Copy shape
            const owned_shape = try self.allocator.alloc(i64, weight.shape.len);
            @memcpy(owned_shape, weight.shape);

            // Create the runtime tensor
            self.buffers[idx] = .{
                .data = owned_data,
                .dtype = weight.dtype,
                .shape = owned_shape,
                .numel = numel,
                .allocator = self.allocator,
                .owns_data = true,
                .data_alignment = @intCast(byte_size),
            };
        }
    }

    /// Set an input tensor by name
    pub fn setInput(self: *Executor, name: []const u8, tensor: RuntimeTensor) !void {
        const idx = self.graph.getTensorIndex(name) orelse return error.TensorNotFound;
        if (self.buffers[idx]) |*existing| {
            existing.deinit();
        }
        self.buffers[idx] = tensor;
    }

    /// Set input from a typed slice
    pub fn setInputFromSlice(
        self: *Executor,
        name: []const u8,
        comptime T: type,
        data: []const T,
        shape: []const i64,
    ) !void {
        const tensor = try RuntimeTensor.fromSlice(self.allocator, T, data, shape);
        try self.setInput(name, tensor);
    }

    /// Get an output tensor by name
    pub fn getOutput(self: *Executor, name: []const u8) ?*RuntimeTensor {
        const idx = self.graph.getTensorIndex(name) orelse return null;
        if (self.buffers[idx]) |*tensor| {
            return tensor;
        }
        return null;
    }

    /// Execute all nodes in the graph (topologically sorted)
    pub fn run(self: *Executor) !void {
        // Build dependency graph: for each node, track which output tensor it produces
        // and which input tensors it needs
        const nodes = self.graph.nodes.items;
        const n = nodes.len;

        // Track which tensors are produced by which node index
        var tensor_producer = try self.allocator.alloc(?usize, self.buffers.len);
        defer self.allocator.free(tensor_producer);
        @memset(tensor_producer, null);

        // Mark tensors that are already available (inputs/weights)
        for (self.buffers, 0..) |buf, i| {
            if (buf != null) {
                tensor_producer[i] = n; // Special value: already available
            }
        }

        // Map each node's outputs to the node index
        for (nodes, 0..) |node, i| {
            for (node.outputs) |out_idx| {
                tensor_producer[out_idx] = i;
            }
        }

        // Track which nodes have been executed
        var executed = try self.allocator.alloc(bool, n);
        defer self.allocator.free(executed);
        @memset(executed, false);

        // Execute nodes in dependency order
        var remaining = n;
        while (remaining > 0) {
            var progress = false;

            for (nodes, 0..) |node, i| {
                if (executed[i]) continue;

                // Check if all inputs are ready
                var ready = true;
                for (node.inputs) |inp_idx| {
                    const producer = tensor_producer[inp_idx];
                    if (producer) |prod_idx| {
                        if (prod_idx != n and !executed[prod_idx]) {
                            ready = false;
                            break;
                        }
                    } else {
                        // Input tensor not produced by any node and not pre-loaded
                        ready = false;
                        break;
                    }
                }

                if (ready) {
                    self.executeNode(node) catch |err| {
                        std.debug.print("Error executing node {d} ({s}): {}\n", .{ i, @tagName(node.op_type), err });
                        // Print input shapes for debugging
                        for (node.inputs, 0..) |inp_idx, j| {
                            const name = if (inp_idx < self.graph.tensors.items.len)
                                self.graph.tensors.items[inp_idx].name
                            else
                                "?";
                            std.debug.print("  Input {d} ({s}): ", .{ j, name });
                            if (self.buffers[inp_idx]) |buf| {
                                std.debug.print("dtype={s} shape=[", .{@tagName(buf.dtype)});
                                for (buf.shape, 0..) |dim, k| {
                                    if (k > 0) std.debug.print(",", .{});
                                    std.debug.print("{}", .{dim});
                                }
                                std.debug.print("]\n", .{});
                            } else {
                                std.debug.print("null\n", .{});
                            }
                        }
                        return err;
                    };
                    executed[i] = true;
                    remaining -= 1;
                    progress = true;
                }
            }

            if (!progress and remaining > 0) {
                // Cyclic dependency or missing input
                for (nodes, 0..) |node, i| {
                    if (!executed[i]) {
                        std.debug.print("Cannot execute node {d} ({s}): missing inputs\n", .{ i, @tagName(node.op_type) });
                        for (node.inputs) |inp_idx| {
                            if (self.buffers[inp_idx] == null) {
                                const name = if (inp_idx < self.graph.tensors.items.len)
                                    self.graph.tensors.items[inp_idx].name
                                else
                                    "?";
                                std.debug.print("  Missing: {s}\n", .{name});
                            }
                        }
                        break;
                    }
                }
                return error.MissingInput;
            }
        }
    }

    /// Execute all nodes with debug output
    pub fn runDebug(self: *Executor) !void {
        // Same dependency logic as run(), but with debug output
        const nodes = self.graph.nodes.items;
        const n = nodes.len;

        var tensor_producer = try self.allocator.alloc(?usize, self.buffers.len);
        defer self.allocator.free(tensor_producer);
        @memset(tensor_producer, null);

        for (self.buffers, 0..) |buf, i| {
            if (buf != null) {
                tensor_producer[i] = n;
            }
        }

        for (nodes, 0..) |node, i| {
            for (node.outputs) |out_idx| {
                tensor_producer[out_idx] = i;
            }
        }

        var executed = try self.allocator.alloc(bool, n);
        defer self.allocator.free(executed);
        @memset(executed, false);

        var exec_order: usize = 0;
        var remaining = n;
        while (remaining > 0) {
            var progress = false;

            for (nodes, 0..) |node, i| {
                if (executed[i]) continue;

                var ready = true;
                for (node.inputs) |inp_idx| {
                    const producer = tensor_producer[inp_idx];
                    if (producer) |prod_idx| {
                        if (prod_idx != n and !executed[prod_idx]) {
                            ready = false;
                            break;
                        }
                    } else {
                        ready = false;
                        break;
                    }
                }

                if (ready) {
                    // Print debug info before execution
                    std.debug.print("[{d}] {s}", .{ exec_order, @tagName(node.op_type) });
                    std.debug.print(" inputs: ", .{});
                    for (node.inputs, 0..) |inp_idx, j| {
                        if (j > 0) std.debug.print(", ", .{});
                        if (self.buffers[inp_idx]) |buf| {
                            std.debug.print("[", .{});
                            for (buf.shape, 0..) |dim, k| {
                                if (k > 0) std.debug.print(",", .{});
                                std.debug.print("{}", .{dim});
                            }
                            std.debug.print("]", .{});
                        } else {
                            const name = if (inp_idx < self.graph.tensors.items.len)
                                self.graph.tensors.items[inp_idx].name
                            else
                                "?";
                            std.debug.print("null({s})", .{name});
                        }
                    }

                    try self.executeNode(node);

                    // Print output shapes
                    std.debug.print(" -> ", .{});
                    for (node.outputs, 0..) |out_idx, j| {
                        if (j > 0) std.debug.print(", ", .{});
                        if (self.buffers[out_idx]) |buf| {
                            std.debug.print("[", .{});
                            for (buf.shape, 0..) |dim, k| {
                                if (k > 0) std.debug.print(",", .{});
                                std.debug.print("{}", .{dim});
                            }
                            std.debug.print("]", .{});
                        } else {
                            std.debug.print("null", .{});
                        }
                    }
                    std.debug.print("\n", .{});

                    executed[i] = true;
                    remaining -= 1;
                    progress = true;
                    exec_order += 1;
                }
            }

            if (!progress and remaining > 0) {
                for (nodes, 0..) |node, i| {
                    if (!executed[i]) {
                        std.debug.print("Cannot execute node {d} ({s}): missing inputs\n", .{ i, @tagName(node.op_type) });
                        for (node.inputs) |inp_idx| {
                            if (self.buffers[inp_idx] == null) {
                                const name = if (inp_idx < self.graph.tensors.items.len)
                                    self.graph.tensors.items[inp_idx].name
                                else
                                    "?";
                                std.debug.print("  Missing: {s}\n", .{name});
                            }
                        }
                        break;
                    }
                }
                return error.MissingInput;
            }
        }
    }

    /// Execute a single node
    fn executeNode(self: *Executor, node: Node) !void {
        switch (node.op_type) {
            .Add => try self.execBinaryOp(node, .add),
            .Sub => try self.execBinaryOp(node, .sub),
            .Mul => try self.execBinaryOp(node, .mul),
            .Div => try self.execBinaryOp(node, .div),
            .Max => try self.execBinaryOp(node, .max),
            .Min => try self.execBinaryOp(node, .min),
            .Neg => try self.execUnaryOp(node, .neg),
            .Abs => try self.execUnaryOp(node, .abs),
            .Exp => try self.execUnaryOp(node, .exp),
            .Log => try self.execUnaryOp(node, .log),
            .Sqrt => try self.execUnaryOp(node, .sqrt),
            .Relu => try self.execUnaryOp(node, .relu),
            .Sigmoid => try self.execUnaryOp(node, .sigmoid),
            .Tanh => try self.execUnaryOp(node, .tanh),
            .Gelu => try self.execUnaryOp(node, .gelu),
            .Silu => try self.execUnaryOp(node, .silu),
            .Sin => try self.execUnaryOp(node, .sin),
            .Cos => try self.execUnaryOp(node, .cos),
            .Ceil => try self.execUnaryOp(node, .ceil),
            .Floor => try self.execUnaryOp(node, .floor),
            .Round => try self.execUnaryOp(node, .round),
            .LeakyRelu => try self.execUnaryOp(node, .leaky_relu),
            .Elu => try self.execUnaryOp(node, .elu),
            .Softplus => try self.execUnaryOp(node, .softplus),
            .Erf => try self.execUnaryOp(node, .erf),
            .Pow => try self.execPow(node),
            .MatMul => try self.execMatMul(node),
            .Gather => try self.execGather(node),
            .Concat => try self.execConcat(node),
            .Reshape => try self.execReshape(node),
            .Transpose => try self.execTranspose(node),
            .Softmax => try self.execSoftmax(node),
            .Slice => try self.execSlice(node),
            .Equal => try self.execEqual(node),
            .Where => try self.execWhere(node),
            .Constant => try self.execConstant(node),
            .Shape => try self.execShape(node),
            .Cast => try self.execCast(node),
            .ReduceSum => try self.execReduceSum(node),
            .LayerNormalization => try self.execLayerNorm(node),
            .GroupQueryAttention => try self.execGroupQueryAttention(node),
            .MultiHeadAttention => try self.execMultiHeadAttention(node),
            .MatMulNBits => try self.execMatMulNBits(node),
            .GatherBlockQuantized => try self.execGatherBlockQuantized(node),
            .Unsqueeze => try self.execUnsqueeze(node),
            .Squeeze => try self.execSqueeze(node),
            .Expand => try self.execExpand(node),
            .ConstantOfShape => try self.execConstantOfShape(node),
            .Range => try self.execRange(node),
            .Clip => try self.execClip(node),
            .Less => try self.execCompare(node, .less),
            .Greater => try self.execCompare(node, .greater),
            .GreaterOrEqual => try self.execCompare(node, .greater_or_equal),
            .LessOrEqual => try self.execCompare(node, .less_or_equal),
            .Not => try self.execNot(node),
            .ReduceMean => try self.execReduceMean(node),
            .ReduceMax => try self.execReduceMax(node),
            .ReduceProd => try self.execReduceProd(node),
            .ReduceL2 => try self.execReduceL2(node),
            .Tile => try self.execTile(node),
            .Gemm => try self.execGemm(node),
            .Pad => try self.execPad(node),
            .STFT => try self.execSTFT(node),
            .Conv => try self.execConv(node),
            .NonZero => try self.execNonZero(node),
            .MaxPool => try self.execMaxPool(node),
            .AveragePool => try self.execAveragePool(node),
            .GlobalAveragePool => try self.execGlobalAveragePool(node),
            .BatchNormalization => try self.execBatchNorm(node),
            .Flatten => try self.execFlatten(node),
            .Split => try self.execSplit(node),
            else => {
                std.debug.print("Unsupported op: {s}\n", .{node.op_type_str});
                return error.UnsupportedOp;
            },
        }
    }

    /// Execute a unary elementwise operation
    fn execUnaryOp(self: *Executor, node: Node, comptime op: kernels.elementwise.OpTag) !void {
        if (node.inputs.len < 1 or node.outputs.len < 1) return error.InvalidNode;

        const input = self.buffers[node.inputs[0]] orelse return error.MissingInput;
        debug.assertValidTensor(input);
        const output_idx = node.outputs[0];

        // Allocate output with same shape
        var output = try RuntimeTensor.alloc(self.allocator, input.dtype, input.shape);
        errdefer output.deinit();

        // Dispatch by dtype
        switch (input.dtype) {
            .f32 => {
                const in_data = input.asConstSlice(f32).?;
                const out_data = output.asSlice(f32).?;
                kernels.elementwise.unaryOp(op, f32, in_data, out_data);
            },
            .f64 => {
                const in_data = input.asConstSlice(f64).?;
                const out_data = output.asSlice(f64).?;
                kernels.elementwise.unaryOp(op, f64, in_data, out_data);
            },
            .f16 => {
                // Convert f16 -> f32, apply op, convert back
                const in_f16 = input.asConstSlice(f16).?;
                const out_f16 = output.asSlice(f16).?;
                // Allocate temp buffers
                const temp_in = try self.allocator.alloc(f32, in_f16.len);
                defer self.allocator.free(temp_in);
                const temp_out = try self.allocator.alloc(f32, out_f16.len);
                defer self.allocator.free(temp_out);
                // Convert input
                for (in_f16, temp_in) |src, *dst| dst.* = @floatCast(src);
                // Apply op
                kernels.elementwise.unaryOp(op, f32, temp_in, temp_out);
                // Convert output
                for (temp_out, out_f16) |src, *dst| dst.* = @floatCast(src);
            },
            else => return error.UnsupportedDType,
        }

        if (self.buffers[output_idx]) |*existing| {
            existing.deinit();
        }
        self.buffers[output_idx] = output;
    }

    /// Compute broadcast output shape for two tensors
    /// Returns null if shapes are not broadcastable
    fn computeBroadcastShape(lhs_shape: []const i64, rhs_shape: []const i64, out_buf: *[8]i64) ?[]i64 {
        const max_ndim = @max(lhs_shape.len, rhs_shape.len);
        if (max_ndim > 8) return null;

        for (0..max_ndim) |i| {
            const l_idx = if (i < lhs_shape.len) lhs_shape.len - 1 - i else null;
            const r_idx = if (i < rhs_shape.len) rhs_shape.len - 1 - i else null;
            const l_dim: i64 = if (l_idx) |idx| lhs_shape[idx] else 1;
            const r_dim: i64 = if (r_idx) |idx| rhs_shape[idx] else 1;

            // Dimensions must be equal or one must be 1
            if (l_dim != r_dim and l_dim != 1 and r_dim != 1) return null;
            out_buf[max_ndim - 1 - i] = @max(l_dim, r_dim);
        }
        return out_buf[0..max_ndim];
    }

    /// Execute a binary elementwise operation
    fn execBinaryOp(self: *Executor, node: Node, comptime op: kernels.elementwise.OpTag) !void {
        if (node.inputs.len < 2 or node.outputs.len < 1) return error.InvalidNode;

        const lhs = self.buffers[node.inputs[0]] orelse return error.MissingInput;
        const rhs = self.buffers[node.inputs[1]] orelse return error.MissingInput;
        debug.assertValidTensor(lhs);
        debug.assertValidTensor(rhs);
        const output_idx = node.outputs[0];

        // Compute broadcast output shape
        var out_shape_buf: [8]i64 = undefined;
        const out_shape = computeBroadcastShape(lhs.shape, rhs.shape, &out_shape_buf) orelse
            return error.ShapeMismatch;

        // Handle empty tensors
        if (lhs.numel == 0 or rhs.numel == 0) {
            const output = try RuntimeTensor.alloc(self.allocator, lhs.dtype, out_shape);
            if (self.buffers[output_idx]) |*existing| {
                existing.deinit();
            }
            self.buffers[output_idx] = output;
            return;
        }

        // Fast path: same shape, no broadcasting needed
        if (lhs.numel == rhs.numel and std.mem.eql(i64, lhs.shape, rhs.shape)) {
            return self.execBinaryOpSameShape(node, op, &lhs, &rhs, output_idx);
        }

        // Scalar optimization
        if (rhs.numel == 1) {
            return self.execBinaryOpScalarRhs(node, op, &lhs, &rhs, output_idx);
        }
        if (lhs.numel == 1) {
            return self.execBinaryOpScalarLhs(node, op, &lhs, &rhs, output_idx);
        }

        // General broadcasting case
        return self.execBinaryOpBroadcast(op, &lhs, &rhs, out_shape, output_idx);
    }

    /// Execute binary op with same shapes (fast path)
    fn execBinaryOpSameShape(
        self: *Executor,
        node: Node,
        comptime op: kernels.elementwise.OpTag,
        lhs: *const RuntimeTensor,
        rhs: *const RuntimeTensor,
        output_idx: u32,
    ) !void {
        _ = node;

        // Handle mixed dtype cases - promote to f32 if types don't match
        const out_dtype = if (lhs.dtype == rhs.dtype)
            lhs.dtype
        else if ((lhs.dtype == .f16 and rhs.dtype == .f32) or (lhs.dtype == .f32 and rhs.dtype == .f16))
            .f32
        else
            lhs.dtype;

        var output = try RuntimeTensor.alloc(self.allocator, out_dtype, lhs.shape);
        errdefer output.deinit();

        // If dtypes match, use fast path
        if (lhs.dtype == rhs.dtype) {
            switch (lhs.dtype) {
                inline .f32, .f64, .f16, .i32, .i64 => |dtype| {
                    const T = dtype.ZigType();
                    const lhs_data = lhs.asConstSlice(T).?;
                    const rhs_data = rhs.asConstSlice(T).?;
                    const out_data = output.asSlice(T).?;
                    kernels.elementwise.binaryOp(op, T, lhs_data, rhs_data, out_data);
                },
                else => return error.UnsupportedDType,
            }
        } else {
            // Mixed dtype: convert both to f32, compute, result is f32
            const lhs_f32 = if (lhs.dtype == .f32)
                lhs.asConstSlice(f32).?
            else if (lhs.dtype == .f16) blk: {
                const h = lhs.asConstSlice(f16).?;
                const f = try self.allocator.alloc(f32, h.len);
                for (h, f) |v, *o| o.* = @floatCast(v);
                break :blk f;
            } else return error.UnsupportedDType;
            defer if (lhs.dtype == .f16) self.allocator.free(lhs_f32);

            const rhs_f32 = if (rhs.dtype == .f32)
                rhs.asConstSlice(f32).?
            else if (rhs.dtype == .f16) blk: {
                const h = rhs.asConstSlice(f16).?;
                const f = try self.allocator.alloc(f32, h.len);
                for (h, f) |v, *o| o.* = @floatCast(v);
                break :blk f;
            } else return error.UnsupportedDType;
            defer if (rhs.dtype == .f16) self.allocator.free(rhs_f32);

            const out_data = output.asSlice(f32).?;
            kernels.elementwise.binaryOp(op, f32, lhs_f32, rhs_f32, out_data);
        }

        if (self.buffers[output_idx]) |*existing| {
            existing.deinit();
        }
        self.buffers[output_idx] = output;
    }

    /// Execute binary op with broadcasting (general case)
    fn execBinaryOpBroadcast(
        self: *Executor,
        comptime op: kernels.elementwise.OpTag,
        lhs: *const RuntimeTensor,
        rhs: *const RuntimeTensor,
        out_shape: []const i64,
        output_idx: u32,
    ) !void {
        // Determine output dtype (promote f16 + f32 to f32)
        const out_dtype: DType = if (lhs.dtype == rhs.dtype)
            lhs.dtype
        else if ((lhs.dtype == .f16 and rhs.dtype == .f32) or (lhs.dtype == .f32 and rhs.dtype == .f16))
            .f32
        else
            lhs.dtype;

        var output = try RuntimeTensor.alloc(self.allocator, out_dtype, out_shape);
        errdefer output.deinit();

        // Compute strides for input tensors (with broadcasting)
        // For broadcasting, stride is 0 when dimension is 1
        var lhs_strides: [8]usize = undefined;
        var rhs_strides: [8]usize = undefined;
        var out_strides: [8]usize = undefined;
        const ndim = out_shape.len;

        // Compute output strides (row-major)
        var stride: usize = 1;
        var i: usize = ndim;
        while (i > 0) {
            i -= 1;
            out_strides[i] = stride;
            stride *= @intCast(out_shape[i]);
        }

        // Compute lhs strides with broadcasting
        stride = 1;
        i = lhs.shape.len;
        while (i > 0) {
            i -= 1;
            const out_idx = ndim - (lhs.shape.len - i);
            if (lhs.shape[i] == 1) {
                lhs_strides[out_idx] = 0; // Broadcast: stride 0
            } else {
                lhs_strides[out_idx] = stride;
                stride *= @intCast(lhs.shape[i]);
            }
        }
        // Fill remaining leading dims with 0 (implicit broadcast from shape 1)
        for (0..ndim - lhs.shape.len) |j| {
            lhs_strides[j] = 0;
        }

        // Compute rhs strides with broadcasting
        stride = 1;
        i = rhs.shape.len;
        while (i > 0) {
            i -= 1;
            const out_idx = ndim - (rhs.shape.len - i);
            if (rhs.shape[i] == 1) {
                rhs_strides[out_idx] = 0; // Broadcast: stride 0
            } else {
                rhs_strides[out_idx] = stride;
                stride *= @intCast(rhs.shape[i]);
            }
        }
        // Fill remaining leading dims with 0
        for (0..ndim - rhs.shape.len) |j| {
            rhs_strides[j] = 0;
        }

        // Execute with type dispatch
        switch (out_dtype) {
            inline .f32, .f16, .i32, .i64 => |dtype| {
                const T = dtype.ZigType();
                try self.execBroadcastTyped(T, op, lhs, rhs, &output, out_shape, lhs_strides[0..ndim], rhs_strides[0..ndim], out_strides[0..ndim]);
            },
            else => return error.UnsupportedDType,
        }

        if (self.buffers[output_idx]) |*existing| {
            existing.deinit();
        }
        self.buffers[output_idx] = output;
    }

    /// Type-specific broadcast execution
    fn execBroadcastTyped(
        self: *Executor,
        comptime T: type,
        comptime op: kernels.elementwise.OpTag,
        lhs: *const RuntimeTensor,
        rhs: *const RuntimeTensor,
        output: *RuntimeTensor,
        out_shape: []const i64,
        lhs_strides: []const usize,
        rhs_strides: []const usize,
        out_strides: []const usize,
    ) !void {
        _ = out_strides;

        // Get data slices, converting dtypes if needed
        const lhs_data = try self.getDataAsType(T, lhs);
        defer if (lhs.dtype != comptime DType.fromZigType(T)) self.allocator.free(lhs_data);

        const rhs_data = try self.getDataAsType(T, rhs);
        defer if (rhs.dtype != comptime DType.fromZigType(T)) self.allocator.free(rhs_data);

        const out_data = output.asSlice(T).?;
        const ndim = out_shape.len;

        // Iterate over output elements
        var coords: [8]usize = .{0} ** 8;
        for (out_data, 0..) |*out, out_idx| {
            _ = out_idx;

            // Compute input indices using strides
            var lhs_idx: usize = 0;
            var rhs_idx: usize = 0;
            for (0..ndim) |d| {
                lhs_idx += coords[d] * lhs_strides[d];
                rhs_idx += coords[d] * rhs_strides[d];
            }

            // Apply operation
            const a = lhs_data[lhs_idx];
            const b = rhs_data[rhs_idx];
            out.* = applyBinaryOp(T, op, a, b);

            // Increment coordinates (row-major order)
            var d: usize = ndim;
            while (d > 0) {
                d -= 1;
                coords[d] += 1;
                if (coords[d] < @as(usize, @intCast(out_shape[d]))) break;
                coords[d] = 0;
            }
        }
    }

    /// Get tensor data as specific type, converting if needed
    fn getDataAsType(self: *Executor, comptime T: type, tensor: *const RuntimeTensor) ![]const T {
        const target_dtype = comptime DType.fromZigType(T);
        if (tensor.dtype == target_dtype) {
            return tensor.asConstSlice(T).?;
        }

        // Need to convert
        const result = try self.allocator.alloc(T, tensor.numel);
        errdefer self.allocator.free(result);

        if (@typeInfo(T) == .float) {
            if (tensor.asConstSlice(f32)) |src| {
                for (src, result) |v, *o| o.* = @floatCast(v);
                return result;
            }
            if (tensor.asConstSlice(f16)) |src| {
                for (src, result) |v, *o| o.* = @floatCast(v);
                return result;
            }
            if (tensor.asConstSlice(i32)) |src| {
                for (src, result) |v, *o| o.* = @floatFromInt(v);
                return result;
            }
            if (tensor.asConstSlice(i64)) |src| {
                for (src, result) |v, *o| o.* = @floatFromInt(v);
                return result;
            }
        }
        return error.UnsupportedDType;
    }

    /// Apply binary operation to two values
    fn applyBinaryOp(comptime T: type, comptime op: kernels.elementwise.OpTag, a: T, b: T) T {
        return switch (op) {
            .add => a + b,
            .sub => a - b,
            .mul => a * b,
            .div => if (@typeInfo(T) == .int) @divTrunc(a, b) else a / b,
            .max => @max(a, b),
            .min => @min(a, b),
            .pow => if (@typeInfo(T) == .float) std.math.pow(T, a, b) else a,
            // Unary ops - shouldn't be called via binary broadcast path
            else => unreachable,
        };
    }

    fn execBinaryOpScalarRhs(
        self: *Executor,
        node: Node,
        comptime op: kernels.elementwise.OpTag,
        lhs: *const RuntimeTensor,
        rhs: *const RuntimeTensor,
        output_idx: u32,
    ) !void {
        _ = node;
        var output = try RuntimeTensor.alloc(self.allocator, lhs.dtype, lhs.shape);
        errdefer output.deinit();

        switch (lhs.dtype) {
            inline .f32, .f64, .f16, .i32, .i64 => |dtype| {
                const T = dtype.ZigType();
                const lhs_data = lhs.asConstSlice(T).?;
                // Get scalar value, possibly converting from rhs dtype
                const rhs_scalar: T = blk: {
                    if (rhs.asConstSlice(T)) |slice| {
                        break :blk slice[0];
                    }
                    // Try to convert from other dtypes
                    if (@typeInfo(T) == .float) {
                        if (rhs.asConstSlice(f32)) |s| break :blk @floatCast(s[0]);
                        if (rhs.asConstSlice(f64)) |s| break :blk @floatCast(s[0]);
                        if (rhs.asConstSlice(f16)) |s| break :blk @floatCast(s[0]);
                        if (rhs.asConstSlice(i32)) |s| break :blk @floatFromInt(s[0]);
                        if (rhs.asConstSlice(i64)) |s| break :blk @floatFromInt(s[0]);
                    } else {
                        if (rhs.asConstSlice(i32)) |s| break :blk @intCast(s[0]);
                        if (rhs.asConstSlice(i64)) |s| break :blk @intCast(s[0]);
                    }
                    return error.UnsupportedDType;
                };
                const out_data = output.asSlice(T).?;
                kernels.elementwise.binaryOpScalarRhs(op, T, lhs_data, rhs_scalar, out_data);
            },
            else => return error.UnsupportedDType,
        }

        if (self.buffers[output_idx]) |*existing| {
            existing.deinit();
        }
        self.buffers[output_idx] = output;
    }

    fn execBinaryOpScalarLhs(
        self: *Executor,
        node: Node,
        comptime op: kernels.elementwise.OpTag,
        lhs: *const RuntimeTensor,
        rhs: *const RuntimeTensor,
        output_idx: u32,
    ) !void {
        _ = node;
        var output = try RuntimeTensor.alloc(self.allocator, rhs.dtype, rhs.shape);
        errdefer output.deinit();

        switch (rhs.dtype) {
            inline .f32, .f64, .f16, .i32, .i64 => |dtype| {
                const T = dtype.ZigType();
                // Get scalar value, possibly converting from lhs dtype
                const lhs_scalar: T = blk: {
                    if (lhs.asConstSlice(T)) |slice| {
                        break :blk slice[0];
                    }
                    // Try to convert from other dtypes
                    if (@typeInfo(T) == .float) {
                        if (lhs.asConstSlice(f32)) |s| break :blk @floatCast(s[0]);
                        if (lhs.asConstSlice(f64)) |s| break :blk @floatCast(s[0]);
                        if (lhs.asConstSlice(f16)) |s| break :blk @floatCast(s[0]);
                        if (lhs.asConstSlice(i32)) |s| break :blk @floatFromInt(s[0]);
                        if (lhs.asConstSlice(i64)) |s| break :blk @floatFromInt(s[0]);
                    } else {
                        if (lhs.asConstSlice(i32)) |s| break :blk @intCast(s[0]);
                        if (lhs.asConstSlice(i64)) |s| break :blk @intCast(s[0]);
                    }
                    return error.UnsupportedDType;
                };
                const rhs_data = rhs.asConstSlice(T).?;
                const out_data = output.asSlice(T).?;
                kernels.elementwise.binaryOpScalarLhs(op, T, lhs_scalar, rhs_data, out_data);
            },
            else => return error.UnsupportedDType,
        }

        if (self.buffers[output_idx]) |*existing| {
            existing.deinit();
        }
        self.buffers[output_idx] = output;
    }

    /// Execute MatMul operation
    fn execMatMul(self: *Executor, node: Node) !void {
        if (node.inputs.len < 2 or node.outputs.len < 1) return error.InvalidNode;

        const a = self.buffers[node.inputs[0]] orelse return error.MissingInput;
        const b = self.buffers[node.inputs[1]] orelse return error.MissingInput;
        const output_idx = node.outputs[0];

        // Determine output shape based on input shapes
        const a_shape = a.shape;
        const b_shape = b.shape;

        // Check for zero-dimension tensors (empty inputs produce empty outputs)
        var has_zero_dim = false;
        for (a_shape) |d| {
            if (d == 0) has_zero_dim = true;
        }
        for (b_shape) |d| {
            if (d == 0) has_zero_dim = true;
        }
        if (has_zero_dim) {
            // Compute output shape with zero dimensions preserved
            var out_shape_buf: [8]i64 = undefined;
            const out_ndim = @max(a_shape.len, b_shape.len);

            // For empty tensor, output shape is empty too
            if (a_shape.len >= 2 and b_shape.len >= 2) {
                // Normal matmul output shape: [batch..., M, N]
                const m = a_shape[a_shape.len - 2];
                const n = b_shape[b_shape.len - 1];
                if (out_ndim == 2) {
                    out_shape_buf[0] = m;
                    out_shape_buf[1] = n;
                } else {
                    // Copy batch dims
                    for (0..out_ndim - 2) |i| {
                        if (i < a_shape.len - 2) {
                            out_shape_buf[i] = a_shape[i];
                        } else {
                            out_shape_buf[i] = 1;
                        }
                    }
                    out_shape_buf[out_ndim - 2] = m;
                    out_shape_buf[out_ndim - 1] = n;
                }
            } else {
                // Fallback - preserve input shape
                for (a_shape, 0..) |d, i| out_shape_buf[i] = d;
            }

            const output = try RuntimeTensor.alloc(self.allocator, a.dtype, out_shape_buf[0..out_ndim]);
            if (self.buffers[output_idx]) |*existing| existing.deinit();
            self.buffers[output_idx] = output;
            return;
        }

        // Handle scalar case - treat as element-wise multiply
        if (b_shape.len == 0 or (b_shape.len == 1 and b_shape[0] == 1)) {
            // B is a scalar - do element-wise multiplication
            var output = try RuntimeTensor.alloc(self.allocator, .f32, a_shape);
            errdefer output.deinit();

            // Get scalar value
            var scalar: f32 = 1.0;
            if (b.asConstSlice(f32)) |bf32| {
                if (bf32.len > 0) scalar = bf32[0];
            } else if (b.asConstSlice(f16)) |bf16| {
                if (bf16.len > 0) scalar = @floatCast(bf16[0]);
            }

            // Get input data as f32
            if (a.asConstSlice(f32)) |af32| {
                const out_data = output.asSlice(f32).?;
                for (af32, 0..) |v, i| {
                    out_data[i] = v * scalar;
                }
            } else if (a.asConstSlice(f16)) |af16| {
                const out_data = output.asSlice(f32).?;
                for (af16, 0..) |v, i| {
                    out_data[i] = @as(f32, @floatCast(v)) * scalar;
                }
            } else {
                return error.UnsupportedDType;
            }

            if (self.buffers[output_idx]) |*existing| existing.deinit();
            self.buffers[output_idx] = output;
            return;
        }

        if (a_shape.len < 1 or b_shape.len < 1) return error.InvalidShape;

        // 2D case: [M, K] @ [K, N] -> [M, N]
        if (a_shape.len == 2 and b_shape.len == 2) {
            const m: usize = @intCast(a_shape[0]);
            const k: usize = @intCast(a_shape[1]);
            const n: usize = @intCast(b_shape[1]);

            if (k != @as(usize, @intCast(b_shape[0]))) return error.ShapeMismatch;

            const out_shape = [_]i64{ @intCast(m), @intCast(n) };
            var output = try RuntimeTensor.alloc(self.allocator, a.dtype, &out_shape);
            errdefer output.deinit();

            switch (a.dtype) {
                .f32 => {
                    const a_data = a.asConstSlice(f32).?;
                    const b_data = b.asConstSlice(f32).?;
                    const out_data = output.asSlice(f32).?;
                    kernels.matmul.matmulTiled(f32, a_data, b_data, out_data, m, k, n);
                },
                .f64 => {
                    const a_data = a.asConstSlice(f64).?;
                    const b_data = b.asConstSlice(f64).?;
                    const out_data = output.asSlice(f64).?;
                    kernels.matmul.matmulTiled(f64, a_data, b_data, out_data, m, k, n);
                },
                else => return error.UnsupportedDType,
            }

            if (self.buffers[output_idx]) |*existing| {
                existing.deinit();
            }
            self.buffers[output_idx] = output;
            return;
        }

        // 3D batched: [B, M, K] @ [B, K, N] -> [B, M, N]
        if (a_shape.len == 3 and b_shape.len == 3) {
            const batch: usize = @intCast(a_shape[0]);
            const m: usize = @intCast(a_shape[1]);
            const k: usize = @intCast(a_shape[2]);
            const n: usize = @intCast(b_shape[2]);

            if (batch != @as(usize, @intCast(b_shape[0]))) return error.ShapeMismatch;
            if (k != @as(usize, @intCast(b_shape[1]))) return error.ShapeMismatch;

            const out_shape = [_]i64{ @intCast(batch), @intCast(m), @intCast(n) };
            var output = try RuntimeTensor.alloc(self.allocator, a.dtype, &out_shape);
            errdefer output.deinit();

            switch (a.dtype) {
                .f32 => {
                    const a_data = a.asConstSlice(f32).?;
                    const b_data = b.asConstSlice(f32).?;
                    const out_data = output.asSlice(f32).?;
                    kernels.matmul.batchedMatmul(f32, a_data, b_data, out_data, batch, m, k, n);
                },
                .f64 => {
                    const a_data = a.asConstSlice(f64).?;
                    const b_data = b.asConstSlice(f64).?;
                    const out_data = output.asSlice(f64).?;
                    kernels.matmul.batchedMatmul(f64, a_data, b_data, out_data, batch, m, k, n);
                },
                else => return error.UnsupportedDType,
            }

            if (self.buffers[output_idx]) |*existing| {
                existing.deinit();
            }
            self.buffers[output_idx] = output;
            return;
        }

        // 3D x 2D broadcast: [B, M, K] @ [K, N] -> [B, M, N]
        if (a_shape.len == 3 and b_shape.len == 2) {
            const batch: usize = @intCast(a_shape[0]);
            const m: usize = @intCast(a_shape[1]);
            const k: usize = @intCast(a_shape[2]);
            const n: usize = @intCast(b_shape[1]);

            if (k != @as(usize, @intCast(b_shape[0]))) return error.ShapeMismatch;

            const out_shape = [_]i64{ @intCast(batch), @intCast(m), @intCast(n) };
            var output = try RuntimeTensor.alloc(self.allocator, a.dtype, &out_shape);
            errdefer output.deinit();

            switch (a.dtype) {
                .f32 => {
                    const a_data = a.asConstSlice(f32).?;
                    const b_data = b.asConstSlice(f32).?;
                    const out_data = output.asSlice(f32).?;
                    // Batch matmul with broadcast B
                    for (0..batch) |bi| {
                        const a_batch = a_data[bi * m * k ..][0 .. m * k];
                        const out_batch = out_data[bi * m * n ..][0 .. m * n];
                        kernels.matmul.matmulTiled(f32, a_batch, b_data, out_batch, m, k, n);
                    }
                },
                .f64 => {
                    const a_data = a.asConstSlice(f64).?;
                    const b_data = b.asConstSlice(f64).?;
                    const out_data = output.asSlice(f64).?;
                    for (0..batch) |bi| {
                        const a_batch = a_data[bi * m * k ..][0 .. m * k];
                        const out_batch = out_data[bi * m * n ..][0 .. m * n];
                        kernels.matmul.matmulTiled(f64, a_batch, b_data, out_batch, m, k, n);
                    }
                },
                else => return error.UnsupportedDType,
            }

            if (self.buffers[output_idx]) |*existing| {
                existing.deinit();
            }
            self.buffers[output_idx] = output;
            return;
        }

        // 2D x 3D broadcast: [M, K] @ [B, K, N] -> [B, M, N]
        if (a_shape.len == 2 and b_shape.len == 3) {
            const batch: usize = @intCast(b_shape[0]);
            const m: usize = @intCast(a_shape[0]);
            const k: usize = @intCast(a_shape[1]);
            const n: usize = @intCast(b_shape[2]);

            if (k != @as(usize, @intCast(b_shape[1]))) return error.ShapeMismatch;

            const out_shape = [_]i64{ @intCast(batch), @intCast(m), @intCast(n) };
            var output = try RuntimeTensor.alloc(self.allocator, a.dtype, &out_shape);
            errdefer output.deinit();

            switch (a.dtype) {
                .f32 => {
                    const a_data = a.asConstSlice(f32).?;
                    const b_data = b.asConstSlice(f32).?;
                    const out_data = output.asSlice(f32).?;
                    for (0..batch) |bi| {
                        const b_batch = b_data[bi * k * n ..][0 .. k * n];
                        const out_batch = out_data[bi * m * n ..][0 .. m * n];
                        kernels.matmul.matmulTiled(f32, a_data, b_batch, out_batch, m, k, n);
                    }
                },
                .f64 => {
                    const a_data = a.asConstSlice(f64).?;
                    const b_data = b.asConstSlice(f64).?;
                    const out_data = output.asSlice(f64).?;
                    for (0..batch) |bi| {
                        const b_batch = b_data[bi * k * n ..][0 .. k * n];
                        const out_batch = out_data[bi * m * n ..][0 .. m * n];
                        kernels.matmul.matmulTiled(f64, a_data, b_batch, out_batch, m, k, n);
                    }
                },
                else => return error.UnsupportedDType,
            }

            if (self.buffers[output_idx]) |*existing| {
                existing.deinit();
            }
            self.buffers[output_idx] = output;
            return;
        }

        // 4D batched: [B1, B2, M, K] @ [B1, B2, K, N] -> [B1, B2, M, N]
        if (a_shape.len == 4 and b_shape.len == 4) {
            const b1: usize = @intCast(a_shape[0]);
            const b2: usize = @intCast(a_shape[1]);
            const m: usize = @intCast(a_shape[2]);
            const k: usize = @intCast(a_shape[3]);
            const n: usize = @intCast(b_shape[3]);

            if (b1 != @as(usize, @intCast(b_shape[0]))) return error.ShapeMismatch;
            if (b2 != @as(usize, @intCast(b_shape[1]))) return error.ShapeMismatch;
            if (k != @as(usize, @intCast(b_shape[2]))) return error.ShapeMismatch;

            const out_shape = [_]i64{ @intCast(b1), @intCast(b2), @intCast(m), @intCast(n) };
            var output = try RuntimeTensor.alloc(self.allocator, a.dtype, &out_shape);
            errdefer output.deinit();

            switch (a.dtype) {
                .f32 => {
                    const a_data = a.asConstSlice(f32).?;
                    const b_data = b.asConstSlice(f32).?;
                    const out_data = output.asSlice(f32).?;
                    for (0..b1) |bi1| {
                        for (0..b2) |bi2| {
                            const batch_idx = bi1 * b2 + bi2;
                            const a_batch = a_data[batch_idx * m * k ..][0 .. m * k];
                            const b_batch = b_data[batch_idx * k * n ..][0 .. k * n];
                            const out_batch = out_data[batch_idx * m * n ..][0 .. m * n];
                            kernels.matmul.matmulTiled(f32, a_batch, b_batch, out_batch, m, k, n);
                        }
                    }
                },
                .f64 => {
                    const a_data = a.asConstSlice(f64).?;
                    const b_data = b.asConstSlice(f64).?;
                    const out_data = output.asSlice(f64).?;
                    for (0..b1) |bi1| {
                        for (0..b2) |bi2| {
                            const batch_idx = bi1 * b2 + bi2;
                            const a_batch = a_data[batch_idx * m * k ..][0 .. m * k];
                            const b_batch = b_data[batch_idx * k * n ..][0 .. k * n];
                            const out_batch = out_data[batch_idx * m * n ..][0 .. m * n];
                            kernels.matmul.matmulTiled(f64, a_batch, b_batch, out_batch, m, k, n);
                        }
                    }
                },
                else => return error.UnsupportedDType,
            }

            if (self.buffers[output_idx]) |*existing| {
                existing.deinit();
            }
            self.buffers[output_idx] = output;
            return;
        }

        // 4D x 2D broadcast: [B1, B2, M, K] @ [K, N] -> [B1, B2, M, N]
        if (a_shape.len == 4 and b_shape.len == 2) {
            const b1: usize = @intCast(a_shape[0]);
            const b2: usize = @intCast(a_shape[1]);
            const m: usize = @intCast(a_shape[2]);
            const k: usize = @intCast(a_shape[3]);
            const n: usize = @intCast(b_shape[1]);

            if (k != @as(usize, @intCast(b_shape[0]))) return error.ShapeMismatch;

            const out_shape = [_]i64{ @intCast(b1), @intCast(b2), @intCast(m), @intCast(n) };
            var output = try RuntimeTensor.alloc(self.allocator, a.dtype, &out_shape);
            errdefer output.deinit();

            switch (a.dtype) {
                .f32 => {
                    const a_data = a.asConstSlice(f32).?;
                    const b_data = b.asConstSlice(f32).?;
                    const out_data = output.asSlice(f32).?;
                    for (0..b1) |bi1| {
                        for (0..b2) |bi2| {
                            const batch_idx = bi1 * b2 + bi2;
                            const a_batch = a_data[batch_idx * m * k ..][0 .. m * k];
                            const out_batch = out_data[batch_idx * m * n ..][0 .. m * n];
                            kernels.matmul.matmulTiled(f32, a_batch, b_data, out_batch, m, k, n);
                        }
                    }
                },
                .f64 => {
                    const a_data = a.asConstSlice(f64).?;
                    const b_data = b.asConstSlice(f64).?;
                    const out_data = output.asSlice(f64).?;
                    for (0..b1) |bi1| {
                        for (0..b2) |bi2| {
                            const batch_idx = bi1 * b2 + bi2;
                            const a_batch = a_data[batch_idx * m * k ..][0 .. m * k];
                            const out_batch = out_data[batch_idx * m * n ..][0 .. m * n];
                            kernels.matmul.matmulTiled(f64, a_batch, b_data, out_batch, m, k, n);
                        }
                    }
                },
                else => return error.UnsupportedDType,
            }

            if (self.buffers[output_idx]) |*existing| {
                existing.deinit();
            }
            self.buffers[output_idx] = output;
            return;
        }

        // 1D x 2D: [K] @ [K, N] -> [N]
        if (a_shape.len == 1 and b_shape.len == 2) {
            const k: usize = @intCast(a_shape[0]);
            const n: usize = @intCast(b_shape[1]);

            if (k != @as(usize, @intCast(b_shape[0]))) return error.ShapeMismatch;

            const out_shape = [_]i64{@intCast(n)};
            var output = try RuntimeTensor.alloc(self.allocator, a.dtype, &out_shape);
            errdefer output.deinit();

            switch (a.dtype) {
                .f32 => {
                    const a_data = a.asConstSlice(f32).?;
                    const b_data = b.asConstSlice(f32).?;
                    const out_data = output.asSlice(f32).?;
                    // Vector-matrix: treat as [1, K] @ [K, N] -> [1, N], then squeeze
                    for (0..n) |j| {
                        var sum: f32 = 0;
                        for (0..k) |ki| {
                            sum += a_data[ki] * b_data[ki * n + j];
                        }
                        out_data[j] = sum;
                    }
                },
                else => return error.UnsupportedDType,
            }

            if (self.buffers[output_idx]) |*existing| {
                existing.deinit();
            }
            self.buffers[output_idx] = output;
            return;
        }

        // 2D x 1D: [M, K] @ [K] -> [M]
        if (a_shape.len == 2 and b_shape.len == 1) {
            const m: usize = @intCast(a_shape[0]);
            const k: usize = @intCast(a_shape[1]);

            if (k != @as(usize, @intCast(b_shape[0]))) return error.ShapeMismatch;

            const out_shape = [_]i64{@intCast(m)};
            var output = try RuntimeTensor.alloc(self.allocator, a.dtype, &out_shape);
            errdefer output.deinit();

            switch (a.dtype) {
                .f32 => {
                    const a_data = a.asConstSlice(f32).?;
                    const b_data = b.asConstSlice(f32).?;
                    const out_data = output.asSlice(f32).?;
                    for (0..m) |i| {
                        var sum: f32 = 0;
                        for (0..k) |ki| {
                            sum += a_data[i * k + ki] * b_data[ki];
                        }
                        out_data[i] = sum;
                    }
                },
                else => return error.UnsupportedDType,
            }

            if (self.buffers[output_idx]) |*existing| {
                existing.deinit();
            }
            self.buffers[output_idx] = output;
            return;
        }

        // Fallback: try to handle as general N-D matmul by treating as 2D
        // This handles cases like [1, 1, M, K] @ [1, 1, K, N] by flattening batch dims
        if (a_shape.len >= 2 and b_shape.len >= 2) {
            const a_m: usize = @intCast(a_shape[a_shape.len - 2]);
            const a_k: usize = @intCast(a_shape[a_shape.len - 1]);
            const b_k: usize = @intCast(b_shape[b_shape.len - 2]);
            const b_n: usize = @intCast(b_shape[b_shape.len - 1]);

            if (a_k != b_k) return error.ShapeMismatch;

            // Calculate batch size
            var a_batch: usize = 1;
            for (a_shape[0 .. a_shape.len - 2]) |d| a_batch *= @intCast(d);
            var b_batch: usize = 1;
            for (b_shape[0 .. b_shape.len - 2]) |d| b_batch *= @intCast(d);

            const batch = @max(a_batch, b_batch);
            const m = a_m;
            const k = a_k;
            const n = b_n;

            // Build output shape
            var out_shape_buf: [8]i64 = undefined;
            const max_batch_dims = @max(a_shape.len, b_shape.len) - 2;
            for (0..max_batch_dims) |i| {
                const a_idx = if (i < a_shape.len - 2) i else null;
                const b_idx = if (i < b_shape.len - 2) i else null;
                const a_dim: i64 = if (a_idx) |idx| a_shape[idx] else 1;
                const b_dim: i64 = if (b_idx) |idx| b_shape[idx] else 1;
                out_shape_buf[i] = @max(a_dim, b_dim);
            }
            out_shape_buf[max_batch_dims] = @intCast(m);
            out_shape_buf[max_batch_dims + 1] = @intCast(n);
            const out_ndim = max_batch_dims + 2;

            var output = try RuntimeTensor.alloc(self.allocator, a.dtype, out_shape_buf[0..out_ndim]);
            errdefer output.deinit();

            switch (a.dtype) {
                .f32 => {
                    const a_data = a.asConstSlice(f32).?;
                    const b_data = b.asConstSlice(f32).?;
                    const out_data = output.asSlice(f32).?;

                    for (0..batch) |bi| {
                        const a_bi = if (a_batch == 1) 0 else bi;
                        const b_bi = if (b_batch == 1) 0 else bi;
                        const a_batch_data = a_data[a_bi * m * k ..][0 .. m * k];
                        const b_batch_data = b_data[b_bi * k * n ..][0 .. k * n];
                        const out_batch_data = out_data[bi * m * n ..][0 .. m * n];
                        kernels.matmul.matmulTiled(f32, a_batch_data, b_batch_data, out_batch_data, m, k, n);
                    }
                },
                else => return error.UnsupportedDType,
            }

            if (self.buffers[output_idx]) |*existing| {
                existing.deinit();
            }
            self.buffers[output_idx] = output;
            return;
        }

        return error.UnsupportedShape;
    }

    /// Execute Gather operation (for embeddings)
    fn execGather(self: *Executor, node: Node) !void {
        if (node.inputs.len < 2 or node.outputs.len < 1) return error.InvalidNode;

        const data = self.buffers[node.inputs[0]] orelse return error.MissingInput;
        const indices = self.buffers[node.inputs[1]] orelse return error.MissingInput;
        const output_idx = node.outputs[0];

        // Handle empty data tensor or scalar - return empty output
        if (data.numel == 0 or data.shape.len == 0) {
            // Output has same dtype as data, shape based on indices shape
            var out_shape_buf: [8]i64 = undefined;
            var out_ndim: usize = 0;
            for (indices.shape) |dim| {
                if (out_ndim < 8) {
                    out_shape_buf[out_ndim] = dim;
                    out_ndim += 1;
                }
            }
            if (out_ndim == 0) {
                out_shape_buf[0] = 0;
                out_ndim = 1;
            }
            const output = try RuntimeTensor.alloc(self.allocator, data.dtype, out_shape_buf[0..out_ndim]);
            if (self.buffers[output_idx]) |*existing| existing.deinit();
            self.buffers[output_idx] = output;
            return;
        }

        // Validate axis attribute - default to 0 if out of reasonable range
        const raw_axis: i64 = node.attributes.gather.axis;
        const axis: i64 = if (raw_axis >= -8 and raw_axis < 8) raw_axis else 0;
        const ndim: i64 = @intCast(data.shape.len);
        const actual_axis: usize = if (axis < 0)
            @intCast(ndim + axis)
        else if (axis >= ndim)
            0 // Default to 0 if out of bounds
        else
            @intCast(axis);

        if (actual_axis >= data.shape.len) return error.InvalidAxis;

        // Output shape: data.shape[0:axis] + indices.shape + data.shape[axis+1:]
        var out_shape_buf: [8]i64 = undefined;
        var out_ndim: usize = 0;

        // Add data dimensions before axis
        for (0..actual_axis) |i| {
            out_shape_buf[out_ndim] = data.shape[i];
            out_ndim += 1;
        }
        // Add indices dimensions
        for (indices.shape) |dim| {
            out_shape_buf[out_ndim] = dim;
            out_ndim += 1;
        }
        // Add data dimensions after axis
        for (actual_axis + 1..data.shape.len) |i| {
            out_shape_buf[out_ndim] = data.shape[i];
            out_ndim += 1;
        }

        var output = try RuntimeTensor.alloc(self.allocator, data.dtype, out_shape_buf[0..out_ndim]);
        errdefer output.deinit();

        // Compute outer iterations (before axis) and inner size (after axis)
        var outer_size: usize = 1;
        for (0..actual_axis) |d| {
            outer_size *= @intCast(data.shape[d]);
        }
        var after_axis_size: usize = 1;
        for (actual_axis + 1..data.shape.len) |d| {
            after_axis_size *= @intCast(data.shape[d]);
        }

        const axis_dim: usize = @intCast(data.shape[actual_axis]);

        // Get indices as i64 (convert from i32 if needed)
        var idx_data_buf: [4096]i64 = undefined;
        const idx_data: []const i64 = switch (indices.dtype) {
            .i64 => indices.asConstSlice(i64).?,
            .i32 => blk: {
                const i32_data = indices.asConstSlice(i32).?;
                for (i32_data, 0..) |v, i| {
                    idx_data_buf[i] = @intCast(v);
                }
                break :blk idx_data_buf[0..i32_data.len];
            },
            else => return error.UnsupportedDType,
        };

        switch (data.dtype) {
            inline .f32, .f64, .f16, .i32, .i64 => |dtype| {
                const T = dtype.ZigType();
                const src = data.asConstSlice(T).?;
                const dst = output.asSlice(T).?;

                var dst_offset: usize = 0;
                for (0..outer_size) |outer| {
                    for (idx_data) |idx_val| {
                        // Handle negative indices and bounds check
                        const adjusted: i64 = if (idx_val < 0) @as(i64, @intCast(axis_dim)) + idx_val else idx_val;
                        if (adjusted < 0 or adjusted >= @as(i64, @intCast(axis_dim))) {
                            std.debug.print("Gather: index {} out of bounds for axis_dim {}\n", .{ idx_val, axis_dim });
                            return error.IndexOutOfBounds;
                        }
                        const row: usize = @intCast(adjusted);
                        const src_base = (outer * axis_dim + row) * after_axis_size;
                        if (src_base + after_axis_size > src.len) {
                            return error.IndexOutOfBounds;
                        }
                        @memcpy(
                            dst[dst_offset .. dst_offset + after_axis_size],
                            src[src_base .. src_base + after_axis_size],
                        );
                        dst_offset += after_axis_size;
                    }
                }
            },
            else => return error.UnsupportedDType,
        }

        if (self.buffers[output_idx]) |*existing| {
            existing.deinit();
        }
        self.buffers[output_idx] = output;
    }

    /// Execute Concat operation
    fn execConcat(self: *Executor, node: Node) !void {
        if (node.inputs.len < 1 or node.outputs.len < 1) return error.InvalidNode;

        const axis: i64 = node.attributes.concat.axis;
        const output_idx = node.outputs[0];

        // Get first input for reference
        const first = self.buffers[node.inputs[0]] orelse return error.MissingInput;
        const ndim = first.shape.len;

        const actual_axis: usize = if (axis < 0)
            @intCast(@as(i64, @intCast(ndim)) + axis)
        else
            @intCast(axis);

        if (actual_axis >= ndim) return error.InvalidAxis;

        // Calculate output shape
        var out_shape_buf: [8]i64 = undefined;
        for (first.shape, 0..) |dim, i| {
            out_shape_buf[i] = dim;
        }

        // Sum dimensions along concat axis
        var concat_dim: i64 = 0;
        for (node.inputs) |input_idx| {
            const input = self.buffers[input_idx] orelse return error.MissingInput;
            concat_dim += input.shape[actual_axis];
        }
        out_shape_buf[actual_axis] = concat_dim;

        var output = try RuntimeTensor.alloc(self.allocator, first.dtype, out_shape_buf[0..ndim]);
        errdefer output.deinit();

        // Execute concat
        switch (first.dtype) {
            inline .f32, .f16, .i64, .i32 => |dtype| {
                const T = dtype.ZigType();
                const dst = output.asSlice(T).?;
                var offset: usize = 0;

                // Simple case: concat along last axis (most common)
                if (actual_axis == ndim - 1) {
                    // Calculate pre_size (product of all dims before axis)
                    var pre_size: usize = 1;
                    for (0..actual_axis) |i| {
                        pre_size *= @intCast(first.shape[i]);
                    }

                    var row_offset: usize = 0;
                    for (0..pre_size) |row| {
                        for (node.inputs) |input_idx| {
                            const input = self.buffers[input_idx].?;
                            const src = input.asConstSlice(T).?;
                            const chunk_size: usize = @intCast(input.shape[actual_axis]);
                            const src_start = row * chunk_size;

                            @memcpy(
                                dst[row_offset .. row_offset + chunk_size],
                                src[src_start .. src_start + chunk_size],
                            );
                            row_offset += chunk_size;
                        }
                    }
                } else if (actual_axis == 0) {
                    // Concat along axis 0 - simple copy
                    for (node.inputs) |input_idx| {
                        const input = self.buffers[input_idx].?;
                        const src = input.asConstSlice(T).?;
                        @memcpy(dst[offset .. offset + src.len], src);
                        offset += src.len;
                    }
                } else {
                    // General case: concat along middle axis
                    // Calculate sizes before, at, and after the concat axis
                    var pre_size: usize = 1;
                    for (0..actual_axis) |i| {
                        pre_size *= @intCast(first.shape[i]);
                    }
                    var post_size: usize = 1;
                    for (actual_axis + 1..ndim) |i| {
                        post_size *= @intCast(first.shape[i]);
                    }

                    // Total output axis size
                    const total_axis: usize = @intCast(concat_dim);
                    _ = total_axis;

                    for (0..pre_size) |pre| {
                        var axis_offset: usize = 0;
                        for (node.inputs) |input_idx| {
                            const input = self.buffers[input_idx].?;
                            const src = input.asConstSlice(T).?;
                            const input_axis_size: usize = @intCast(input.shape[actual_axis]);

                            for (0..input_axis_size) |ax| {
                                const src_idx = (pre * input_axis_size + ax) * post_size;
                                const dst_idx = (pre * @as(usize, @intCast(concat_dim)) + axis_offset + ax) * post_size;
                                @memcpy(dst[dst_idx..][0..post_size], src[src_idx..][0..post_size]);
                            }
                            axis_offset += input_axis_size;
                        }
                    }
                }
            },
            else => return error.UnsupportedDType,
        }

        if (self.buffers[output_idx]) |*existing| {
            existing.deinit();
        }
        self.buffers[output_idx] = output;
    }

    /// Execute Reshape (metadata only, no data copy)
    fn execReshape(self: *Executor, node: Node) !void {
        if (node.inputs.len < 2 or node.outputs.len < 1) return error.InvalidNode;

        const input = self.buffers[node.inputs[0]] orelse return error.MissingInput;
        const shape_tensor = self.buffers[node.inputs[1]] orelse return error.MissingInput;
        const output_idx = node.outputs[0];

        // Get new shape from shape tensor
        const new_shape = shape_tensor.asConstSlice(i64) orelse return error.UnsupportedDType;

        // Handle -1 (infer dimension)
        var total_new: i64 = 1;
        var infer_idx: ?usize = null;
        for (new_shape, 0..) |dim, i| {
            if (dim == -1) {
                if (infer_idx != null) return error.InvalidShape;
                infer_idx = i;
            } else if (dim == 0) {
                // 0 means keep original dim - bounds check
                if (i < input.shape.len) {
                    total_new *= input.shape[i];
                } else {
                    total_new *= 1; // Default to 1 if out of bounds
                }
            } else {
                total_new *= dim;
            }
        }

        var final_shape = try self.allocator.alloc(i64, new_shape.len);
        errdefer self.allocator.free(final_shape);

        const input_numel: i64 = @intCast(input.numel);
        for (new_shape, 0..) |dim, i| {
            if (dim == -1) {
                if (total_new != 0) {
                    final_shape[i] = @divFloor(input_numel, total_new);
                } else {
                    final_shape[i] = 0;
                }
            } else if (dim == 0) {
                // 0 means keep original dim - bounds check
                if (i < input.shape.len) {
                    final_shape[i] = input.shape[i];
                } else {
                    final_shape[i] = 1; // Default to 1 if out of bounds
                }
            } else {
                final_shape[i] = dim;
            }
        }

        // Create output sharing data with input
        const output = RuntimeTensor{
            .data = input.data,
            .dtype = input.dtype,
            .shape = final_shape,
            .numel = input.numel,
            .allocator = self.allocator,
            .owns_data = false, // Shared with input
        };

        if (self.buffers[output_idx]) |*existing| {
            existing.deinit();
        }
        self.buffers[output_idx] = output;
    }

    /// Execute Transpose
    fn execTranspose(self: *Executor, node: Node) !void {
        if (node.inputs.len < 1 or node.outputs.len < 1) return error.InvalidNode;

        const input = self.buffers[node.inputs[0]] orelse return error.MissingInput;
        const output_idx = node.outputs[0];

        const ndim = input.shape.len;

        // Default permutation: reverse
        var perm_buf: [8]usize = undefined;
        var actual_perm: []const usize = undefined;

        // Determine permutation: use provided perm or default to reverse
        if (node.attributes.transpose.perm) |perm| {
            if (perm.len == ndim and ndim <= 8) {
                // Copy and validate perm values
                var valid = true;
                for (perm, 0..) |v, i| {
                    const adjusted: i64 = if (v < 0)
                        @as(i64, @intCast(ndim)) + v
                    else
                        v;
                    if (adjusted < 0 or adjusted >= @as(i64, @intCast(ndim))) {
                        valid = false;
                        break;
                    }
                    perm_buf[i] = @intCast(adjusted);
                }
                if (valid) {
                    actual_perm = perm_buf[0..ndim];
                } else {
                    // Fall back to default reverse permutation
                    for (0..ndim) |i| {
                        perm_buf[i] = ndim - 1 - i;
                    }
                    actual_perm = perm_buf[0..ndim];
                }
            } else {
                // Length mismatch, use default
                for (0..ndim) |i| {
                    perm_buf[i] = ndim - 1 - i;
                }
                actual_perm = perm_buf[0..ndim];
            }
        } else {
            // No perm specified, use default reverse permutation
            for (0..ndim) |i| {
                perm_buf[i] = ndim - 1 - i;
            }
            actual_perm = perm_buf[0..ndim];
        }

        // Calculate output shape
        var out_shape_buf: [8]i64 = undefined;
        for (actual_perm, 0..) |p, i| {
            out_shape_buf[i] = input.shape[p];
        }

        var output = try RuntimeTensor.alloc(self.allocator, input.dtype, out_shape_buf[0..ndim]);
        errdefer output.deinit();

        // Early return for empty tensors
        if (input.numel == 0) {
            if (self.buffers[output_idx]) |*existing| {
                existing.deinit();
            }
            self.buffers[output_idx] = output;
            return;
        }

        // Convert shapes to usize for kernel
        var in_shape: [8]usize = undefined;
        var out_shape: [8]usize = undefined;
        for (0..ndim) |i| {
            in_shape[i] = @intCast(input.shape[i]);
            out_shape[i] = @intCast(out_shape_buf[i]);
        }

        // Use tenzor's transpose kernel
        switch (input.dtype) {
            inline .f32, .f16, .f64, .i64, .i32, .i16, .i8, .u8, .u16, .u32, .u64, .bool_ => |dtype| {
                const T = dtype.ZigType();
                const in_data = input.asConstSlice(T).?;
                const out_data = output.asSlice(T).?;
                kernels.transpose.transpose(T, in_data, out_data, in_shape[0..ndim], out_shape[0..ndim], actual_perm);
            },
            else => return error.UnsupportedDType,
        }

        if (self.buffers[output_idx]) |*existing| {
            existing.deinit();
        }
        self.buffers[output_idx] = output;
    }

    /// Execute Softmax
    fn execSoftmax(self: *Executor, node: Node) !void {
        if (node.inputs.len < 1 or node.outputs.len < 1) return error.InvalidNode;

        const input = self.buffers[node.inputs[0]] orelse return error.MissingInput;
        const output_idx = node.outputs[0];

        const axis: i64 = node.attributes.softmax.axis;
        const ndim = input.shape.len;
        const actual_axis: usize = if (axis < 0)
            @intCast(@as(i64, @intCast(ndim)) + axis)
        else
            @intCast(axis);

        if (actual_axis >= ndim) return error.InvalidAxis;

        var output = try RuntimeTensor.alloc(self.allocator, input.dtype, input.shape);
        errdefer output.deinit();

        switch (input.dtype) {
            .f32 => {
                const in_data = input.asConstSlice(f32).?;
                const out_data = output.asSlice(f32).?;

                // Calculate axis dimension and stride
                const axis_size: usize = @intCast(input.shape[actual_axis]);
                var outer_size: usize = 1;
                var inner_size: usize = 1;

                for (0..actual_axis) |i| {
                    outer_size *= @intCast(input.shape[i]);
                }
                for (actual_axis + 1..ndim) |i| {
                    inner_size *= @intCast(input.shape[i]);
                }

                // Apply softmax along axis
                for (0..outer_size) |o| {
                    for (0..inner_size) |i| {
                        // Find max for numerical stability
                        var max_val: f32 = -std.math.inf(f32);
                        for (0..axis_size) |a| {
                            const idx = o * axis_size * inner_size + a * inner_size + i;
                            max_val = @max(max_val, in_data[idx]);
                        }

                        // Compute exp and sum
                        var sum: f32 = 0;
                        for (0..axis_size) |a| {
                            const idx = o * axis_size * inner_size + a * inner_size + i;
                            const exp_val = @exp(in_data[idx] - max_val);
                            out_data[idx] = exp_val;
                            sum += exp_val;
                        }

                        // Normalize
                        for (0..axis_size) |a| {
                            const idx = o * axis_size * inner_size + a * inner_size + i;
                            out_data[idx] /= sum;
                        }
                    }
                }
            },
            else => return error.UnsupportedDType,
        }

        if (self.buffers[output_idx]) |*existing| {
            existing.deinit();
        }
        self.buffers[output_idx] = output;
    }

    /// Execute Slice operation
    /// Inputs: data, starts, ends, axes (optional), steps (optional)
    fn execSlice(self: *Executor, node: Node) !void {
        if (node.inputs.len < 3 or node.outputs.len < 1) return error.InvalidNode;

        const data = self.buffers[node.inputs[0]] orelse return error.MissingInput;
        const starts_tensor = self.buffers[node.inputs[1]] orelse return error.MissingInput;
        const ends_tensor = self.buffers[node.inputs[2]] orelse return error.MissingInput;
        const output_idx = node.outputs[0];

        const starts = starts_tensor.asConstSlice(i64) orelse return error.UnsupportedDType;
        const ends = ends_tensor.asConstSlice(i64) orelse return error.UnsupportedDType;

        // Get optional axes and steps
        var axes_buf: [8]i64 = undefined;
        var steps_buf: [8]i64 = undefined;
        var axes: []const i64 = undefined;
        var steps: []const i64 = undefined;

        if (node.inputs.len > 3) {
            if (self.buffers[node.inputs[3]]) |axes_tensor| {
                axes = axes_tensor.asConstSlice(i64) orelse return error.UnsupportedDType;
            } else {
                // Default axes: 0, 1, 2, ...
                for (0..starts.len) |i| {
                    axes_buf[i] = @intCast(i);
                }
                axes = axes_buf[0..starts.len];
            }
        } else {
            for (0..starts.len) |i| {
                axes_buf[i] = @intCast(i);
            }
            axes = axes_buf[0..starts.len];
        }

        if (node.inputs.len > 4) {
            if (self.buffers[node.inputs[4]]) |steps_tensor| {
                steps = steps_tensor.asConstSlice(i64) orelse return error.UnsupportedDType;
            } else {
                for (0..starts.len) |i| {
                    steps_buf[i] = 1;
                }
                steps = steps_buf[0..starts.len];
            }
        } else {
            for (0..starts.len) |i| {
                steps_buf[i] = 1;
            }
            steps = steps_buf[0..starts.len];
        }

        const ndim = data.shape.len;

        // Compute actual start/end/step indices handling negative values
        var actual_starts: [8]usize = undefined;
        var actual_ends: [8]usize = undefined;
        var actual_steps: [8]i64 = undefined;
        var out_shape_buf: [8]i64 = undefined;

        // Initialize to full range with step 1
        for (0..ndim) |i| {
            actual_starts[i] = 0;
            actual_ends[i] = @intCast(data.shape[i]);
            actual_steps[i] = 1;
            out_shape_buf[i] = data.shape[i];
        }

        // Apply slice parameters
        for (axes, 0..) |axis_val, i| {
            // Handle negative axis (Python-style indexing)
            const adjusted_axis: i64 = if (axis_val < 0)
                @as(i64, @intCast(ndim)) + axis_val
            else
                axis_val;

            // Validate axis is in bounds
            if (adjusted_axis < 0 or adjusted_axis >= @as(i64, @intCast(ndim))) {
                // Axis out of bounds - skip this slice dimension
                // This can happen when the model expects more dimensions than we have
                // Common in models with dynamic batch dimensions
                continue;
            }
            const axis: usize = @intCast(adjusted_axis);

            const dim_size: i64 = data.shape[axis];
            var start = starts[i];
            var end = ends[i];
            const step = steps[i];

            // Handle negative indices
            if (start < 0) start = dim_size + start;
            if (end < 0) end = dim_size + end;

            // Clamp to valid range
            start = @max(0, @min(start, dim_size));
            end = @max(0, @min(end, dim_size));

            // Handle large end values (INT_MAX convention)
            if (end > dim_size) end = dim_size;

            actual_starts[axis] = @intCast(start);
            actual_ends[axis] = @intCast(end);
            actual_steps[axis] = step;

            // Calculate output dimension
            if (step > 0) {
                out_shape_buf[axis] = @divFloor(end - start + step - 1, step);
            } else {
                out_shape_buf[axis] = @divFloor(start - end - step - 1, -step);
            }
            if (out_shape_buf[axis] < 0) out_shape_buf[axis] = 0;
        }

        var output = try RuntimeTensor.alloc(self.allocator, data.dtype, out_shape_buf[0..ndim]);
        errdefer output.deinit();

        // Execute slice
        switch (data.dtype) {
            .f32 => try self.sliceData(f32, &data, &output, actual_starts[0..ndim], actual_ends[0..ndim], actual_steps[0..ndim]),
            .f64 => try self.sliceData(f64, &data, &output, actual_starts[0..ndim], actual_ends[0..ndim], actual_steps[0..ndim]),
            .f16 => try self.sliceData(f16, &data, &output, actual_starts[0..ndim], actual_ends[0..ndim], actual_steps[0..ndim]),
            .i64 => try self.sliceData(i64, &data, &output, actual_starts[0..ndim], actual_ends[0..ndim], actual_steps[0..ndim]),
            .i32 => try self.sliceData(i32, &data, &output, actual_starts[0..ndim], actual_ends[0..ndim], actual_steps[0..ndim]),
            else => return error.UnsupportedDType,
        }

        if (self.buffers[output_idx]) |*existing| {
            existing.deinit();
        }
        self.buffers[output_idx] = output;
    }

    fn sliceData(
        self: *Executor,
        comptime T: type,
        data: *const RuntimeTensor,
        output: *RuntimeTensor,
        starts: []const usize,
        _: []const usize, // ends (unused - we use output shape)
        steps: []const i64,
    ) !void {
        _ = self;
        const src = data.asConstSlice(T) orelse return error.UnsupportedDType;
        const dst = output.asSlice(T) orelse return error.UnsupportedDType;

        const ndim = data.shape.len;

        // Calculate strides for source
        var src_strides: [8]usize = undefined;
        src_strides[ndim - 1] = 1;
        var i: usize = ndim - 1;
        while (i > 0) : (i -= 1) {
            src_strides[i - 1] = src_strides[i] * @as(usize, @intCast(data.shape[i]));
        }

        // Calculate strides for destination
        var dst_strides: [8]usize = undefined;
        dst_strides[ndim - 1] = 1;
        i = ndim - 1;
        while (i > 0) : (i -= 1) {
            dst_strides[i - 1] = dst_strides[i] * @as(usize, @intCast(output.shape[i]));
        }

        // Iterate through output and copy from source
        var dst_idx: usize = 0;
        var coords: [8]usize = undefined;
        @memset(&coords, 0);

        while (dst_idx < output.numel) {
            // Calculate source index from output coordinates
            var src_idx: usize = 0;
            for (0..ndim) |d| {
                const step: usize = @intCast(steps[d]);
                const src_coord = starts[d] + coords[d] * step;
                src_idx += src_coord * src_strides[d];
            }

            dst[dst_idx] = src[src_idx];
            dst_idx += 1;

            // Increment coordinates
            var d: usize = ndim;
            while (d > 0) {
                d -= 1;
                coords[d] += 1;
                if (coords[d] < @as(usize, @intCast(output.shape[d]))) break;
                coords[d] = 0;
            }
        }
    }

    /// Execute Equal operation (element-wise comparison)
    fn execEqual(self: *Executor, node: Node) !void {
        if (node.inputs.len < 2 or node.outputs.len < 1) return error.InvalidNode;

        const a = self.buffers[node.inputs[0]] orelse return error.MissingInput;
        const b = self.buffers[node.inputs[1]] orelse return error.MissingInput;
        const output_idx = node.outputs[0];

        // Output is always bool type
        var output = try RuntimeTensor.alloc(self.allocator, .bool_, a.shape);
        errdefer output.deinit();

        const out_data = output.asSlice(u8) orelse return error.UnsupportedDType;

        // Handle scalar broadcasting
        if (b.numel == 1) {
            switch (a.dtype) {
                .i64 => {
                    const a_data = a.asConstSlice(i64).?;
                    const b_scalar = b.asConstSlice(i64).?[0];
                    for (a_data, 0..) |val, idx| {
                        out_data[idx] = if (val == b_scalar) 1 else 0;
                    }
                },
                .i32 => {
                    const a_data = a.asConstSlice(i32).?;
                    const b_scalar = b.asConstSlice(i32).?[0];
                    for (a_data, 0..) |val, idx| {
                        out_data[idx] = if (val == b_scalar) 1 else 0;
                    }
                },
                .f32 => {
                    const a_data = a.asConstSlice(f32).?;
                    const b_scalar = b.asConstSlice(f32).?[0];
                    for (a_data, 0..) |val, idx| {
                        out_data[idx] = if (val == b_scalar) 1 else 0;
                    }
                },
                else => return error.UnsupportedDType,
            }
        } else if (a.numel == b.numel) {
            // Element-wise comparison
            switch (a.dtype) {
                .i64 => {
                    const a_data = a.asConstSlice(i64).?;
                    const b_data = b.asConstSlice(i64).?;
                    for (a_data, b_data, 0..) |av, bv, idx| {
                        out_data[idx] = if (av == bv) 1 else 0;
                    }
                },
                .i32 => {
                    const a_data = a.asConstSlice(i32).?;
                    const b_data = b.asConstSlice(i32).?;
                    for (a_data, b_data, 0..) |av, bv, idx| {
                        out_data[idx] = if (av == bv) 1 else 0;
                    }
                },
                .f32 => {
                    const a_data = a.asConstSlice(f32).?;
                    const b_data = b.asConstSlice(f32).?;
                    for (a_data, b_data, 0..) |av, bv, idx| {
                        out_data[idx] = if (av == bv) 1 else 0;
                    }
                },
                else => return error.UnsupportedDType,
            }
        } else {
            return error.ShapeMismatch;
        }

        if (self.buffers[output_idx]) |*existing| {
            existing.deinit();
        }
        self.buffers[output_idx] = output;
    }

    /// Execute Where operation (conditional selection)
    /// output[i] = x[i] if condition[i] else y[i]
    fn execWhere(self: *Executor, node: Node) !void {
        if (node.inputs.len < 3 or node.outputs.len < 1) return error.InvalidNode;

        const condition = self.buffers[node.inputs[0]] orelse return error.MissingInput;
        const x = self.buffers[node.inputs[1]] orelse return error.MissingInput;
        const y = self.buffers[node.inputs[2]] orelse return error.MissingInput;
        const output_idx = node.outputs[0];

        // Get condition as bool (stored as u8)
        const cond_data = condition.asConstSlice(u8) orelse return error.UnsupportedDType;

        // Determine output shape by broadcasting all three inputs (condition, x, y)
        var temp_shape_buf: [8]i64 = undefined;
        const xy_shape = computeBroadcastShape(x.shape, y.shape, &temp_shape_buf) orelse
            return error.ShapeMismatch;

        var out_shape_buf: [8]i64 = undefined;
        const out_shape = computeBroadcastShape(condition.shape, xy_shape, &out_shape_buf) orelse
            return error.ShapeMismatch;

        var out_numel: usize = 1;
        for (out_shape) |dim| {
            out_numel *= @intCast(dim);
        }

        var output = try RuntimeTensor.alloc(self.allocator, x.dtype, out_shape);
        errdefer output.deinit();

        switch (x.dtype) {
            .f32 => {
                const x_data = x.asConstSlice(f32).?;
                const y_data = y.asConstSlice(f32).?;
                const out_data = output.asSlice(f32).?;

                // Handle broadcasting
                for (0..out_numel) |i| {
                    const cond_idx = i % condition.numel;
                    const x_idx = i % x.numel;
                    const y_idx = i % y.numel;
                    out_data[i] = if (cond_data[cond_idx] != 0) x_data[x_idx] else y_data[y_idx];
                }
            },
            .i64 => {
                const x_data = x.asConstSlice(i64).?;
                const y_data = y.asConstSlice(i64).?;
                const out_data = output.asSlice(i64).?;

                for (0..out_numel) |i| {
                    const cond_idx = i % condition.numel;
                    const x_idx = i % x.numel;
                    const y_idx = i % y.numel;
                    out_data[i] = if (cond_data[cond_idx] != 0) x_data[x_idx] else y_data[y_idx];
                }
            },
            .i32 => {
                const x_data = x.asConstSlice(i32).?;
                const y_data = y.asConstSlice(i32).?;
                const out_data = output.asSlice(i32).?;

                for (0..out_numel) |i| {
                    const cond_idx = i % condition.numel;
                    const x_idx = i % x.numel;
                    const y_idx = i % y.numel;
                    out_data[i] = if (cond_data[cond_idx] != 0) x_data[x_idx] else y_data[y_idx];
                }
            },
            else => return error.UnsupportedDType,
        }

        if (self.buffers[output_idx]) |*existing| {
            existing.deinit();
        }
        self.buffers[output_idx] = output;
    }

    /// Execute Constant operation (creates a constant tensor from attributes)
    fn execConstant(self: *Executor, node: Node) !void {
        if (node.outputs.len < 1) return error.InvalidNode;

        const output_idx = node.outputs[0];

        // Get constant value from attributes
        const const_attr = node.attributes.constant;
        if (const_attr.value) |weight| {
            const output = try RuntimeTensor.fromWeightData(self.allocator, weight);
            if (self.buffers[output_idx]) |*existing| {
                existing.deinit();
            }
            self.buffers[output_idx] = output;
        } else {
            return error.MissingAttribute;
        }
    }

    /// Execute Shape operation (returns tensor shape as i64 tensor)
    fn execShape(self: *Executor, node: Node) !void {
        if (node.inputs.len < 1 or node.outputs.len < 1) return error.InvalidNode;

        const input = self.buffers[node.inputs[0]] orelse return error.MissingInput;
        const output_idx = node.outputs[0];

        // Create output tensor with shape [ndim]
        const ndim = input.shape.len;
        var output = try RuntimeTensor.alloc(self.allocator, .i64, &.{@intCast(ndim)});
        errdefer output.deinit();

        // Copy shape values
        const out_data = output.asSlice(i64).?;
        for (input.shape, 0..) |dim, i| {
            out_data[i] = dim;
        }

        if (self.buffers[output_idx]) |*existing| {
            existing.deinit();
        }
        self.buffers[output_idx] = output;
    }

    /// Execute Cast operation (type conversion)
    fn execCast(self: *Executor, node: Node) !void {
        if (node.inputs.len < 1 or node.outputs.len < 1) return error.InvalidNode;

        const input = self.buffers[node.inputs[0]] orelse return error.MissingInput;
        const output_idx = node.outputs[0];

        // Get target dtype from attributes
        const target_dtype = node.attributes.cast.to;

        // Allocate output with same shape but new dtype
        var output = try RuntimeTensor.alloc(self.allocator, target_dtype, input.shape);
        errdefer output.deinit();

        // Dispatch by dtype pairs
        switch (input.dtype) {
            .f16 => {
                const in_data = input.asConstSlice(f16).?;
                switch (target_dtype) {
                    .f16 => @memcpy(output.asSlice(f16).?, in_data),
                    .f32 => for (in_data, output.asSlice(f32).?) |v, *o| { o.* = @floatCast(v); },
                    .f64 => for (in_data, output.asSlice(f64).?) |v, *o| { o.* = @floatCast(v); },
                    .i32 => for (in_data, output.asSlice(i32).?) |v, *o| { o.* = @intFromFloat(v); },
                    .i64 => for (in_data, output.asSlice(i64).?) |v, *o| { o.* = @intFromFloat(v); },
                    else => return error.UnsupportedDType,
                }
            },
            .f32 => {
                const in_data = input.asConstSlice(f32).?;
                switch (target_dtype) {
                    .f16 => for (in_data, output.asSlice(f16).?) |v, *o| { o.* = @floatCast(v); },
                    .f32 => @memcpy(output.asSlice(f32).?, in_data),
                    .f64 => for (in_data, output.asSlice(f64).?) |v, *o| { o.* = @floatCast(v); },
                    .i32 => for (in_data, output.asSlice(i32).?) |v, *o| {
                        if (std.math.isNan(v) or std.math.isInf(v)) {
                            o.* = 0;
                        } else {
                            const clamped = @max(@as(f32, @floatFromInt(std.math.minInt(i32))), @min(@as(f32, @floatFromInt(std.math.maxInt(i32))), v));
                            o.* = @intFromFloat(clamped);
                        }
                    },
                    .i64 => for (in_data, output.asSlice(i64).?) |v, *o| {
                        if (std.math.isNan(v) or std.math.isInf(v)) {
                            o.* = 0;
                        } else {
                            // Clamp to safe range for f32->i64 conversion
                            const clamped = @max(-9007199254740992.0, @min(9007199254740992.0, v));
                            o.* = @intFromFloat(clamped);
                        }
                    },
                    else => return error.UnsupportedDType,
                }
            },
            .f64 => {
                const in_data = input.asConstSlice(f64).?;
                switch (target_dtype) {
                    .f16 => for (in_data, output.asSlice(f16).?) |v, *o| { o.* = @floatCast(v); },
                    .f32 => for (in_data, output.asSlice(f32).?) |v, *o| { o.* = @floatCast(v); },
                    .f64 => @memcpy(output.asSlice(f64).?, in_data),
                    .i32 => for (in_data, output.asSlice(i32).?) |v, *o| {
                        if (std.math.isNan(v) or std.math.isInf(v)) {
                            o.* = 0;
                        } else {
                            const clamped = @max(@as(f64, @floatFromInt(std.math.minInt(i32))), @min(@as(f64, @floatFromInt(std.math.maxInt(i32))), v));
                            o.* = @intFromFloat(clamped);
                        }
                    },
                    .i64 => for (in_data, output.asSlice(i64).?) |v, *o| {
                        if (std.math.isNan(v) or std.math.isInf(v)) {
                            o.* = 0;
                        } else {
                            const clamped = @max(@as(f64, @floatFromInt(std.math.minInt(i64))), @min(@as(f64, @floatFromInt(std.math.maxInt(i64))), v));
                            o.* = @intFromFloat(clamped);
                        }
                    },
                    else => return error.UnsupportedDType,
                }
            },
            .i32 => {
                const in_data = input.asConstSlice(i32).?;
                switch (target_dtype) {
                    .f16 => for (in_data, output.asSlice(f16).?) |v, *o| { o.* = @floatFromInt(v); },
                    .f32 => for (in_data, output.asSlice(f32).?) |v, *o| { o.* = @floatFromInt(v); },
                    .f64 => for (in_data, output.asSlice(f64).?) |v, *o| { o.* = @floatFromInt(v); },
                    .i32 => @memcpy(output.asSlice(i32).?, in_data),
                    .i64 => for (in_data, output.asSlice(i64).?) |v, *o| { o.* = @intCast(v); },
                    else => return error.UnsupportedDType,
                }
            },
            .i64 => {
                const in_data = input.asConstSlice(i64).?;
                switch (target_dtype) {
                    .f16 => for (in_data, output.asSlice(f16).?) |v, *o| { o.* = @floatFromInt(v); },
                    .f32 => for (in_data, output.asSlice(f32).?) |v, *o| { o.* = @floatFromInt(v); },
                    .f64 => for (in_data, output.asSlice(f64).?) |v, *o| { o.* = @floatFromInt(v); },
                    .i32 => for (in_data, output.asSlice(i32).?) |v, *o| { o.* = @intCast(v); },
                    .i64 => @memcpy(output.asSlice(i64).?, in_data),
                    else => return error.UnsupportedDType,
                }
            },
            .bool_ => {
                // Bool stored as u8: 0 = false, non-zero = true
                const in_data = input.asConstSlice(u8).?;
                switch (target_dtype) {
                    .f16 => for (in_data, output.asSlice(f16).?) |v, *o| { o.* = if (v != 0) @as(f16, 1.0) else @as(f16, 0.0); },
                    .f32 => for (in_data, output.asSlice(f32).?) |v, *o| { o.* = if (v != 0) @as(f32, 1.0) else @as(f32, 0.0); },
                    .f64 => for (in_data, output.asSlice(f64).?) |v, *o| { o.* = if (v != 0) @as(f64, 1.0) else @as(f64, 0.0); },
                    .i32 => for (in_data, output.asSlice(i32).?) |v, *o| { o.* = if (v != 0) @as(i32, 1) else @as(i32, 0); },
                    .i64 => for (in_data, output.asSlice(i64).?) |v, *o| { o.* = if (v != 0) @as(i64, 1) else @as(i64, 0); },
                    .bool_ => @memcpy(output.asSlice(u8).?, in_data),
                    else => return error.UnsupportedDType,
                }
            },
            .u8 => {
                const in_data = input.asConstSlice(u8).?;
                switch (target_dtype) {
                    .f16 => for (in_data, output.asSlice(f16).?) |v, *o| { o.* = @floatFromInt(v); },
                    .f32 => for (in_data, output.asSlice(f32).?) |v, *o| { o.* = @floatFromInt(v); },
                    .f64 => for (in_data, output.asSlice(f64).?) |v, *o| { o.* = @floatFromInt(v); },
                    .i32 => for (in_data, output.asSlice(i32).?) |v, *o| { o.* = @intCast(v); },
                    .i64 => for (in_data, output.asSlice(i64).?) |v, *o| { o.* = @intCast(v); },
                    .u8 => @memcpy(output.asSlice(u8).?, in_data),
                    .bool_ => @memcpy(output.asSlice(u8).?, in_data),
                    else => return error.UnsupportedDType,
                }
            },
            else => return error.UnsupportedDType,
        }

        if (self.buffers[output_idx]) |*existing| {
            existing.deinit();
        }
        self.buffers[output_idx] = output;
    }

    /// Execute ReduceSum operation
    fn execReduceSum(self: *Executor, node: Node) !void {
        if (node.inputs.len < 1 or node.outputs.len < 1) return error.InvalidNode;

        const input = self.buffers[node.inputs[0]] orelse return error.MissingInput;
        const output_idx = node.outputs[0];

        // Get reduce attributes
        const attrs = node.attributes.reduce;
        const keepdims = attrs.keepdims;

        // Get axes from attributes or second input
        var axes_to_reduce: []const i64 = attrs.axes orelse &.{};

        // If axes empty and noop_with_empty_axes, just copy input
        if (axes_to_reduce.len == 0 and attrs.noop_with_empty_axes) {
            var output = try RuntimeTensor.alloc(self.allocator, input.dtype, input.shape);
            @memcpy(output.data, input.data);
            if (self.buffers[output_idx]) |*existing| {
                existing.deinit();
            }
            self.buffers[output_idx] = output;
            return;
        }

        // If axes empty, reduce all dimensions
        var all_axes_buf: [8]i64 = undefined;
        if (axes_to_reduce.len == 0) {
            for (0..input.shape.len) |i| {
                all_axes_buf[i] = @intCast(i);
            }
            axes_to_reduce = all_axes_buf[0..input.shape.len];
        }

        // Normalize negative axes
        var norm_axes_buf: [8]usize = undefined;
        for (axes_to_reduce, 0..) |ax, i| {
            const norm: usize = if (ax < 0)
                @intCast(@as(i64, @intCast(input.shape.len)) + ax)
            else
                @intCast(ax);
            norm_axes_buf[i] = norm;
        }
        const norm_axes = norm_axes_buf[0..axes_to_reduce.len];

        // Compute output shape
        var out_shape_buf: [8]i64 = undefined;
        var out_ndim: usize = 0;
        for (input.shape, 0..) |dim, i| {
            var is_reduced = false;
            for (norm_axes) |ax| {
                if (ax == i) {
                    is_reduced = true;
                    break;
                }
            }
            if (is_reduced) {
                if (keepdims) {
                    out_shape_buf[out_ndim] = 1;
                    out_ndim += 1;
                }
            } else {
                out_shape_buf[out_ndim] = dim;
                out_ndim += 1;
            }
        }

        var output = try RuntimeTensor.alloc(self.allocator, input.dtype, out_shape_buf[0..out_ndim]);
        errdefer output.deinit();

        // Compute strides for input
        var in_strides: [8]usize = undefined;
        {
            var stride: usize = 1;
            var i = input.shape.len;
            while (i > 0) {
                i -= 1;
                in_strides[i] = stride;
                stride *= @intCast(input.shape[i]);
            }
        }

        // Compute strides for output
        var out_strides: [8]usize = undefined;
        {
            var stride: usize = 1;
            var i = out_ndim;
            while (i > 0) {
                i -= 1;
                out_strides[i] = stride;
                stride *= @intCast(out_shape_buf[i]);
            }
        }

        // Simple reduction implementation
        switch (input.dtype) {
            inline .f32, .f64, .f16, .i32, .i64 => |dtype| {
                const T = dtype.ZigType();
                const in_data = input.asConstSlice(T).?;
                const out_data = output.asSlice(T).?;
                @memset(out_data, 0);

                // Iterate over input and accumulate
                for (0..input.numel) |flat_idx| {
                    // Compute multi-index for input
                    var remaining = flat_idx;
                    var out_flat_idx: usize = 0;
                    var out_dim_idx: usize = 0;

                    for (0..input.shape.len) |d| {
                        const idx = remaining / in_strides[d];
                        remaining = remaining % in_strides[d];

                        var is_reduced = false;
                        for (norm_axes) |ax| {
                            if (ax == d) {
                                is_reduced = true;
                                break;
                            }
                        }

                        if (!is_reduced or keepdims) {
                            const out_idx = if (is_reduced) 0 else idx;
                            out_flat_idx += out_idx * out_strides[out_dim_idx];
                            out_dim_idx += 1;
                        }
                    }

                    out_data[out_flat_idx] += in_data[flat_idx];
                }
            },
            else => return error.UnsupportedDType,
        }

        if (self.buffers[output_idx]) |*existing| {
            existing.deinit();
        }
        self.buffers[output_idx] = output;
    }

    /// Execute ReduceProd operation
    fn execReduceProd(self: *Executor, node: Node) !void {
        if (node.inputs.len < 1 or node.outputs.len < 1) return error.InvalidNode;

        const input = self.buffers[node.inputs[0]] orelse return error.MissingInput;
        const output_idx = node.outputs[0];

        // Get reduce attributes (if present)
        const has_reduce_attrs = node.attributes == .reduce;
        const keepdims = if (has_reduce_attrs) node.attributes.reduce.keepdims else true;
        const noop_with_empty_axes = if (has_reduce_attrs) node.attributes.reduce.noop_with_empty_axes else false;

        // Get axes from attributes or second input (ONNX opset 13+)
        var axes_to_reduce: []const i64 = if (has_reduce_attrs)
            node.attributes.reduce.axes orelse &.{}
        else
            &.{};

        // Check for axes as second input (opset 13+)
        var axes_from_input_buf: [8]i64 = undefined;
        if (axes_to_reduce.len == 0 and node.inputs.len > 1) {
            if (self.buffers[node.inputs[1]]) |axes_tensor| {
                if (axes_tensor.dtype == .i64) {
                    const axes_data = axes_tensor.asConstSlice(i64).?;
                    const count = @min(axes_data.len, 8);
                    @memcpy(axes_from_input_buf[0..count], axes_data[0..count]);
                    axes_to_reduce = axes_from_input_buf[0..count];
                }
            }
        }

        // If axes empty and noop_with_empty_axes, just copy input
        if (axes_to_reduce.len == 0 and noop_with_empty_axes) {
            var output = try RuntimeTensor.alloc(self.allocator, input.dtype, input.shape);
            @memcpy(output.data, input.data);
            if (self.buffers[output_idx]) |*existing| {
                existing.deinit();
            }
            self.buffers[output_idx] = output;
            return;
        }

        // If axes empty, reduce all dimensions
        var all_axes_buf: [8]i64 = undefined;
        if (axes_to_reduce.len == 0) {
            for (0..input.shape.len) |i| {
                all_axes_buf[i] = @intCast(i);
            }
            axes_to_reduce = all_axes_buf[0..input.shape.len];
        }

        // Normalize negative axes
        var norm_axes_buf: [8]usize = undefined;
        for (axes_to_reduce, 0..) |ax, i| {
            const norm: usize = if (ax < 0)
                @intCast(@as(i64, @intCast(input.shape.len)) + ax)
            else
                @intCast(ax);
            norm_axes_buf[i] = norm;
        }
        const norm_axes = norm_axes_buf[0..axes_to_reduce.len];

        // Compute output shape
        var out_shape_buf: [8]i64 = undefined;
        var out_ndim: usize = 0;
        for (input.shape, 0..) |dim, i| {
            var is_reduced = false;
            for (norm_axes) |ax| {
                if (ax == i) {
                    is_reduced = true;
                    break;
                }
            }
            if (is_reduced) {
                if (keepdims) {
                    out_shape_buf[out_ndim] = 1;
                    out_ndim += 1;
                }
            } else {
                out_shape_buf[out_ndim] = dim;
                out_ndim += 1;
            }
        }

        var output = try RuntimeTensor.alloc(self.allocator, input.dtype, out_shape_buf[0..out_ndim]);
        errdefer output.deinit();

        // Compute strides for input
        var in_strides: [8]usize = undefined;
        {
            var stride: usize = 1;
            var i = input.shape.len;
            while (i > 0) {
                i -= 1;
                in_strides[i] = stride;
                stride *= @intCast(input.shape[i]);
            }
        }

        // Compute strides for output
        var out_strides: [8]usize = undefined;
        {
            var stride: usize = 1;
            var i = out_ndim;
            while (i > 0) {
                i -= 1;
                out_strides[i] = stride;
                stride *= @intCast(out_shape_buf[i]);
            }
        }

        // Product reduction - initialize to 1 and multiply
        switch (input.dtype) {
            inline .f32, .f64, .f16, .i32, .i64 => |dtype| {
                const T = dtype.ZigType();
                const in_data = input.asConstSlice(T).?;
                const out_data = output.asSlice(T).?;

                // Initialize to 1 (multiplicative identity)
                for (out_data) |*v| {
                    v.* = 1;
                }

                // Iterate over input and multiply
                for (0..input.numel) |flat_idx| {
                    // Compute multi-index for input
                    var remaining = flat_idx;
                    var out_flat_idx: usize = 0;
                    var out_dim_idx: usize = 0;

                    for (0..input.shape.len) |d| {
                        const idx = remaining / in_strides[d];
                        remaining = remaining % in_strides[d];

                        var is_reduced = false;
                        for (norm_axes) |ax| {
                            if (ax == d) {
                                is_reduced = true;
                                break;
                            }
                        }

                        if (!is_reduced or keepdims) {
                            const out_idx = if (is_reduced) 0 else idx;
                            out_flat_idx += out_idx * out_strides[out_dim_idx];
                            out_dim_idx += 1;
                        }
                    }

                    out_data[out_flat_idx] *= in_data[flat_idx];
                }
            },
            else => return error.UnsupportedDType,
        }

        if (self.buffers[output_idx]) |*existing| {
            existing.deinit();
        }
        self.buffers[output_idx] = output;
    }

    /// Execute LayerNormalization operation
    fn execLayerNorm(self: *Executor, node: Node) !void {
        // LayerNorm: Y = (X - mean) / sqrt(var + epsilon) * scale + bias
        // Inputs: X, scale, bias (optional)
        // Outputs: Y, mean (optional), inv_std_dev (optional)
        if (node.inputs.len < 2 or node.outputs.len < 1) return error.InvalidNode;

        const input = self.buffers[node.inputs[0]] orelse return error.MissingInput;
        const scale = self.buffers[node.inputs[1]] orelse return error.MissingInput;
        const bias: ?RuntimeTensor = if (node.inputs.len > 2) self.buffers[node.inputs[2]] else null;

        const output_idx = node.outputs[0];

        // Get epsilon (default 1e-5)
        const epsilon: f32 = 1e-5;

        // Get axis (default -1 = last dimension)
        // For now assume normalizing over last axis
        const norm_axis = input.shape.len - 1;
        const norm_size: usize = @intCast(input.shape[norm_axis]);

        // Compute number of normalization groups
        var num_groups: usize = 1;
        for (input.shape[0..norm_axis]) |dim| {
            num_groups *= @intCast(dim);
        }

        var output = try RuntimeTensor.alloc(self.allocator, input.dtype, input.shape);
        errdefer output.deinit();

        switch (input.dtype) {
            .f32 => {
                const in_data = input.asConstSlice(f32).?;
                // Handle f16 scales by converting to f32
                const scale_f32 = if (scale.asConstSlice(f32)) |s32| blk: {
                    break :blk s32;
                } else if (scale.asConstSlice(f16)) |s16| blk: {
                    const temp = try self.allocator.alloc(f32, s16.len);
                    for (s16, temp) |s, *d| d.* = @floatCast(s);
                    break :blk temp;
                } else return error.UnsupportedDType;
                defer if (scale.dtype == .f16) self.allocator.free(scale_f32);

                const scale_data = scale_f32;
                const bias_data = if (bias) |b| blk: {
                    if (b.asConstSlice(f32)) |b32| {
                        break :blk b32;
                    } else if (b.asConstSlice(f16)) |b16| {
                        const temp = try self.allocator.alloc(f32, b16.len);
                        for (b16, temp) |bi, *d| d.* = @floatCast(bi);
                        break :blk temp;
                    } else break :blk null;
                } else null;
                defer if (bias) |b| if (b.dtype == .f16) if (bias_data) |bd| self.allocator.free(bd);
                const out_data = output.asSlice(f32).?;

                for (0..num_groups) |g| {
                    const start = g * norm_size;
                    const end = start + norm_size;
                    const group_in = in_data[start..end];
                    const group_out = out_data[start..end];

                    // Compute mean
                    var sum: f32 = 0;
                    for (group_in) |v| sum += v;
                    const mean = sum / @as(f32, @floatFromInt(norm_size));

                    // Compute variance
                    var var_sum: f32 = 0;
                    for (group_in) |v| {
                        const diff = v - mean;
                        var_sum += diff * diff;
                    }
                    const variance = var_sum / @as(f32, @floatFromInt(norm_size));
                    const inv_std = 1.0 / @sqrt(variance + epsilon);

                    // Normalize and apply scale/bias
                    for (0..norm_size) |i| {
                        const normalized = (group_in[i] - mean) * inv_std;
                        const scaled = normalized * scale_data[i];
                        group_out[i] = if (bias_data) |bd| scaled + bd[i] else scaled;
                    }
                }
            },
            else => return error.UnsupportedDType,
        }

        if (self.buffers[output_idx]) |*existing| {
            existing.deinit();
        }
        self.buffers[output_idx] = output;
    }

    /// Execute GroupQueryAttention (Microsoft extension)
    /// This is a fused multi-head attention with grouped queries
    fn execGroupQueryAttention(self: *Executor, node: Node) !void {
        // GroupQueryAttention inputs:
        // 0: query (batch, seq, num_heads * head_dim)
        // 1: key (batch, seq, kv_num_heads * head_dim)
        // 2: value (batch, seq, kv_num_heads * head_dim)
        // 3: past_key (batch, kv_num_heads, past_seq, head_dim) - optional
        // 4: past_value (batch, kv_num_heads, past_seq, head_dim) - optional
        // ...
        // Outputs:
        // 0: output (batch, seq, num_heads * head_dim)
        // 1: present_key
        // 2: present_value

        if (node.inputs.len < 3 or node.outputs.len < 1) return error.InvalidNode;

        const query = self.buffers[node.inputs[0]] orelse return error.MissingInput;
        const key = self.buffers[node.inputs[1]] orelse return error.MissingInput;
        const value = self.buffers[node.inputs[2]] orelse return error.MissingInput;

        const output_idx = node.outputs[0];

        // For now, implement a simplified version:
        // Standard attention: softmax(Q @ K^T / sqrt(d)) @ V
        // This doesn't handle past KV cache or grouped queries yet

        if (query.dtype != .f32 and query.dtype != .f16) return error.UnsupportedDType;

        // Get dimensions from query shape [batch, seq, hidden]
        if (query.shape.len != 3) return error.InvalidShape;

        const batch_size: usize = @intCast(query.shape[0]);
        const seq_len: usize = @intCast(query.shape[1]);
        const hidden_dim: usize = @intCast(query.shape[2]);

        // Assume num_heads=16, head_dim=64 based on the model (16*64=1024)
        const num_heads: usize = 16;
        const head_dim: usize = hidden_dim / num_heads;

        if (head_dim * num_heads != hidden_dim) return error.InvalidShape;

        // Get key/value dimensions
        const kv_seq_len: usize = @intCast(key.shape[1]);

        // Allocate output with same shape as query (always f32 for computation)
        var output = try RuntimeTensor.alloc(self.allocator, .f32, query.shape);
        errdefer output.deinit();

        // Convert inputs to f32 if needed
        const q_data = if (query.dtype == .f32)
            query.asConstSlice(f32).?
        else blk: {
            const q16 = query.asConstSlice(f16).?;
            const q32 = try self.allocator.alloc(f32, q16.len);
            for (q16, q32) |v, *o| o.* = @floatCast(v);
            break :blk q32;
        };
        defer if (query.dtype != .f32) self.allocator.free(q_data);

        const k_data = if (key.dtype == .f32)
            key.asConstSlice(f32).?
        else blk: {
            const k16 = key.asConstSlice(f16).?;
            const k32 = try self.allocator.alloc(f32, k16.len);
            for (k16, k32) |v, *o| o.* = @floatCast(v);
            break :blk k32;
        };
        defer if (key.dtype != .f32) self.allocator.free(k_data);

        const v_data = if (value.dtype == .f32)
            value.asConstSlice(f32).?
        else blk: {
            const v16 = value.asConstSlice(f16).?;
            const v32 = try self.allocator.alloc(f32, v16.len);
            for (v16, v32) |v, *o| o.* = @floatCast(v);
            break :blk v32;
        };
        defer if (value.dtype != .f32) self.allocator.free(v_data);

        const out_data = output.asSlice(f32).?;

        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

        // Compute attention per batch and head
        for (0..batch_size) |b| {
            for (0..num_heads) |h| {
                for (0..seq_len) |q_pos| {
                    // Compute attention scores for this query position
                    var scores_buf: [512]f32 = undefined;
                    const scores = scores_buf[0..kv_seq_len];

                    // Q[b, q_pos, h*head_dim : (h+1)*head_dim] @ K[b, :, h*head_dim : (h+1)*head_dim]^T
                    for (0..kv_seq_len) |k_pos| {
                        var dot: f32 = 0;
                        for (0..head_dim) |d| {
                            const q_idx = b * seq_len * hidden_dim + q_pos * hidden_dim + h * head_dim + d;
                            const k_idx = b * kv_seq_len * hidden_dim + k_pos * hidden_dim + h * head_dim + d;
                            dot += q_data[q_idx] * k_data[k_idx];
                        }
                        scores[k_pos] = dot * scale;
                    }

                    // Softmax
                    var max_score: f32 = scores[0];
                    for (scores[1..]) |s| {
                        if (s > max_score) max_score = s;
                    }

                    var sum_exp: f32 = 0;
                    for (scores) |*s| {
                        s.* = @exp(s.* - max_score);
                        sum_exp += s.*;
                    }
                    for (scores) |*s| {
                        s.* /= sum_exp;
                    }

                    // Weighted sum of values
                    for (0..head_dim) |d| {
                        var weighted_sum: f32 = 0;
                        for (0..kv_seq_len) |v_pos| {
                            const v_idx = b * kv_seq_len * hidden_dim + v_pos * hidden_dim + h * head_dim + d;
                            weighted_sum += scores[v_pos] * v_data[v_idx];
                        }
                        const out_idx = b * seq_len * hidden_dim + q_pos * hidden_dim + h * head_dim + d;
                        out_data[out_idx] = weighted_sum;
                    }
                }
            }
        }

        if (self.buffers[output_idx]) |*existing| {
            existing.deinit();
        }
        self.buffers[output_idx] = output;

        // Handle present_key and present_value outputs if needed
        if (node.outputs.len > 1) {
            // For now, just copy key to present_key
            var present_key = try RuntimeTensor.alloc(self.allocator, key.dtype, key.shape);
            @memcpy(present_key.data, key.data);
            if (self.buffers[node.outputs[1]]) |*existing| {
                existing.deinit();
            }
            self.buffers[node.outputs[1]] = present_key;
        }

        if (node.outputs.len > 2) {
            // Copy value to present_value
            var present_value = try RuntimeTensor.alloc(self.allocator, value.dtype, value.shape);
            @memcpy(present_value.data, value.data);
            if (self.buffers[node.outputs[2]]) |*existing| {
                existing.deinit();
            }
            self.buffers[node.outputs[2]] = present_value;
        }
    }

    /// Execute Pow operation (element-wise power: base ^ exponent)
    fn execPow(self: *Executor, node: Node) !void {
        if (node.inputs.len < 2 or node.outputs.len < 1) return error.InvalidNode;

        const base = self.buffers[node.inputs[0]] orelse return error.MissingInput;
        const exp_tensor = self.buffers[node.inputs[1]] orelse return error.MissingInput;
        const output_idx = node.outputs[0];

        // Convert base to f32 if needed
        var base_f32: []const f32 = undefined;
        var base_f32_buf: ?[]f32 = null;
        if (base.asConstSlice(f32)) |bf32| {
            base_f32 = bf32;
        } else if (base.asConstSlice(f16)) |bf16| {
            base_f32_buf = try self.allocator.alloc(f32, bf16.len);
            for (bf16, base_f32_buf.?) |src, *dst| dst.* = @floatCast(src);
            base_f32 = base_f32_buf.?;
        } else {
            return error.UnsupportedDType;
        }
        defer if (base_f32_buf) |buf| self.allocator.free(buf);

        // Get exponent value(s) as f32
        var exp_f32: []const f32 = undefined;
        var exp_f32_buf: ?[]f32 = null;
        if (exp_tensor.asConstSlice(f32)) |ef32| {
            exp_f32 = ef32;
        } else if (exp_tensor.asConstSlice(f16)) |ef16| {
            exp_f32_buf = try self.allocator.alloc(f32, ef16.len);
            for (ef16, exp_f32_buf.?) |src, *dst| dst.* = @floatCast(src);
            exp_f32 = exp_f32_buf.?;
        } else {
            return error.UnsupportedDType;
        }
        defer if (exp_f32_buf) |buf| self.allocator.free(buf);

        var output = try RuntimeTensor.alloc(self.allocator, .f32, base.shape);
        errdefer output.deinit();
        const out_data = output.asSlice(f32).?;

        // Check if exponent is a scalar
        if (exp_f32.len == 1) {
            const exp_val = exp_f32[0];
            for (base_f32, 0..) |b, i| {
                out_data[i] = std.math.pow(f32, b, exp_val);
            }
        } else {
            // Element-wise power
            for (base_f32, exp_f32, 0..) |b, e, i| {
                out_data[i] = std.math.pow(f32, b, e);
            }
        }

        if (self.buffers[output_idx]) |*existing| existing.deinit();
        self.buffers[output_idx] = output;
    }

    /// Execute Unsqueeze - insert dimension at specified axes
    fn execUnsqueeze(self: *Executor, node: Node) !void {
        if (node.inputs.len < 1 or node.outputs.len < 1) return error.InvalidNode;

        const input = self.buffers[node.inputs[0]] orelse return error.MissingInput;
        const output_idx = node.outputs[0];

        // Get axes from input[1] or attributes
        var axes_buf: [16]i64 = undefined;
        var num_axes: usize = 0;

        if (node.inputs.len > 1) {
            if (self.buffers[node.inputs[1]]) |axes_tensor| {
                if (axes_tensor.asConstSlice(i64)) |axes| {
                    num_axes = axes.len;
                    for (axes, 0..) |a, i| axes_buf[i] = a;
                }
            }
        }

        if (num_axes == 0) {
            // Get from attributes
            switch (node.attributes) {
                .unsqueeze => |attrs| {
                    if (attrs.axes) |axes| {
                        num_axes = axes.len;
                        for (axes, 0..) |a, i| axes_buf[i] = a;
                    }
                },
                else => return error.InvalidNode,
            }
        }

        if (num_axes == 0) return error.InvalidNode;

        // Calculate new shape
        const new_ndim = input.shape.len + num_axes;
        var new_shape: [16]i64 = undefined;

        // Normalize negative axes and validate
        var sorted_axes: [16]usize = undefined;
        for (0..num_axes) |i| {
            const axis = axes_buf[i];
            const norm: usize = if (axis < 0) @intCast(@as(i64, @intCast(new_ndim)) + axis) else @intCast(axis);
            if (norm >= new_ndim) return error.InvalidAxis;
            sorted_axes[i] = norm;
        }

        // Sort axes
        for (0..num_axes) |i| {
            for (i + 1..num_axes) |j| {
                if (sorted_axes[i] > sorted_axes[j]) {
                    const tmp = sorted_axes[i];
                    sorted_axes[i] = sorted_axes[j];
                    sorted_axes[j] = tmp;
                }
            }
        }

        // Build new shape
        var in_idx: usize = 0;
        var ax_idx: usize = 0;
        for (0..new_ndim) |i| {
            if (ax_idx < num_axes and sorted_axes[ax_idx] == i) {
                new_shape[i] = 1;
                ax_idx += 1;
            } else {
                // Assert we won't access out of bounds
                debug.assertInBounds(in_idx, input.shape.len);
                new_shape[i] = input.shape[in_idx];
                in_idx += 1;
            }
        }

        // Verify we consumed all input dimensions
        std.debug.assert(in_idx == input.shape.len);

        // Create output sharing data with input (just a view with different shape)
        const shape_slice = try self.allocator.dupe(i64, new_shape[0..new_ndim]);
        const output = RuntimeTensor{
            .dtype = input.dtype,
            .shape = shape_slice,
            .data = input.data,
            .numel = input.numel,
            .allocator = self.allocator,
            .owns_data = false, // Shared with input
        };

        debug.assertValidTensor(output);

        if (self.buffers[output_idx]) |*existing| existing.deinit();
        self.buffers[output_idx] = output;
    }

    /// Execute Squeeze - remove dimensions of size 1
    fn execSqueeze(self: *Executor, node: Node) !void {
        if (node.inputs.len < 1 or node.outputs.len < 1) return error.InvalidNode;

        const input = self.buffers[node.inputs[0]] orelse return error.MissingInput;
        debug.assertValidTensor(input);
        const output_idx = node.outputs[0];

        // Get axes to squeeze (if not specified, squeeze all size-1 dims)
        var axes_to_squeeze: [16]bool = [_]bool{false} ** 16;
        var squeeze_all = true;

        if (node.inputs.len > 1) {
            if (self.buffers[node.inputs[1]]) |axes_tensor| {
                if (axes_tensor.asConstSlice(i64)) |axes| {
                    squeeze_all = false;
                    for (axes) |a| {
                        const norm: usize = if (a < 0) @intCast(@as(i64, @intCast(input.shape.len)) + a) else @intCast(a);
                        if (norm < 16) axes_to_squeeze[norm] = true;
                    }
                }
            }
        } else {
            switch (node.attributes) {
                .squeeze => |attrs| {
                    if (attrs.axes) |axes| {
                        if (axes.len > 0) {
                            squeeze_all = false;
                            for (axes) |a| {
                                const norm: usize = if (a < 0) @intCast(@as(i64, @intCast(input.shape.len)) + a) else @intCast(a);
                                if (norm < 16) axes_to_squeeze[norm] = true;
                            }
                        }
                    }
                },
                else => {},
            }
        }

        // Build new shape
        var new_shape: [16]i64 = undefined;
        var new_ndim: usize = 0;
        for (input.shape, 0..) |dim, i| {
            const should_squeeze = if (squeeze_all) (dim == 1) else (axes_to_squeeze[i] and dim == 1);
            if (!should_squeeze) {
                new_shape[new_ndim] = dim;
                new_ndim += 1;
            }
        }

        // Handle edge case: all dims squeezed -> scalar (shape [])
        if (new_ndim == 0) new_ndim = 0; // Keep as 0-dim tensor

        // Create output sharing data with input (just a view with different shape)
        const shape_slice = try self.allocator.dupe(i64, new_shape[0..new_ndim]);
        const output = RuntimeTensor{
            .dtype = input.dtype,
            .shape = shape_slice,
            .data = input.data,
            .numel = input.numel,
            .allocator = self.allocator,
            .owns_data = false, // Shared with input
        };

        if (self.buffers[output_idx]) |*existing| existing.deinit();
        self.buffers[output_idx] = output;
    }

    /// Execute Expand - broadcast tensor to specified shape
    fn execExpand(self: *Executor, node: Node) !void {
        if (node.inputs.len < 2 or node.outputs.len < 1) return error.InvalidNode;

        const input = self.buffers[node.inputs[0]] orelse return error.MissingInput;
        const shape_tensor = self.buffers[node.inputs[1]] orelse return error.MissingInput;
        const output_idx = node.outputs[0];

        const target_shape = shape_tensor.asConstSlice(i64) orelse return error.InvalidNode;

        // Compute output shape (broadcast rules)
        const out_ndim = @max(input.shape.len, target_shape.len);
        var out_shape: [16]i64 = undefined;

        for (0..out_ndim) |i| {
            const in_idx: i64 = @as(i64, @intCast(input.shape.len)) - @as(i64, @intCast(out_ndim)) + @as(i64, @intCast(i));
            const tgt_idx: i64 = @as(i64, @intCast(target_shape.len)) - @as(i64, @intCast(out_ndim)) + @as(i64, @intCast(i));

            const in_dim: i64 = if (in_idx >= 0) input.shape[@intCast(in_idx)] else 1;
            const tgt_dim: i64 = if (tgt_idx >= 0) target_shape[@intCast(tgt_idx)] else 1;

            if (in_dim == tgt_dim or in_dim == 1) {
                out_shape[i] = tgt_dim;
            } else if (tgt_dim == 1) {
                out_shape[i] = in_dim;
            } else {
                return error.InvalidShape;
            }
        }

        // Allocate output
        var output = try RuntimeTensor.alloc(self.allocator, input.dtype, out_shape[0..out_ndim]);
        errdefer output.deinit();

        // Broadcast copy
        const dtype = input.dtype;
        switch (dtype) {
            inline .f32, .f64, .f16, .i32, .i64 => |dt| {
                const T = dt.ZigType();
                const in_data = input.asConstSlice(T).?;
                const out_data = output.asSlice(T).?;

                const out_size = output.numel;
                for (0..out_size) |out_flat| {
                    // Convert flat index to multi-dim, then to input index
                    var remaining = out_flat;
                    var in_flat: usize = 0;
                    var in_stride: usize = 1;

                    // Work backwards through dimensions
                    var d: usize = out_ndim;
                    while (d > 0) {
                        d -= 1;
                        const out_dim: usize = @intCast(out_shape[d]);
                        const coord = remaining % out_dim;
                        remaining /= out_dim;

                        // Map to input dimension
                        const in_dim_offset: i64 = @as(i64, @intCast(d)) - @as(i64, @intCast(out_ndim)) + @as(i64, @intCast(input.shape.len));
                        if (in_dim_offset >= 0) {
                            const in_dim: usize = @intCast(input.shape[@intCast(in_dim_offset)]);
                            const in_coord = if (in_dim == 1) 0 else coord;
                            in_flat += in_coord * in_stride;
                            in_stride *= in_dim;
                        }
                    }

                    out_data[out_flat] = in_data[in_flat];
                }
            },
            else => return error.UnsupportedDType,
        }

        if (self.buffers[output_idx]) |*existing| existing.deinit();
        self.buffers[output_idx] = output;
    }

    /// Execute ConstantOfShape - create tensor filled with constant value
    fn execConstantOfShape(self: *Executor, node: Node) !void {
        if (node.inputs.len < 1 or node.outputs.len < 1) return error.InvalidNode;

        const shape_tensor = self.buffers[node.inputs[0]] orelse return error.MissingInput;
        const output_idx = node.outputs[0];

        const shape = shape_tensor.asConstSlice(i64) orelse return error.InvalidNode;

        // Handle negative dimensions (treat as 0 for empty output)
        var has_invalid_dim = false;
        for (shape) |dim| {
            if (dim < 0) {
                has_invalid_dim = true;
                break;
            }
        }

        // If any dimension is negative, create an empty tensor
        if (has_invalid_dim) {
            var zero_shape: [8]i64 = undefined;
            for (shape, 0..) |dim, i| {
                zero_shape[i] = if (dim < 0) 0 else dim;
            }
            const output = try RuntimeTensor.alloc(self.allocator, .f32, zero_shape[0..shape.len]);
            if (self.buffers[output_idx]) |*existing| existing.deinit();
            self.buffers[output_idx] = output;
            return;
        }

        // Default value is 0.0f32
        var dtype: DType = .f32;
        var fill_value: f32 = 0.0;

        // Check attributes for value tensor
        switch (node.attributes) {
            .constant => |attrs| {
                if (attrs.value) |val| {
                    dtype = val.dtype;
                    if (val.dtype == .f32) {
                        if (val.data.len >= 4) {
                            fill_value = @as(*const f32, @ptrCast(@alignCast(val.data.ptr))).*;
                        }
                    }
                }
            },
            else => {},
        }

        var output = try RuntimeTensor.alloc(self.allocator, dtype, shape);
        errdefer output.deinit();

        // Fill with value
        if (dtype == .f32) {
            const out_data = output.asSlice(f32).?;
            @memset(out_data, fill_value);
        } else if (dtype == .i64) {
            const out_data = output.asSlice(i64).?;
            @memset(out_data, @intFromFloat(fill_value));
        }

        if (self.buffers[output_idx]) |*existing| existing.deinit();
        self.buffers[output_idx] = output;
    }

    /// Execute Range - generate sequence of numbers
    fn execRange(self: *Executor, node: Node) !void {
        if (node.inputs.len < 3 or node.outputs.len < 1) return error.InvalidNode;

        const start_tensor = self.buffers[node.inputs[0]] orelse return error.MissingInput;
        const limit_tensor = self.buffers[node.inputs[1]] orelse return error.MissingInput;
        const delta_tensor = self.buffers[node.inputs[2]] orelse return error.MissingInput;
        const output_idx = node.outputs[0];

        const dtype = start_tensor.dtype;

        switch (dtype) {
            inline .f32, .i32, .i64 => |dt| {
                const T = dt.ZigType();
                const start = start_tensor.asConstSlice(T).?[0];
                const limit = limit_tensor.asConstSlice(T).?[0];
                const delta = delta_tensor.asConstSlice(T).?[0];

                // Calculate number of elements
                const num_elem: usize = blk: {
                    if (@typeInfo(T) == .int) {
                        break :blk @intCast(@divTrunc(limit - start + delta - 1, delta));
                    } else {
                        break :blk @intFromFloat(@ceil((limit - start) / delta));
                    }
                };

                const shape = [_]i64{@intCast(num_elem)};
                var output = try RuntimeTensor.alloc(self.allocator, dtype, &shape);
                errdefer output.deinit();

                const out_data = output.asSlice(T).?;
                var val = start;
                for (out_data) |*o| {
                    o.* = val;
                    val += delta;
                }

                if (self.buffers[output_idx]) |*existing| existing.deinit();
                self.buffers[output_idx] = output;
            },
            else => return error.UnsupportedDType,
        }
    }

    /// Execute Clip - clamp values to [min, max] range
    fn execClip(self: *Executor, node: Node) !void {
        if (node.inputs.len < 1 or node.outputs.len < 1) return error.InvalidNode;

        const input = self.buffers[node.inputs[0]] orelse return error.MissingInput;
        const output_idx = node.outputs[0];

        // Get min/max from inputs or use defaults
        var min_val: f32 = -std.math.inf(f32);
        var max_val: f32 = std.math.inf(f32);

        if (node.inputs.len > 1) {
            if (self.buffers[node.inputs[1]]) |min_tensor| {
                if (min_tensor.asConstSlice(f32)) |m| {
                    if (m.len > 0) min_val = m[0];
                } else if (min_tensor.asConstSlice(f16)) |m| {
                    if (m.len > 0) min_val = @floatCast(m[0]);
                }
            }
        }
        if (node.inputs.len > 2) {
            if (self.buffers[node.inputs[2]]) |max_tensor| {
                if (max_tensor.asConstSlice(f32)) |m| {
                    if (m.len > 0) max_val = m[0];
                } else if (max_tensor.asConstSlice(f16)) |m| {
                    if (m.len > 0) max_val = @floatCast(m[0]);
                }
            }
        }

        var output = try RuntimeTensor.alloc(self.allocator, input.dtype, input.shape);
        errdefer output.deinit();

        switch (input.dtype) {
            .f32 => {
                const in_data = input.asConstSlice(f32).?;
                const out_data = output.asSlice(f32).?;
                for (in_data, out_data) |x, *o| {
                    o.* = @max(min_val, @min(max_val, x));
                }
            },
            .f16 => {
                const in_data = input.asConstSlice(f16).?;
                const out_data = output.asSlice(f16).?;
                const min_f16: f16 = @floatCast(min_val);
                const max_f16: f16 = @floatCast(max_val);
                for (in_data, out_data) |x, *o| {
                    o.* = @max(min_f16, @min(max_f16, x));
                }
            },
            else => return error.UnsupportedDType,
        }

        if (self.buffers[output_idx]) |*existing| existing.deinit();
        self.buffers[output_idx] = output;
    }

    /// Execute comparison ops (Less, Greater, GreaterOrEqual)
    fn execCompare(self: *Executor, node: Node, comptime cmp: enum { less, greater, greater_or_equal, less_or_equal }) !void {
        if (node.inputs.len < 2 or node.outputs.len < 1) return error.InvalidNode;

        const lhs = self.buffers[node.inputs[0]] orelse return error.MissingInput;
        const rhs = self.buffers[node.inputs[1]] orelse return error.MissingInput;
        const output_idx = node.outputs[0];

        // Output is bool (stored as u8)
        var output = try RuntimeTensor.alloc(self.allocator, .bool_, lhs.shape);
        errdefer output.deinit();

        const out_data = output.asSlice(u8).?;
        const rhs_size = rhs.numel;

        switch (lhs.dtype) {
            inline .f32, .i32, .i64 => |dt| {
                const T = dt.ZigType();
                const lhs_data = lhs.asConstSlice(T).?;

                if (rhs_size == 1) {
                    // Scalar broadcast
                    const rhs_scalar = rhs.asConstSlice(T).?[0];
                    for (lhs_data, out_data) |a, *o| {
                        o.* = switch (cmp) {
                            .less => if (a < rhs_scalar) 1 else 0,
                            .greater => if (a > rhs_scalar) 1 else 0,
                            .greater_or_equal => if (a >= rhs_scalar) 1 else 0,
                            .less_or_equal => if (a <= rhs_scalar) 1 else 0,
                        };
                    }
                } else {
                    const rhs_data = rhs.asConstSlice(T).?;
                    for (lhs_data, rhs_data, out_data) |a, b, *o| {
                        o.* = switch (cmp) {
                            .less => if (a < b) 1 else 0,
                            .greater => if (a > b) 1 else 0,
                            .greater_or_equal => if (a >= b) 1 else 0,
                            .less_or_equal => if (a <= b) 1 else 0,
                        };
                    }
                }
            },
            else => return error.UnsupportedDType,
        }

        if (self.buffers[output_idx]) |*existing| existing.deinit();
        self.buffers[output_idx] = output;
    }

    /// Execute ReduceMean
    fn execReduceMean(self: *Executor, node: Node) !void {
        if (node.inputs.len < 1 or node.outputs.len < 1) return error.InvalidNode;

        const input = self.buffers[node.inputs[0]] orelse return error.MissingInput;
        const output_idx = node.outputs[0];

        // Get reduction axes
        var axes_buf: [16]i64 = undefined;
        var num_axes: usize = 0;
        var keepdims: bool = true;

        switch (node.attributes) {
            .reduce => |attrs| {
                if (attrs.axes) |axes| {
                    num_axes = axes.len;
                    for (axes, 0..) |a, i| axes_buf[i] = a;
                }
                keepdims = attrs.keepdims;
            },
            else => {},
        }

        if (num_axes == 0) {
            // Reduce all dimensions
            num_axes = input.shape.len;
            for (0..num_axes) |i| axes_buf[i] = @intCast(i);
        }

        // Normalize negative axes
        for (0..num_axes) |i| {
            if (axes_buf[i] < 0) {
                axes_buf[i] += @intCast(input.shape.len);
            }
        }

        if (input.dtype != .f32) return error.UnsupportedDType;

        // Calculate output shape
        var out_shape: [16]i64 = undefined;
        var out_ndim: usize = 0;

        for (input.shape, 0..) |dim, i| {
            var is_reduced = false;
            for (axes_buf[0..num_axes]) |ax| {
                if (@as(usize, @intCast(ax)) == i) {
                    is_reduced = true;
                    break;
                }
            }
            if (is_reduced) {
                if (keepdims) {
                    out_shape[out_ndim] = 1;
                    out_ndim += 1;
                }
            } else {
                out_shape[out_ndim] = dim;
                out_ndim += 1;
            }
        }

        if (out_ndim == 0) {
            out_shape[0] = 1;
            out_ndim = 1;
        }

        var output = try RuntimeTensor.alloc(self.allocator, input.dtype, out_shape[0..out_ndim]);
        errdefer output.deinit();

        const in_data = input.asConstSlice(f32).?;
        const out_data = output.asSlice(f32).?;

        // Simple implementation: compute mean for each output element
        @memset(out_data, 0);
        var counts = try self.allocator.alloc(usize, out_data.len);
        defer self.allocator.free(counts);
        @memset(counts, 0);

        // For each input element, find its output position and accumulate
        const in_size = input.numel;
        for (0..in_size) |in_flat| {
            // Convert flat index to multi-dim
            var remaining = in_flat;
            var coords: [16]usize = undefined;
            var d: usize = input.shape.len;
            while (d > 0) {
                d -= 1;
                const dim: usize = @intCast(input.shape[d]);
                coords[d] = remaining % dim;
                remaining /= dim;
            }

            // Compute output flat index (skip reduced dims if !keepdims)
            var out_flat: usize = 0;
            var out_stride: usize = 1;
            var out_d: usize = out_ndim;
            d = input.shape.len;
            while (d > 0) {
                d -= 1;
                var is_reduced = false;
                for (axes_buf[0..num_axes]) |ax| {
                    if (@as(usize, @intCast(ax)) == d) {
                        is_reduced = true;
                        break;
                    }
                }
                if (!is_reduced or keepdims) {
                    out_d -= 1;
                    const coord = if (is_reduced) 0 else coords[d];
                    out_flat += coord * out_stride;
                    const out_dim: usize = @intCast(out_shape[out_d]);
                    out_stride *= out_dim;
                }
            }

            out_data[out_flat] += in_data[in_flat];
            counts[out_flat] += 1;
        }

        // Divide by counts
        for (out_data, counts) |*o, c| {
            if (c > 0) o.* /= @floatFromInt(c);
        }

        if (self.buffers[output_idx]) |*existing| existing.deinit();
        self.buffers[output_idx] = output;
    }

    /// Execute ReduceMax
    fn execReduceMax(self: *Executor, node: Node) !void {
        if (node.inputs.len < 1 or node.outputs.len < 1) return error.InvalidNode;

        const input = self.buffers[node.inputs[0]] orelse return error.MissingInput;
        const output_idx = node.outputs[0];

        // Get reduction axes
        var axes_buf: [16]i64 = undefined;
        var num_axes: usize = 0;
        var keepdims: bool = true;

        switch (node.attributes) {
            .reduce => |attrs| {
                if (attrs.axes) |axes| {
                    num_axes = axes.len;
                    for (axes, 0..) |a, i| axes_buf[i] = a;
                }
                keepdims = attrs.keepdims;
            },
            else => {},
        }

        // Also check for axes from input tensor (ONNX opset >= 18)
        if (num_axes == 0 and node.inputs.len > 1) {
            if (self.buffers[node.inputs[1]]) |axes_tensor| {
                if (axes_tensor.asConstSlice(i64)) |axes| {
                    num_axes = axes.len;
                    for (axes, 0..) |a, i| axes_buf[i] = a;
                }
            }
        }

        if (num_axes == 0) {
            // Reduce all dimensions
            num_axes = input.shape.len;
            for (0..num_axes) |i| axes_buf[i] = @intCast(i);
        }

        // Normalize negative axes
        for (0..num_axes) |i| {
            if (axes_buf[i] < 0) {
                axes_buf[i] += @intCast(input.shape.len);
            }
        }

        // Calculate output shape
        var out_shape: [16]i64 = undefined;
        var out_ndim: usize = 0;

        for (input.shape, 0..) |dim, i| {
            var is_reduced = false;
            for (axes_buf[0..num_axes]) |ax| {
                if (@as(usize, @intCast(ax)) == i) {
                    is_reduced = true;
                    break;
                }
            }
            if (is_reduced) {
                if (keepdims) {
                    out_shape[out_ndim] = 1;
                    out_ndim += 1;
                }
            } else {
                out_shape[out_ndim] = dim;
                out_ndim += 1;
            }
        }

        if (out_ndim == 0) {
            out_shape[0] = 1;
            out_ndim = 1;
        }

        switch (input.dtype) {
            inline .f32, .f16, .i32, .i64 => |dt| {
                const T = dt.ZigType();
                var output = try RuntimeTensor.alloc(self.allocator, input.dtype, out_shape[0..out_ndim]);
                errdefer output.deinit();

                const in_data = input.asConstSlice(T).?;
                const out_data = output.asSlice(T).?;

                // Initialize with minimum value
                const min_val = if (@typeInfo(T) == .int) std.math.minInt(T) else -std.math.inf(T);
                @memset(out_data, min_val);

                // For each input element, find its output position and take max
                const in_size = input.numel;
                for (0..in_size) |in_flat| {
                    // Convert flat index to multi-dim
                    var remaining = in_flat;
                    var coords: [16]usize = undefined;
                    var d: usize = input.shape.len;
                    while (d > 0) {
                        d -= 1;
                        const dim: usize = @intCast(input.shape[d]);
                        coords[d] = remaining % dim;
                        remaining /= dim;
                    }

                    // Compute output flat index
                    var out_flat: usize = 0;
                    var out_stride: usize = 1;
                    var out_d: usize = out_ndim;
                    d = input.shape.len;
                    while (d > 0) {
                        d -= 1;
                        var is_reduced = false;
                        for (axes_buf[0..num_axes]) |ax| {
                            if (@as(usize, @intCast(ax)) == d) {
                                is_reduced = true;
                                break;
                            }
                        }
                        if (!is_reduced or keepdims) {
                            out_d -= 1;
                            const coord = if (is_reduced) 0 else coords[d];
                            out_flat += coord * out_stride;
                            const out_dim: usize = @intCast(out_shape[out_d]);
                            out_stride *= out_dim;
                        }
                    }

                    if (in_data[in_flat] > out_data[out_flat]) {
                        out_data[out_flat] = in_data[in_flat];
                    }
                }

                if (self.buffers[output_idx]) |*existing| existing.deinit();
                self.buffers[output_idx] = output;
            },
            else => return error.UnsupportedDType,
        }
    }

    /// Execute ReduceL2 - L2 norm reduction
    fn execReduceL2(self: *Executor, node: Node) !void {
        if (node.inputs.len < 1 or node.outputs.len < 1) return error.InvalidNode;

        const input = self.buffers[node.inputs[0]] orelse return error.MissingInput;
        const output_idx = node.outputs[0];

        var axes_buf: [16]i64 = undefined;
        var num_axes: usize = 0;
        var keepdims: bool = true;

        switch (node.attributes) {
            .reduce => |attrs| {
                if (attrs.axes) |axes| {
                    num_axes = axes.len;
                    for (axes, 0..) |a, i| axes_buf[i] = a;
                }
                keepdims = attrs.keepdims;
            },
            else => {},
        }

        if (num_axes == 0) {
            num_axes = input.shape.len;
            for (0..num_axes) |i| axes_buf[i] = @intCast(i);
        }

        for (0..num_axes) |i| {
            if (axes_buf[i] < 0) {
                axes_buf[i] += @intCast(input.shape.len);
            }
        }

        if (input.dtype != .f32) return error.UnsupportedDType;

        var out_shape: [16]i64 = undefined;
        var out_ndim: usize = 0;

        for (input.shape, 0..) |dim, i| {
            var is_reduced = false;
            for (axes_buf[0..num_axes]) |ax| {
                if (@as(usize, @intCast(ax)) == i) {
                    is_reduced = true;
                    break;
                }
            }
            if (is_reduced) {
                if (keepdims) {
                    out_shape[out_ndim] = 1;
                    out_ndim += 1;
                }
            } else {
                out_shape[out_ndim] = dim;
                out_ndim += 1;
            }
        }

        if (out_ndim == 0) {
            out_shape[0] = 1;
            out_ndim = 1;
        }

        var output = try RuntimeTensor.alloc(self.allocator, input.dtype, out_shape[0..out_ndim]);
        errdefer output.deinit();

        const in_data = input.asConstSlice(f32).?;
        const out_data = output.asSlice(f32).?;

        @memset(out_data, 0);

        const in_size = input.numel;
        for (0..in_size) |in_flat| {
            var remaining = in_flat;
            var coords: [16]usize = undefined;
            var d: usize = input.shape.len;
            while (d > 0) {
                d -= 1;
                const dim: usize = @intCast(input.shape[d]);
                coords[d] = remaining % dim;
                remaining /= dim;
            }

            var out_flat: usize = 0;
            var out_stride: usize = 1;
            var out_d: usize = out_ndim;
            d = input.shape.len;
            while (d > 0) {
                d -= 1;
                var is_reduced = false;
                for (axes_buf[0..num_axes]) |ax| {
                    if (@as(usize, @intCast(ax)) == d) {
                        is_reduced = true;
                        break;
                    }
                }
                if (!is_reduced or keepdims) {
                    out_d -= 1;
                    const coord = if (is_reduced) 0 else coords[d];
                    out_flat += coord * out_stride;
                    const out_dim: usize = @intCast(out_shape[out_d]);
                    out_stride *= out_dim;
                }
            }

            // Accumulate squared values
            out_data[out_flat] += in_data[in_flat] * in_data[in_flat];
        }

        // Take square root
        for (out_data) |*o| {
            o.* = @sqrt(o.*);
        }

        if (self.buffers[output_idx]) |*existing| existing.deinit();
        self.buffers[output_idx] = output;
    }

    /// Execute Tile - repeat tensor along axes
    fn execTile(self: *Executor, node: Node) !void {
        if (node.inputs.len < 2 or node.outputs.len < 1) return error.InvalidNode;

        const input = self.buffers[node.inputs[0]] orelse return error.MissingInput;
        const repeats_tensor = self.buffers[node.inputs[1]] orelse return error.MissingInput;
        const output_idx = node.outputs[0];

        const repeats = repeats_tensor.asConstSlice(i64) orelse return error.UnsupportedDType;

        if (repeats.len != input.shape.len) return error.ShapeMismatch;

        // Calculate output shape
        var out_shape_buf: [8]i64 = undefined;
        for (input.shape, 0..) |dim, i| {
            out_shape_buf[i] = dim * repeats[i];
        }
        const out_shape = out_shape_buf[0..input.shape.len];

        var output = try RuntimeTensor.alloc(self.allocator, input.dtype, out_shape);
        errdefer output.deinit();

        // Calculate input strides
        var in_strides: [8]usize = undefined;
        var stride: usize = 1;
        var d: usize = input.shape.len;
        while (d > 0) {
            d -= 1;
            in_strides[d] = stride;
            stride *= @intCast(input.shape[d]);
        }

        // Calculate output strides
        var out_strides: [8]usize = undefined;
        stride = 1;
        d = input.shape.len;
        while (d > 0) {
            d -= 1;
            out_strides[d] = stride;
            stride *= @intCast(out_shape[d]);
        }

        const numel = output.numel;
        const ndim = input.shape.len;

        switch (input.dtype) {
            inline .f32, .f16, .i32, .i64 => |dt| {
                const T = dt.ZigType();
                const in_data = input.asConstSlice(T).?;
                const out_data = output.asSlice(T).?;

                for (0..numel) |out_flat| {
                    // Convert output flat index to coordinates
                    var remaining = out_flat;
                    var in_flat: usize = 0;

                    for (0..ndim) |di| {
                        const out_coord = remaining / out_strides[di];
                        remaining %= out_strides[di];

                        // Map to input coordinate (wrap around)
                        const in_dim: usize = @intCast(input.shape[di]);
                        const in_coord = out_coord % in_dim;
                        in_flat += in_coord * in_strides[di];
                    }

                    out_data[out_flat] = in_data[in_flat];
                }
            },
            else => return error.UnsupportedDType,
        }

        if (self.buffers[output_idx]) |*existing| existing.deinit();
        self.buffers[output_idx] = output;
    }

    /// Execute Gemm: Y = alpha * A @ B + beta * C
    fn execGemm(self: *Executor, node: Node) !void {
        if (node.inputs.len < 2 or node.outputs.len < 1) return error.InvalidNode;

        const a = self.buffers[node.inputs[0]] orelse return error.MissingInput;
        const b = self.buffers[node.inputs[1]] orelse return error.MissingInput;
        const output_idx = node.outputs[0];

        // Get attributes
        var alpha: f32 = 1.0;
        var beta: f32 = 1.0;
        var transA = false;
        var transB = false;

        switch (node.attributes) {
            .gemm => |attrs| {
                alpha = attrs.alpha;
                beta = attrs.beta;
                transA = attrs.transA;
                transB = attrs.transB;
            },
            else => {},
        }

        if (a.dtype != .f32) return error.UnsupportedDType;

        // Get dimensions
        const M: usize = if (transA) @intCast(a.shape[1]) else @intCast(a.shape[0]);
        const K: usize = if (transA) @intCast(a.shape[0]) else @intCast(a.shape[1]);
        const N: usize = if (transB) @intCast(b.shape[0]) else @intCast(b.shape[1]);

        const out_shape = [_]i64{ @intCast(M), @intCast(N) };
        var output = try RuntimeTensor.alloc(self.allocator, a.dtype, &out_shape);
        errdefer output.deinit();

        const a_data = a.asConstSlice(f32).?;
        const b_data = b.asConstSlice(f32).?;
        const out_data = output.asSlice(f32).?;

        // Initialize with beta * C if present
        if (node.inputs.len > 2) {
            if (self.buffers[node.inputs[2]]) |c| {
                const c_data = c.asConstSlice(f32).?;
                const c_size = c.numel;
                // Broadcast C to output shape
                for (0..M) |i| {
                    for (0..N) |j| {
                        const out_idx = i * N + j;
                        const c_idx = if (c_size == 1) 0 else if (c.shape.len == 1) j else out_idx;
                        out_data[out_idx] = beta * c_data[c_idx % c_size];
                    }
                }
            } else {
                @memset(out_data, 0);
            }
        } else {
            @memset(out_data, 0);
        }

        // Compute alpha * A @ B
        const lda: usize = @intCast(a.shape[1]);
        const ldb: usize = @intCast(b.shape[1]);

        for (0..M) |i| {
            for (0..N) |j| {
                var sum: f32 = 0;
                for (0..K) |k| {
                    const a_idx = if (transA) k * lda + i else i * lda + k;
                    const b_idx = if (transB) j * ldb + k else k * ldb + j;
                    sum += a_data[a_idx] * b_data[b_idx];
                }
                out_data[i * N + j] += alpha * sum;
            }
        }

        if (self.buffers[output_idx]) |*existing| existing.deinit();
        self.buffers[output_idx] = output;
    }

    /// Execute Pad operation
    fn execPad(self: *Executor, node: Node) !void {
        if (node.inputs.len < 1 or node.outputs.len < 1) return error.InvalidNode;

        const input = self.buffers[node.inputs[0]] orelse return error.MissingInput;
        const output_idx = node.outputs[0];

        // Get pads from input tensor or attributes
        var pads_buf: [32]i64 = undefined;
        var constant_value: f32 = 0;

        if (node.inputs.len > 1) {
            if (self.buffers[node.inputs[1]]) |pads_tensor| {
                if (pads_tensor.asConstSlice(i64)) |pads| {
                    for (pads, 0..) |p, i| pads_buf[i] = p;
                }
            }
        }

        if (node.inputs.len > 2) {
            if (self.buffers[node.inputs[2]]) |val_tensor| {
                if (val_tensor.asConstSlice(f32)) |v| {
                    if (v.len > 0) constant_value = v[0];
                }
            }
        }

        // Calculate output shape
        const ndim = input.shape.len;
        var out_shape: [16]i64 = undefined;
        for (input.shape, 0..) |dim, i| {
            out_shape[i] = dim + pads_buf[i] + pads_buf[ndim + i];
        }

        var output = try RuntimeTensor.alloc(self.allocator, input.dtype, out_shape[0..ndim]);
        errdefer output.deinit();

        switch (input.dtype) {
            inline .f32, .f16 => |dtype| {
                const T = dtype.ZigType();
                const out_data = output.asSlice(T).?;
                @memset(out_data, @as(T, @floatCast(constant_value)));

                // Copy input data to output with offset
                const in_data = input.asConstSlice(T).?;
                const in_size = input.numel;

                for (0..in_size) |in_flat| {
                    // Convert to coords
                    var remaining = in_flat;
                    var coords: [16]usize = undefined;
                    var d: usize = ndim;
                    while (d > 0) {
                        d -= 1;
                        const dim: usize = @intCast(input.shape[d]);
                        coords[d] = remaining % dim;
                        remaining /= dim;
                    }

                    // Add pad offset and compute output flat index
                    var out_flat: usize = 0;
                    var stride: usize = 1;
                    d = ndim;
                    while (d > 0) {
                        d -= 1;
                        const out_coord = coords[d] + @as(usize, @intCast(pads_buf[d]));
                        out_flat += out_coord * stride;
                        stride *= @intCast(out_shape[d]);
                    }

                    out_data[out_flat] = in_data[in_flat];
                }
            },
            else => return error.UnsupportedDType,
        }

        if (self.buffers[output_idx]) |*existing| existing.deinit();
        self.buffers[output_idx] = output;
    }

    /// Execute STFT (Short-Time Fourier Transform)
    fn execSTFT(self: *Executor, node: Node) !void {
        // STFT inputs:
        // 0: signal [batch, signal_length] or [batch, signal_length, 1]
        // 1: frame_step (scalar i64)
        // 2: window [frame_length] (optional)
        // 3: frame_length (scalar i64, optional)
        if (node.inputs.len < 2 or node.outputs.len < 1) return error.InvalidNode;

        const signal = self.buffers[node.inputs[0]] orelse return error.MissingInput;
        const frame_step_tensor = self.buffers[node.inputs[1]] orelse return error.MissingInput;
        const output_idx = node.outputs[0];

        const frame_step: usize = blk: {
            if (frame_step_tensor.asConstSlice(i64)) |s| {
                break :blk @intCast(s[0]);
            } else {
                return error.UnsupportedDType;
            }
        };

        // Get window if provided
        var window: ?[]const f32 = null;
        var window_f16_buf: ?[]f32 = null;
        if (node.inputs.len > 2) {
            if (self.buffers[node.inputs[2]]) |w| {
                if (w.asConstSlice(f32)) |wf32| {
                    window = wf32;
                } else if (w.asConstSlice(f16)) |wf16| {
                    window_f16_buf = try self.allocator.alloc(f32, wf16.len);
                    for (wf16, window_f16_buf.?) |src, *dst| dst.* = @floatCast(src);
                    window = window_f16_buf.?;
                }
            }
        }
        defer if (window_f16_buf) |buf| self.allocator.free(buf);

        // Get frame_length
        const frame_length: usize = blk: {
            if (node.inputs.len > 3) {
                if (self.buffers[node.inputs[3]]) |fl| {
                    if (fl.asConstSlice(i64)) |s| {
                        break :blk @intCast(s[0]);
                    }
                }
            }
            if (window) |w| break :blk w.len;
            break :blk 256; // default
        };

        // Get signal dimensions
        const batch_size: usize = @intCast(signal.shape[0]);
        const signal_length: usize = @intCast(signal.shape[1]);

        // Calculate number of frames
        const num_frames = (signal_length - frame_length) / frame_step + 1;
        const fft_length = frame_length; // assume no padding
        const output_bins = fft_length / 2 + 1;

        // Output shape: [batch, num_frames, output_bins, 2]
        const out_shape = [_]i64{
            @intCast(batch_size),
            @intCast(num_frames),
            @intCast(output_bins),
            2, // real and imag
        };

        var output = try RuntimeTensor.alloc(self.allocator, .f32, &out_shape);
        errdefer output.deinit();
        const out_data = output.asSlice(f32).?;

        // Get input signal as f32
        var signal_f32: []const f32 = undefined;
        var signal_f32_buf: ?[]f32 = null;
        if (signal.asConstSlice(f32)) |sf32| {
            signal_f32 = sf32;
        } else if (signal.asConstSlice(f16)) |sf16| {
            signal_f32_buf = try self.allocator.alloc(f32, sf16.len);
            for (sf16, signal_f32_buf.?) |src, *dst| dst.* = @floatCast(src);
            signal_f32 = signal_f32_buf.?;
        } else {
            return error.UnsupportedDType;
        }
        defer if (signal_f32_buf) |buf| self.allocator.free(buf);

        // Allocate temp buffer for windowed frame
        const frame_buf = try self.allocator.alloc(f32, frame_length);
        defer self.allocator.free(frame_buf);

        // Process each batch
        for (0..batch_size) |b| {
            const sig_offset = b * signal_length;

            for (0..num_frames) |f| {
                const frame_start = f * frame_step;

                // Extract and window the frame
                for (0..frame_length) |i| {
                    const sample = signal_f32[sig_offset + frame_start + i];
                    frame_buf[i] = if (window) |w| sample * w[i] else sample;
                }

                // Simple DFT (O(n²) - not optimal but works)
                // For production, would use FFT
                for (0..output_bins) |k| {
                    var sum_real: f32 = 0;
                    var sum_imag: f32 = 0;
                    const k_f: f32 = @floatFromInt(k);
                    const n_f: f32 = @floatFromInt(frame_length);

                    for (0..frame_length) |n| {
                        const n_idx: f32 = @floatFromInt(n);
                        const angle = -2.0 * std.math.pi * k_f * n_idx / n_f;
                        sum_real += frame_buf[n] * @cos(angle);
                        sum_imag += frame_buf[n] * @sin(angle);
                    }

                    // Output index: [b, f, k, real/imag]
                    const out_idx = ((b * num_frames + f) * output_bins + k) * 2;
                    out_data[out_idx] = sum_real;
                    out_data[out_idx + 1] = sum_imag;
                }
            }
        }

        if (self.buffers[output_idx]) |*existing| existing.deinit();
        self.buffers[output_idx] = output;
    }

    /// Execute Conv (1D/2D convolution)
    fn execConv(self: *Executor, node: Node) !void {
        // Conv inputs:
        // 0: X [batch, in_channels, ...]
        // 1: W [out_channels, in_channels/group, ...]
        // 2: B (optional) [out_channels]
        if (node.inputs.len < 2 or node.outputs.len < 1) return error.InvalidNode;

        const x = self.buffers[node.inputs[0]] orelse return error.MissingInput;
        const w = self.buffers[node.inputs[1]] orelse return error.MissingInput;
        const output_idx = node.outputs[0];

        // Get optional bias
        var bias: ?*const RuntimeTensor = null;
        if (node.inputs.len > 2) {
            bias = if (self.buffers[node.inputs[2]]) |*b| b else null;
        }

        const attrs = node.attributes.conv;
        const group: usize = @intCast(attrs.group);

        // Convert inputs to f32 if needed
        var x_f32: []const f32 = undefined;
        var x_f32_buf: ?[]f32 = null;
        if (x.asConstSlice(f32)) |xf32| {
            x_f32 = xf32;
        } else if (x.asConstSlice(f16)) |xf16| {
            x_f32_buf = try self.allocator.alloc(f32, xf16.len);
            for (xf16, x_f32_buf.?) |src, *dst| dst.* = @floatCast(src);
            x_f32 = x_f32_buf.?;
        } else {
            return error.UnsupportedDType;
        }
        defer if (x_f32_buf) |buf| self.allocator.free(buf);

        var w_f32: []const f32 = undefined;
        var w_f32_buf: ?[]f32 = null;
        if (w.asConstSlice(f32)) |wf32| {
            w_f32 = wf32;
        } else if (w.asConstSlice(f16)) |wf16| {
            w_f32_buf = try self.allocator.alloc(f32, wf16.len);
            for (wf16, w_f32_buf.?) |src, *dst| dst.* = @floatCast(src);
            w_f32 = w_f32_buf.?;
        } else {
            return error.UnsupportedDType;
        }
        defer if (w_f32_buf) |buf| self.allocator.free(buf);

        var bias_f32: ?[]const f32 = null;
        var bias_f32_buf: ?[]f32 = null;
        if (bias) |b| {
            if (b.asConstSlice(f32)) |bf32| {
                bias_f32 = bf32;
            } else if (b.asConstSlice(f16)) |bf16| {
                bias_f32_buf = try self.allocator.alloc(f32, bf16.len);
                for (bf16, bias_f32_buf.?) |src, *dst| dst.* = @floatCast(src);
                bias_f32 = bias_f32_buf.?;
            }
        }
        defer if (bias_f32_buf) |buf| self.allocator.free(buf);

        // Check dimensionality (support 1D and 2D convolutions, requiring 3D or 4D input)
        const spatial_dims = x.shape.len - 2;
        if (spatial_dims != 1 and spatial_dims != 2) {
            std.debug.print("Conv requires 3D or 4D input, got {}D input with shape {any}\n", .{ x.shape.len, x.shape });
            return error.InvalidShape;
        }

        const batch: usize = @intCast(x.shape[0]);
        const in_channels: usize = @intCast(x.shape[1]);
        const out_channels: usize = @intCast(w.shape[0]);
        const weight_in_channels: usize = @intCast(w.shape[1]);

        // Validate channel dimensions match (input_channels == weight_in_channels * groups)
        if (in_channels != weight_in_channels * group) {
            std.debug.print("Conv channel mismatch: input has {} channels but weight expects {} * {} = {}\n", .{
                in_channels,
                weight_in_channels,
                group,
                weight_in_channels * group,
            });
            return error.InvalidShape;
        }

        // Get strides, pads, dilations (defaults to 1s and 0s)
        const default_ones = [_]i64{ 1, 1 };
        const default_zeros = [_]i64{ 0, 0, 0, 0 };

        const strides = attrs.strides orelse &default_ones;
        const pads = attrs.pads orelse &default_zeros;
        const dilations = attrs.dilations orelse &default_ones;

        if (spatial_dims == 1) {
            // 1D Convolution
            const in_length: usize = @intCast(x.shape[2]);
            const kernel_size: usize = @intCast(w.shape[2]);
            // Validate strides and pads for garbage values
            const raw_stride: i64 = if (strides.len > 0) strides[0] else 1;
            const stride: usize = if (raw_stride > 0 and raw_stride < 1000) @intCast(raw_stride) else 1;
            const raw_pad_left: i64 = if (pads.len > 0) pads[0] else 0;
            const pad_left: usize = if (raw_pad_left >= 0 and raw_pad_left < 10000) @intCast(raw_pad_left) else 0;
            const raw_pad_right: i64 = if (pads.len > 1) pads[1] else raw_pad_left;
            const pad_right: usize = if (raw_pad_right >= 0 and raw_pad_right < 10000) @intCast(raw_pad_right) else pad_left;
            // Get dilation with validation for garbage values
            const raw_dilation: i64 = if (dilations.len > 0) dilations[0] else 1;
            const dilation: usize = if (raw_dilation > 0 and raw_dilation < 1000)
                @intCast(raw_dilation)
            else
                1;

            if (kernel_size == 0) return error.InvalidNode;

            const dilated_kernel = (kernel_size - 1) * dilation + 1;
            const padded_length = in_length + pad_left + pad_right;
            const out_length = if (padded_length >= dilated_kernel)
                (padded_length - dilated_kernel) / stride + 1
            else
                0;

            const out_shape = [_]i64{
                @intCast(batch),
                @intCast(out_channels),
                @intCast(out_length),
            };

            var output = try RuntimeTensor.alloc(self.allocator, .f32, &out_shape);
            errdefer output.deinit();
            const out_data = output.asSlice(f32).?;

            const in_c_per_group = in_channels / group;
            const out_c_per_group = out_channels / group;

            // Perform convolution
            for (0..batch) |b| {
                for (0..out_channels) |oc| {
                    const g = oc / out_c_per_group;
                    const oc_in_group = oc % out_c_per_group;

                    for (0..out_length) |ol| {
                        var sum: f32 = 0;
                        const in_start = ol * stride;

                        for (0..in_c_per_group) |ic_g| {
                            const ic = g * in_c_per_group + ic_g;

                            for (0..kernel_size) |k| {
                                const in_pos_signed: i64 = @as(i64, @intCast(in_start + k * dilation)) - @as(i64, @intCast(pad_left));
                                if (in_pos_signed >= 0 and in_pos_signed < @as(i64, @intCast(in_length))) {
                                    const in_pos: usize = @intCast(in_pos_signed);
                                    const x_idx = (b * in_channels + ic) * in_length + in_pos;
                                    const w_idx = (oc * in_c_per_group + ic_g) * kernel_size + k;
                                    _ = oc_in_group;
                                    sum += x_f32[x_idx] * w_f32[w_idx];
                                }
                            }
                        }

                        if (bias_f32) |bf| {
                            sum += bf[oc];
                        }

                        const out_idx = (b * out_channels + oc) * out_length + ol;
                        out_data[out_idx] = sum;
                    }
                }
            }

            if (self.buffers[output_idx]) |*existing| existing.deinit();
            self.buffers[output_idx] = output;
        } else {
            // 2D Convolution
            const in_height: usize = @intCast(x.shape[2]);
            const in_width: usize = @intCast(x.shape[3]);
            const kernel_h: usize = @intCast(w.shape[2]);
            const kernel_w: usize = @intCast(w.shape[3]);

            // Validate and extract strides
            const raw_stride_h: i64 = if (strides.len > 0) strides[0] else 1;
            const raw_stride_w: i64 = if (strides.len > 1) strides[1] else raw_stride_h;
            const stride_h: usize = if (raw_stride_h > 0 and raw_stride_h < 1000) @intCast(raw_stride_h) else 1;
            const stride_w: usize = if (raw_stride_w > 0 and raw_stride_w < 1000) @intCast(raw_stride_w) else 1;

            // Validate and extract pads [top, left, bottom, right] or [top, left] repeated
            const raw_pad_top: i64 = if (pads.len > 0) pads[0] else 0;
            const raw_pad_left: i64 = if (pads.len > 1) pads[1] else raw_pad_top;
            const raw_pad_bottom: i64 = if (pads.len > 2) pads[2] else raw_pad_top;
            const raw_pad_right: i64 = if (pads.len > 3) pads[3] else raw_pad_left;
            const pad_top: usize = if (raw_pad_top >= 0 and raw_pad_top < 10000) @intCast(raw_pad_top) else 0;
            const pad_left: usize = if (raw_pad_left >= 0 and raw_pad_left < 10000) @intCast(raw_pad_left) else 0;
            const pad_bottom: usize = if (raw_pad_bottom >= 0 and raw_pad_bottom < 10000) @intCast(raw_pad_bottom) else 0;
            const pad_right: usize = if (raw_pad_right >= 0 and raw_pad_right < 10000) @intCast(raw_pad_right) else 0;

            // Validate and extract dilations
            const raw_dilation_h: i64 = if (dilations.len > 0) dilations[0] else 1;
            const raw_dilation_w: i64 = if (dilations.len > 1) dilations[1] else raw_dilation_h;
            const dilation_h: usize = if (raw_dilation_h > 0 and raw_dilation_h < 1000) @intCast(raw_dilation_h) else 1;
            const dilation_w: usize = if (raw_dilation_w > 0 and raw_dilation_w < 1000) @intCast(raw_dilation_w) else 1;

            if (kernel_h == 0 or kernel_w == 0) return error.InvalidNode;

            // Calculate output dimensions
            const dilated_kernel_h = (kernel_h - 1) * dilation_h + 1;
            const dilated_kernel_w = (kernel_w - 1) * dilation_w + 1;
            const padded_height = in_height + pad_top + pad_bottom;
            const padded_width = in_width + pad_left + pad_right;
            const out_height = if (padded_height >= dilated_kernel_h)
                (padded_height - dilated_kernel_h) / stride_h + 1
            else
                0;
            const out_width = if (padded_width >= dilated_kernel_w)
                (padded_width - dilated_kernel_w) / stride_w + 1
            else
                0;

            const out_shape = [_]i64{
                @intCast(batch),
                @intCast(out_channels),
                @intCast(out_height),
                @intCast(out_width),
            };

            var output = try RuntimeTensor.alloc(self.allocator, .f32, &out_shape);
            errdefer output.deinit();
            const out_data = output.asSlice(f32).?;

            const in_c_per_group = in_channels / group;
            const out_c_per_group = out_channels / group;

            // Perform 2D convolution
            for (0..batch) |b| {
                for (0..out_channels) |oc| {
                    const g = oc / out_c_per_group;

                    for (0..out_height) |oh| {
                        for (0..out_width) |ow| {
                            var sum: f32 = 0;
                            const in_start_h = oh * stride_h;
                            const in_start_w = ow * stride_w;

                            for (0..in_c_per_group) |ic_g| {
                                const ic = g * in_c_per_group + ic_g;

                                for (0..kernel_h) |kh| {
                                    for (0..kernel_w) |kw| {
                                        const in_h_signed: i64 = @as(i64, @intCast(in_start_h + kh * dilation_h)) - @as(i64, @intCast(pad_top));
                                        const in_w_signed: i64 = @as(i64, @intCast(in_start_w + kw * dilation_w)) - @as(i64, @intCast(pad_left));

                                        if (in_h_signed >= 0 and in_h_signed < @as(i64, @intCast(in_height)) and
                                            in_w_signed >= 0 and in_w_signed < @as(i64, @intCast(in_width)))
                                        {
                                            const in_h: usize = @intCast(in_h_signed);
                                            const in_w: usize = @intCast(in_w_signed);
                                            const x_idx = ((b * in_channels + ic) * in_height + in_h) * in_width + in_w;
                                            const w_idx = ((oc * in_c_per_group + ic_g) * kernel_h + kh) * kernel_w + kw;
                                            sum += x_f32[x_idx] * w_f32[w_idx];
                                        }
                                    }
                                }
                            }

                            if (bias_f32) |bf| {
                                sum += bf[oc];
                            }

                            const out_idx = ((b * out_channels + oc) * out_height + oh) * out_width + ow;
                            out_data[out_idx] = sum;
                        }
                    }
                }
            }

            if (self.buffers[output_idx]) |*existing| existing.deinit();
            self.buffers[output_idx] = output;
        }
    }

    /// Execute NonZero - find indices of non-zero elements
    fn execNonZero(self: *Executor, node: Node) !void {
        if (node.inputs.len < 1 or node.outputs.len < 1) return error.InvalidNode;

        const input = self.buffers[node.inputs[0]] orelse return error.MissingInput;
        const output_idx = node.outputs[0];

        const ndim = input.shape.len;

        // First pass: count non-zero elements
        var count: usize = 0;
        if (input.asConstSlice(f32)) |data| {
            for (data) |v| {
                if (v != 0) count += 1;
            }
        } else if (input.asConstSlice(f16)) |data| {
            for (data) |v| {
                if (v != @as(f16, 0)) count += 1;
            }
        } else if (input.asConstSlice(i64)) |data| {
            for (data) |v| {
                if (v != 0) count += 1;
            }
        } else if (input.asConstSlice(i32)) |data| {
            for (data) |v| {
                if (v != 0) count += 1;
            }
        } else {
            return error.UnsupportedDType;
        }

        // Output shape: [ndim, count]
        const out_shape = [_]i64{ @intCast(ndim), @intCast(count) };
        var output = try RuntimeTensor.alloc(self.allocator, .i64, &out_shape);
        errdefer output.deinit();

        // If no non-zero elements, we're done
        if (count == 0) {
            if (self.buffers[output_idx]) |*existing| existing.deinit();
            self.buffers[output_idx] = output;
            return;
        }

        const out_data = output.asSlice(i64).?;

        // Calculate strides for multi-dimensional indexing
        var strides: [8]usize = undefined;
        var stride: usize = 1;
        var d: usize = ndim;
        while (d > 0) {
            d -= 1;
            strides[d] = stride;
            stride *= @intCast(input.shape[d]);
        }

        // Second pass: collect indices
        var idx: usize = 0;
        if (input.asConstSlice(f32)) |data| {
            for (data, 0..) |v, flat_idx| {
                if (v != 0) {
                    // Convert flat index to multi-dimensional indices
                    var remaining = flat_idx;
                    for (0..ndim) |dim| {
                        out_data[dim * count + idx] = @intCast(remaining / strides[dim]);
                        remaining %= strides[dim];
                    }
                    idx += 1;
                }
            }
        } else if (input.asConstSlice(f16)) |data| {
            for (data, 0..) |v, flat_idx| {
                if (v != @as(f16, 0)) {
                    var remaining = flat_idx;
                    for (0..ndim) |dim| {
                        out_data[dim * count + idx] = @intCast(remaining / strides[dim]);
                        remaining %= strides[dim];
                    }
                    idx += 1;
                }
            }
        } else if (input.asConstSlice(i64)) |data| {
            for (data, 0..) |v, flat_idx| {
                if (v != 0) {
                    var remaining = flat_idx;
                    for (0..ndim) |dim| {
                        out_data[dim * count + idx] = @intCast(remaining / strides[dim]);
                        remaining %= strides[dim];
                    }
                    idx += 1;
                }
            }
        } else if (input.asConstSlice(i32)) |data| {
            for (data, 0..) |v, flat_idx| {
                if (v != 0) {
                    var remaining = flat_idx;
                    for (0..ndim) |dim| {
                        out_data[dim * count + idx] = @intCast(remaining / strides[dim]);
                        remaining %= strides[dim];
                    }
                    idx += 1;
                }
            }
        }

        if (self.buffers[output_idx]) |*existing| existing.deinit();
        self.buffers[output_idx] = output;
    }

    /// Execute MaxPool (2D max pooling)
    fn execMaxPool(self: *Executor, node: Node) !void {
        // MaxPool inputs: X [batch, channels, height, width]
        // MaxPool outputs: Y, optional Indices
        if (node.inputs.len < 1 or node.outputs.len < 1) return error.InvalidNode;

        const input = self.buffers[node.inputs[0]] orelse return error.MissingInput;
        const output_idx = node.outputs[0];

        if (input.shape.len != 4) return error.InvalidShape;

        const batch: usize = @intCast(input.shape[0]);
        const channels: usize = @intCast(input.shape[1]);
        const in_height: usize = @intCast(input.shape[2]);
        const in_width: usize = @intCast(input.shape[3]);

        // Get attributes
        const attrs = node.attributes.pool;
        const default_kernel = [_]i64{ 2, 2 };
        const default_stride = [_]i64{ 1, 1 };
        const default_pads = [_]i64{ 0, 0, 0, 0 };

        const kernel_shape = attrs.kernel_shape orelse &default_kernel;
        const strides_attr = attrs.strides orelse &default_stride;
        const pads = attrs.pads orelse &default_pads;

        const kernel_h: usize = @intCast(kernel_shape[0]);
        const kernel_w: usize = if (kernel_shape.len > 1) @intCast(kernel_shape[1]) else kernel_h;
        const stride_h: usize = @intCast(strides_attr[0]);
        const stride_w: usize = if (strides_attr.len > 1) @intCast(strides_attr[1]) else stride_h;
        const pad_top: usize = @intCast(pads[0]);
        const pad_left: usize = if (pads.len > 1) @intCast(pads[1]) else pad_top;
        const pad_bottom: usize = if (pads.len > 2) @intCast(pads[2]) else pad_top;
        const pad_right: usize = if (pads.len > 3) @intCast(pads[3]) else pad_left;

        const padded_height = in_height + pad_top + pad_bottom;
        const padded_width = in_width + pad_left + pad_right;
        const out_height = if (padded_height >= kernel_h) (padded_height - kernel_h) / stride_h + 1 else 0;
        const out_width = if (padded_width >= kernel_w) (padded_width - kernel_w) / stride_w + 1 else 0;

        const out_shape = [_]i64{
            @intCast(batch),
            @intCast(channels),
            @intCast(out_height),
            @intCast(out_width),
        };

        var output = try RuntimeTensor.alloc(self.allocator, input.dtype, &out_shape);
        errdefer output.deinit();

        if (input.asConstSlice(f32)) |in_data| {
            const out_data = output.asSlice(f32).?;

            for (0..batch) |b| {
                for (0..channels) |c| {
                    for (0..out_height) |oh| {
                        for (0..out_width) |ow| {
                            var max_val: f32 = -std.math.inf(f32);
                            const in_start_h = oh * stride_h;
                            const in_start_w = ow * stride_w;

                            for (0..kernel_h) |kh| {
                                for (0..kernel_w) |kw| {
                                    const h_signed: i64 = @as(i64, @intCast(in_start_h + kh)) - @as(i64, @intCast(pad_top));
                                    const w_signed: i64 = @as(i64, @intCast(in_start_w + kw)) - @as(i64, @intCast(pad_left));

                                    if (h_signed >= 0 and h_signed < @as(i64, @intCast(in_height)) and
                                        w_signed >= 0 and w_signed < @as(i64, @intCast(in_width)))
                                    {
                                        const h: usize = @intCast(h_signed);
                                        const w: usize = @intCast(w_signed);
                                        const idx = ((b * channels + c) * in_height + h) * in_width + w;
                                        max_val = @max(max_val, in_data[idx]);
                                    }
                                }
                            }

                            const out_idx = ((b * channels + c) * out_height + oh) * out_width + ow;
                            out_data[out_idx] = max_val;
                        }
                    }
                }
            }
        } else {
            return error.UnsupportedDType;
        }

        if (self.buffers[output_idx]) |*existing| existing.deinit();
        self.buffers[output_idx] = output;
    }

    /// Execute AveragePool (2D average pooling)
    fn execAveragePool(self: *Executor, node: Node) !void {
        if (node.inputs.len < 1 or node.outputs.len < 1) return error.InvalidNode;

        const input = self.buffers[node.inputs[0]] orelse return error.MissingInput;
        const output_idx = node.outputs[0];

        // Support both 3D (1D pooling) and 4D (2D pooling)
        if (input.shape.len == 3) {
            return self.execAveragePool1D(node, input, output_idx);
        }
        if (input.shape.len != 4) return error.InvalidShape;

        const batch: usize = @intCast(input.shape[0]);
        const channels: usize = @intCast(input.shape[1]);
        const in_height: usize = @intCast(input.shape[2]);
        const in_width: usize = @intCast(input.shape[3]);

        const attrs = node.attributes.pool;
        const default_kernel = [_]i64{ 2, 2 };
        const default_stride = [_]i64{ 1, 1 };
        const default_pads = [_]i64{ 0, 0, 0, 0 };

        const kernel_shape = attrs.kernel_shape orelse &default_kernel;
        const strides_attr = attrs.strides orelse &default_stride;
        const pads = attrs.pads orelse &default_pads;

        const kernel_h: usize = @intCast(kernel_shape[0]);
        const kernel_w: usize = if (kernel_shape.len > 1) @intCast(kernel_shape[1]) else kernel_h;
        const stride_h: usize = @intCast(strides_attr[0]);
        const stride_w: usize = if (strides_attr.len > 1) @intCast(strides_attr[1]) else stride_h;
        const pad_top: usize = @intCast(pads[0]);
        const pad_left: usize = if (pads.len > 1) @intCast(pads[1]) else pad_top;
        const pad_bottom: usize = if (pads.len > 2) @intCast(pads[2]) else pad_top;
        const pad_right: usize = if (pads.len > 3) @intCast(pads[3]) else pad_left;
        const count_include_pad = attrs.count_include_pad;

        const padded_height = in_height + pad_top + pad_bottom;
        const padded_width = in_width + pad_left + pad_right;
        const out_height = if (padded_height >= kernel_h) (padded_height - kernel_h) / stride_h + 1 else 0;
        const out_width = if (padded_width >= kernel_w) (padded_width - kernel_w) / stride_w + 1 else 0;

        const out_shape = [_]i64{
            @intCast(batch),
            @intCast(channels),
            @intCast(out_height),
            @intCast(out_width),
        };

        var output = try RuntimeTensor.alloc(self.allocator, input.dtype, &out_shape);
        errdefer output.deinit();

        if (input.asConstSlice(f32)) |in_data| {
            const out_data = output.asSlice(f32).?;

            for (0..batch) |b| {
                for (0..channels) |c| {
                    for (0..out_height) |oh| {
                        for (0..out_width) |ow| {
                            var sum: f32 = 0;
                            var count: usize = 0;
                            const in_start_h = oh * stride_h;
                            const in_start_w = ow * stride_w;

                            for (0..kernel_h) |kh| {
                                for (0..kernel_w) |kw| {
                                    const h_signed: i64 = @as(i64, @intCast(in_start_h + kh)) - @as(i64, @intCast(pad_top));
                                    const w_signed: i64 = @as(i64, @intCast(in_start_w + kw)) - @as(i64, @intCast(pad_left));

                                    if (h_signed >= 0 and h_signed < @as(i64, @intCast(in_height)) and
                                        w_signed >= 0 and w_signed < @as(i64, @intCast(in_width)))
                                    {
                                        const h: usize = @intCast(h_signed);
                                        const w: usize = @intCast(w_signed);
                                        const idx = ((b * channels + c) * in_height + h) * in_width + w;
                                        sum += in_data[idx];
                                        count += 1;
                                    } else if (count_include_pad) {
                                        count += 1;
                                    }
                                }
                            }

                            const out_idx = ((b * channels + c) * out_height + oh) * out_width + ow;
                            out_data[out_idx] = if (count > 0) sum / @as(f32, @floatFromInt(count)) else 0;
                        }
                    }
                }
            }
        } else {
            return error.UnsupportedDType;
        }

        if (self.buffers[output_idx]) |*existing| existing.deinit();
        self.buffers[output_idx] = output;
    }

    /// Execute 1D Average Pool (for 3D input: [batch, channels, length])
    fn execAveragePool1D(self: *Executor, node: Node, input: RuntimeTensor, output_idx: u32) !void {
        const batch: usize = @intCast(input.shape[0]);
        const channels: usize = @intCast(input.shape[1]);
        const in_length: usize = @intCast(input.shape[2]);

        const attrs = node.attributes.pool;
        const default_kernel = [_]i64{2};
        const default_stride = [_]i64{1};
        const default_pads = [_]i64{ 0, 0 };

        const kernel_shape = attrs.kernel_shape orelse &default_kernel;
        const strides_attr = attrs.strides orelse &default_stride;
        const pads = attrs.pads orelse &default_pads;

        const kernel_size: usize = @intCast(kernel_shape[0]);
        const stride: usize = @intCast(strides_attr[0]);
        const pad_start: usize = @intCast(pads[0]);
        const pad_end: usize = if (pads.len > 1) @intCast(pads[1]) else pad_start;
        const count_include_pad = attrs.count_include_pad;

        const padded_length = in_length + pad_start + pad_end;
        const out_length = if (padded_length >= kernel_size) (padded_length - kernel_size) / stride + 1 else 0;

        const out_shape = [_]i64{
            @intCast(batch),
            @intCast(channels),
            @intCast(out_length),
        };

        var output = try RuntimeTensor.alloc(self.allocator, input.dtype, &out_shape);
        errdefer output.deinit();

        if (input.asConstSlice(f32)) |in_data| {
            const out_data = output.asSlice(f32).?;

            for (0..batch) |b| {
                for (0..channels) |c| {
                    for (0..out_length) |ol| {
                        var sum: f32 = 0;
                        var count: usize = 0;
                        const in_start = ol * stride;

                        for (0..kernel_size) |k| {
                            const pos_signed: i64 = @as(i64, @intCast(in_start + k)) - @as(i64, @intCast(pad_start));

                            if (pos_signed >= 0 and pos_signed < @as(i64, @intCast(in_length))) {
                                const pos: usize = @intCast(pos_signed);
                                const idx = (b * channels + c) * in_length + pos;
                                sum += in_data[idx];
                                count += 1;
                            } else if (count_include_pad) {
                                count += 1;
                            }
                        }

                        const out_idx = (b * channels + c) * out_length + ol;
                        out_data[out_idx] = if (count > 0) sum / @as(f32, @floatFromInt(count)) else 0;
                    }
                }
            }
        }

        if (self.buffers[output_idx]) |*existing| existing.deinit();
        self.buffers[output_idx] = output;
    }

    /// Execute GlobalAveragePool
    fn execGlobalAveragePool(self: *Executor, node: Node) !void {
        if (node.inputs.len < 1 or node.outputs.len < 1) return error.InvalidNode;

        const input = self.buffers[node.inputs[0]] orelse return error.MissingInput;
        const output_idx = node.outputs[0];

        if (input.shape.len < 3) return error.InvalidShape;

        // Output shape: same as input but spatial dims become 1
        var out_shape_buf: [8]i64 = undefined;
        for (input.shape, 0..) |dim, i| {
            out_shape_buf[i] = if (i >= 2) 1 else dim;
        }
        const out_shape = out_shape_buf[0..input.shape.len];

        var output = try RuntimeTensor.alloc(self.allocator, input.dtype, out_shape);
        errdefer output.deinit();

        if (input.asConstSlice(f32)) |in_data| {
            const out_data = output.asSlice(f32).?;

            // Calculate spatial size
            var spatial_size: usize = 1;
            for (input.shape[2..]) |dim| {
                spatial_size *= @intCast(dim);
            }

            const batch: usize = @intCast(input.shape[0]);
            const channels: usize = @intCast(input.shape[1]);

            for (0..batch) |b| {
                for (0..channels) |c| {
                    var sum: f32 = 0;
                    const base_idx = (b * channels + c) * spatial_size;
                    for (0..spatial_size) |s| {
                        sum += in_data[base_idx + s];
                    }
                    out_data[b * channels + c] = sum / @as(f32, @floatFromInt(spatial_size));
                }
            }
        } else {
            return error.UnsupportedDType;
        }

        if (self.buffers[output_idx]) |*existing| existing.deinit();
        self.buffers[output_idx] = output;
    }

    /// Execute BatchNormalization
    fn execBatchNorm(self: *Executor, node: Node) !void {
        // BatchNormalization inputs:
        // 0: X [batch, channels, ...]
        // 1: scale [channels]
        // 2: B (bias) [channels]
        // 3: input_mean [channels]
        // 4: input_var [channels]
        if (node.inputs.len < 5 or node.outputs.len < 1) return error.InvalidNode;

        const input = self.buffers[node.inputs[0]] orelse return error.MissingInput;
        const scale = self.buffers[node.inputs[1]] orelse return error.MissingInput;
        const bias = self.buffers[node.inputs[2]] orelse return error.MissingInput;
        const mean = self.buffers[node.inputs[3]] orelse return error.MissingInput;
        const variance = self.buffers[node.inputs[4]] orelse return error.MissingInput;
        const output_idx = node.outputs[0];

        if (input.shape.len < 2) return error.InvalidShape;

        const attrs = node.attributes.batch_norm;
        const epsilon: f32 = attrs.epsilon;

        var output = try RuntimeTensor.alloc(self.allocator, input.dtype, input.shape);
        errdefer output.deinit();

        if (input.asConstSlice(f32)) |in_data| {
            const out_data = output.asSlice(f32).?;
            const scale_data = scale.asConstSlice(f32) orelse return error.UnsupportedDType;
            const bias_data = bias.asConstSlice(f32) orelse return error.UnsupportedDType;
            const mean_data = mean.asConstSlice(f32) orelse return error.UnsupportedDType;
            const var_data = variance.asConstSlice(f32) orelse return error.UnsupportedDType;

            const batch: usize = @intCast(input.shape[0]);
            const channels: usize = @intCast(input.shape[1]);

            // Calculate spatial size (all dimensions after channels)
            var spatial_size: usize = 1;
            for (input.shape[2..]) |dim| {
                spatial_size *= @intCast(dim);
            }

            for (0..batch) |b| {
                for (0..channels) |c| {
                    const gamma = scale_data[c];
                    const beta = bias_data[c];
                    const mu = mean_data[c];
                    const sigma = @sqrt(var_data[c] + epsilon);

                    const base_idx = (b * channels + c) * spatial_size;
                    for (0..spatial_size) |s| {
                        const idx = base_idx + s;
                        out_data[idx] = gamma * (in_data[idx] - mu) / sigma + beta;
                    }
                }
            }
        } else {
            return error.UnsupportedDType;
        }

        if (self.buffers[output_idx]) |*existing| existing.deinit();
        self.buffers[output_idx] = output;
    }

    /// Execute Flatten
    fn execFlatten(self: *Executor, node: Node) !void {
        if (node.inputs.len < 1 or node.outputs.len < 1) return error.InvalidNode;

        const input = self.buffers[node.inputs[0]] orelse return error.MissingInput;
        const output_idx = node.outputs[0];

        const attrs = node.attributes.flatten;
        var axis: i64 = attrs.axis;

        // Handle negative axis
        const ndim = @as(i64, @intCast(input.shape.len));
        if (axis < 0) axis += ndim;
        if (axis < 0 or axis > ndim) return error.InvalidAxis;

        const axis_usize: usize = @intCast(axis);

        // Calculate output shape [dim0 * ... * dim(axis-1), dim(axis) * ... * dim(n-1)]
        var dim0: i64 = 1;
        var dim1: i64 = 1;

        for (input.shape[0..axis_usize]) |d| {
            dim0 *= d;
        }
        for (input.shape[axis_usize..]) |d| {
            dim1 *= d;
        }

        const out_shape = [_]i64{ dim0, dim1 };

        // Allocate output and copy data (reshape is just a view change)
        var output = try RuntimeTensor.alloc(self.allocator, input.dtype, &out_shape);
        errdefer output.deinit();

        @memcpy(output.data, input.data[0..@min(output.data.len, input.data.len)]);

        if (self.buffers[output_idx]) |*existing| existing.deinit();
        self.buffers[output_idx] = output;
    }

    /// Execute Split
    fn execSplit(self: *Executor, node: Node) !void {
        if (node.inputs.len < 1 or node.outputs.len < 1) return error.InvalidNode;

        const input = self.buffers[node.inputs[0]] orelse return error.MissingInput;

        const attrs = node.attributes.split;
        var axis: i64 = attrs.axis;

        // Handle negative axis
        const ndim = @as(i64, @intCast(input.shape.len));
        if (axis < 0) axis += ndim;
        if (axis < 0 or axis >= ndim) return error.InvalidAxis;

        const axis_usize: usize = @intCast(axis);
        const axis_dim: usize = @intCast(input.shape[axis_usize]);
        const num_outputs = node.outputs.len;

        // Get split sizes from attribute or input
        var split_sizes_buf: [16]usize = undefined;
        var split_sizes: []usize = undefined;

        if (attrs.split) |split_attr| {
            // Split sizes from attribute
            if (split_attr.len > 16) return error.TooManySplits;
            for (split_attr, 0..) |s, i| {
                split_sizes_buf[i] = @intCast(s);
            }
            split_sizes = split_sizes_buf[0..split_attr.len];
        } else if (node.inputs.len > 1) {
            // Split sizes from input tensor
            if (self.buffers[node.inputs[1]]) |split_tensor| {
                if (split_tensor.asConstSlice(i64)) |split_data| {
                    if (split_data.len > 16) return error.TooManySplits;
                    for (split_data, 0..) |s, i| {
                        split_sizes_buf[i] = @intCast(s);
                    }
                    split_sizes = split_sizes_buf[0..split_data.len];
                } else {
                    return error.UnsupportedDType;
                }
            } else {
                // Equal split
                const chunk_size = axis_dim / num_outputs;
                for (0..num_outputs) |i| {
                    split_sizes_buf[i] = chunk_size;
                }
                split_sizes = split_sizes_buf[0..num_outputs];
            }
        } else {
            // Equal split
            const chunk_size = axis_dim / num_outputs;
            for (0..num_outputs) |i| {
                split_sizes_buf[i] = chunk_size;
            }
            split_sizes = split_sizes_buf[0..num_outputs];
        }

        // Calculate strides
        var in_strides: [8]usize = undefined;
        var stride: usize = 1;
        var d: usize = input.shape.len;
        while (d > 0) {
            d -= 1;
            in_strides[d] = stride;
            stride *= @intCast(input.shape[d]);
        }

        var offset: usize = 0;
        for (node.outputs, 0..) |output_idx, out_i| {
            if (out_i >= split_sizes.len) break;

            const split_size = split_sizes[out_i];

            // Create output shape
            var out_shape_buf: [8]i64 = undefined;
            for (input.shape, 0..) |dim, i| {
                out_shape_buf[i] = if (i == axis_usize) @intCast(split_size) else dim;
            }
            const out_shape = out_shape_buf[0..input.shape.len];

            var output = try RuntimeTensor.alloc(self.allocator, input.dtype, out_shape);
            errdefer output.deinit();

            // Copy data for this split
            const in_data = input.data;
            const out_data = output.data;

            // Elements before axis, at axis, after axis
            var outer_size: usize = 1;
            for (input.shape[0..axis_usize]) |dim| {
                outer_size *= @intCast(dim);
            }
            var inner_size: usize = 1;
            for (input.shape[axis_usize + 1 ..]) |dim| {
                inner_size *= @intCast(dim);
            }

            const elem_size: usize = switch (input.dtype) {
                .f32, .i32 => 4,
                .f64, .i64 => 8,
                .f16, .bf16 => 2,
                .u8, .i8 => 1,
                else => 4,
            };

            for (0..outer_size) |o| {
                const in_base = (o * axis_dim + offset) * inner_size * elem_size;
                const out_base = o * split_size * inner_size * elem_size;
                const copy_size = split_size * inner_size * elem_size;

                if (in_base + copy_size <= in_data.len and out_base + copy_size <= out_data.len) {
                    @memcpy(out_data[out_base..][0..copy_size], in_data[in_base..][0..copy_size]);
                }
            }

            if (self.buffers[output_idx]) |*existing| existing.deinit();
            self.buffers[output_idx] = output;

            offset += split_size;
        }
    }

    /// Execute MultiHeadAttention (Microsoft ONNX extension)
    fn execMultiHeadAttention(self: *Executor, node: Node) !void {
        // MultiHeadAttention inputs:
        // 0: query [batch, seq_q, hidden]
        // 1: key [batch, seq_k, hidden]
        // 2: value [batch, seq_k, hidden]
        // Optional: bias, key_padding_mask, attention_bias
        if (node.inputs.len < 3 or node.outputs.len < 1) return error.InvalidNode;

        const query = self.buffers[node.inputs[0]] orelse return error.MissingInput;
        const key = self.buffers[node.inputs[1]] orelse return error.MissingInput;
        const value = self.buffers[node.inputs[2]] orelse return error.MissingInput;
        const output_idx = node.outputs[0];

        if (query.dtype != .f32) return error.UnsupportedDType;
        if (query.shape.len != 3) return error.InvalidShape;

        const batch_size: usize = @intCast(query.shape[0]);
        const seq_len_q: usize = @intCast(query.shape[1]);
        const hidden_dim: usize = @intCast(query.shape[2]);
        const seq_len_k: usize = @intCast(key.shape[1]);

        // Assume num_heads from hidden_dim (typical: 8 or 16 heads)
        const num_heads: usize = 8;
        const head_dim = hidden_dim / num_heads;

        if (head_dim * num_heads != hidden_dim) return error.InvalidShape;

        var output = try RuntimeTensor.alloc(self.allocator, query.dtype, query.shape);
        errdefer output.deinit();

        const q_data = query.asConstSlice(f32).?;
        const k_data = key.asConstSlice(f32).?;
        const v_data = value.asConstSlice(f32).?;
        const out_data = output.asSlice(f32).?;

        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

        // Standard multi-head attention
        for (0..batch_size) |b| {
            for (0..num_heads) |h| {
                for (0..seq_len_q) |q_pos| {
                    // Compute attention scores
                    var scores_buf: [2048]f32 = undefined;
                    const scores = scores_buf[0..seq_len_k];

                    for (0..seq_len_k) |k_pos| {
                        var dot: f32 = 0;
                        for (0..head_dim) |d| {
                            const q_idx = b * seq_len_q * hidden_dim + q_pos * hidden_dim + h * head_dim + d;
                            const k_idx = b * seq_len_k * hidden_dim + k_pos * hidden_dim + h * head_dim + d;
                            dot += q_data[q_idx] * k_data[k_idx];
                        }
                        scores[k_pos] = dot * scale;
                    }

                    // Softmax
                    var max_score: f32 = scores[0];
                    for (scores[1..]) |s| {
                        if (s > max_score) max_score = s;
                    }
                    var sum_exp: f32 = 0;
                    for (scores) |*s| {
                        s.* = @exp(s.* - max_score);
                        sum_exp += s.*;
                    }
                    for (scores) |*s| {
                        s.* /= sum_exp;
                    }

                    // Weighted sum of values
                    for (0..head_dim) |d| {
                        var weighted_sum: f32 = 0;
                        for (0..seq_len_k) |v_pos| {
                            const v_idx = b * seq_len_k * hidden_dim + v_pos * hidden_dim + h * head_dim + d;
                            weighted_sum += scores[v_pos] * v_data[v_idx];
                        }
                        const out_idx = b * seq_len_q * hidden_dim + q_pos * hidden_dim + h * head_dim + d;
                        out_data[out_idx] = weighted_sum;
                    }
                }
            }
        }

        if (self.buffers[output_idx]) |*existing| existing.deinit();
        self.buffers[output_idx] = output;
    }

    /// Execute Not (logical negation)
    fn execNot(self: *Executor, node: Node) !void {
        if (node.inputs.len < 1 or node.outputs.len < 1) return error.InvalidNode;

        const input = self.buffers[node.inputs[0]] orelse return error.MissingInput;
        const output_idx = node.outputs[0];

        var output = try RuntimeTensor.alloc(self.allocator, input.dtype, input.shape);
        errdefer output.deinit();

        const in_data = input.asSlice(u8) orelse input.asConstSlice(u8).?;
        const out_data = output.asSlice(u8).?;

        for (in_data, out_data) |x, *o| {
            o.* = if (x == 0) 1 else 0;
        }

        if (self.buffers[output_idx]) |*existing| existing.deinit();
        self.buffers[output_idx] = output;
    }

    /// Execute MatMulNBits - Microsoft extension for N-bit quantized matmul
    /// Inputs: A (f32/f16), B (packed uint8), scales, zero_points (optional)
    /// Attributes: K, N, bits, block_size
    fn execMatMulNBits(self: *Executor, node: Node) !void {
        if (node.inputs.len < 3 or node.outputs.len < 1) return error.InvalidNode;

        const a = self.buffers[node.inputs[0]] orelse return error.MissingInput;
        const b_packed = self.buffers[node.inputs[1]] orelse return error.MissingInput;
        const scales = self.buffers[node.inputs[2]] orelse return error.MissingInput;
        const output_idx = node.outputs[0];

        // Get attributes (K, N, bits, block_size)
        var attr_k: usize = 0;
        var attr_n: usize = 0;
        var bits: usize = 4;
        var block_size: usize = 32;

        switch (node.attributes) {
            .matmul_nbits => |attrs| {
                attr_k = @intCast(attrs.K);
                attr_n = @intCast(attrs.N);
                bits = @intCast(attrs.bits);
                block_size = @intCast(attrs.block_size);
            },
            else => {
                // Try to infer from tensor shapes
                if (a.shape.len >= 1) attr_k = @intCast(a.shape[a.shape.len - 1]);
                if (scales.shape.len >= 1) attr_n = @intCast(scales.shape[scales.shape.len - 1]);
            },
        }

        if (attr_k == 0 or attr_n == 0) return error.InvalidNode;

        // Determine batch dimensions from A
        var batch_size: usize = 1;
        var m: usize = 1;
        if (a.shape.len == 2) {
            m = @intCast(a.shape[0]);
        } else if (a.shape.len == 3) {
            batch_size = @intCast(a.shape[0]);
            m = @intCast(a.shape[1]);
        }

        const k = attr_k;
        const n = attr_n;

        // Allocate output
        var out_shape: [3]i64 = undefined;
        var out_ndim: usize = 2;
        if (a.shape.len == 3) {
            out_shape[0] = @intCast(batch_size);
            out_shape[1] = @intCast(m);
            out_shape[2] = @intCast(n);
            out_ndim = 3;
        } else {
            out_shape[0] = @intCast(m);
            out_shape[1] = @intCast(n);
        }

        var output = try RuntimeTensor.alloc(self.allocator, a.dtype, out_shape[0..out_ndim]);
        errdefer output.deinit();

        // Dequantize weights and perform matmul
        // For 4-bit: each byte contains 2 values (low 4 bits, high 4 bits)

        // Allocate temporary buffer for dequantized weights
        const dequant_weights = try self.allocator.alloc(f32, k * n);
        defer self.allocator.free(dequant_weights);

        const b_data = b_packed.data;
        const scale_data = scales.asConstSlice(f32) orelse {
            // Try f16 scales
            if (scales.asConstSlice(f16)) |s16| {
                // Convert f16 scales to f32 for computation
                const scale_f32 = try self.allocator.alloc(f32, s16.len);
                defer self.allocator.free(scale_f32);
                for (s16, scale_f32) |s, *d| d.* = @floatCast(s);

                // Dequantize with f16->f32 scales
                dequantize4Bit(b_data, scale_f32, dequant_weights, k, n, block_size, bits);
            } else {
                return error.UnsupportedDType;
            }

            // Perform matmul: A @ dequant_weights
            const a_data = a.asConstSlice(f32) orelse return error.UnsupportedDType;
            const out_data = output.asSlice(f32).?;

            if (batch_size > 1) {
                kernels.matmul.batchedMatmulBroadcastB(f32, a_data, dequant_weights, out_data, batch_size, m, k, n);
            } else {
                kernels.matmul.matmulTiled(f32, a_data, dequant_weights, out_data, m, k, n);
            }

            if (self.buffers[output_idx]) |*existing| existing.deinit();
            self.buffers[output_idx] = output;
            return;
        };

        // Dequantize 4-bit weights
        dequantize4Bit(b_data, scale_data, dequant_weights, k, n, block_size, bits);

        // Perform matmul: A @ dequant_weights
        const a_data = a.asConstSlice(f32) orelse return error.UnsupportedDType;
        const out_data = output.asSlice(f32).?;

        if (batch_size > 1) {
            kernels.matmul.batchedMatmulBroadcastB(f32, a_data, dequant_weights, out_data, batch_size, m, k, n);
        } else {
            kernels.matmul.matmulTiled(f32, a_data, dequant_weights, out_data, m, k, n);
        }

        if (self.buffers[output_idx]) |*existing| existing.deinit();
        self.buffers[output_idx] = output;
    }

    /// Execute GatherBlockQuantized - quantized gather for embeddings
    fn execGatherBlockQuantized(self: *Executor, node: Node) !void {
        if (node.inputs.len < 3 or node.outputs.len < 1) return error.InvalidNode;

        const data = self.buffers[node.inputs[0]] orelse return error.MissingInput;
        const indices = self.buffers[node.inputs[1]] orelse return error.MissingInput;
        const scales = self.buffers[node.inputs[2]] orelse return error.MissingInput;
        const output_idx = node.outputs[0];

        // Get block_size from attributes
        var block_size: usize = 32;
        switch (node.attributes) {
            .gather_block_quantized => |attrs| {
                block_size = @intCast(attrs.block_size);
            },
            else => {},
        }

        const idx_data = indices.asConstSlice(i64) orelse {
            if (indices.asConstSlice(i32)) |i32_data| {
                // Handle i32 indices
                const num_indices = indices.numel;
                // For 4-bit quantized data, each byte contains 2 values
                const embed_dim: usize = @intCast(data.shape[1] * 2);

                var out_shape: [3]i64 = undefined;
                var out_ndim: usize = 2;
                if (indices.shape.len == 2) {
                    out_shape[0] = indices.shape[0];
                    out_shape[1] = indices.shape[1];
                    out_shape[2] = @intCast(embed_dim);
                    out_ndim = 3;
                } else {
                    out_shape[0] = @intCast(num_indices);
                    out_shape[1] = @intCast(embed_dim);
                }

                var output = try RuntimeTensor.alloc(self.allocator, .f32, out_shape[0..out_ndim]);
                errdefer output.deinit();

                const out_data = output.asSlice(f32).?;

                // Handle f32 or f16 scales
                const scale_f32 = if (scales.asConstSlice(f32)) |s32| blk: {
                    break :blk s32;
                } else if (scales.asConstSlice(f16)) |s16| blk: {
                    const temp = try self.allocator.alloc(f32, s16.len);
                    for (s16, temp) |s, *d| d.* = @floatCast(s);
                    break :blk temp;
                } else return error.UnsupportedDType;
                defer if (scales.dtype == .f16) self.allocator.free(scale_f32);

                // Dequantize gathered embeddings
                for (0..num_indices) |i| {
                    const idx: usize = @intCast(i32_data[i]);
                    dequantizeRow4Bit(data.data, scale_f32, out_data[i * embed_dim ..][0..embed_dim], idx, embed_dim, block_size);
                }

                if (self.buffers[output_idx]) |*existing| existing.deinit();
                self.buffers[output_idx] = output;
                return;
            }
            return error.UnsupportedDType;
        };

        const num_indices = indices.numel;
        // For 4-bit quantized data, each byte contains 2 values
        const embed_dim: usize = @intCast(data.shape[1] * 2);

        var out_shape: [3]i64 = undefined;
        var out_ndim: usize = 2;
        if (indices.shape.len == 2) {
            out_shape[0] = indices.shape[0];
            out_shape[1] = indices.shape[1];
            out_shape[2] = @intCast(embed_dim);
            out_ndim = 3;
        } else {
            out_shape[0] = @intCast(num_indices);
            out_shape[1] = @intCast(embed_dim);
        }

        var output = try RuntimeTensor.alloc(self.allocator, .f32, out_shape[0..out_ndim]);
        errdefer output.deinit();

        const out_data = output.asSlice(f32).?;

        // Handle f32 or f16 scales
        const scale_f32 = if (scales.asConstSlice(f32)) |s32| blk: {
            break :blk s32;
        } else if (scales.asConstSlice(f16)) |s16| blk: {
            const temp = try self.allocator.alloc(f32, s16.len);
            for (s16, temp) |s, *d| d.* = @floatCast(s);
            break :blk temp;
        } else return error.UnsupportedDType;
        defer if (scales.dtype == .f16) self.allocator.free(scale_f32);

        // Dequantize gathered embeddings
        for (0..num_indices) |i| {
            const idx: usize = @intCast(idx_data[i]);
            dequantizeRow4Bit(data.data, scale_f32, out_data[i * embed_dim ..][0..embed_dim], idx, embed_dim, block_size);
        }

        if (self.buffers[output_idx]) |*existing| existing.deinit();
        self.buffers[output_idx] = output;
    }

    /// Dequantize 4-bit packed weights to f32
    fn dequantize4Bit(packed_data: []const u8, scales: []const f32, output: []f32, k: usize, n: usize, block_size: usize, bits: usize) void {
        _ = bits; // Assume 4-bit for now

        for (0..n) |col| {
            for (0..k) |row| {
                const block_idx = row / block_size;
                const scale_idx = block_idx * n + col;
                const scale = if (scale_idx < scales.len) scales[scale_idx] else 1.0;

                // Each byte contains 2 4-bit values
                const packed_idx = (col * k + row) / 2;
                const is_high = (row % 2) == 1;

                if (packed_idx < packed_data.len) {
                    const byte = packed_data[packed_idx];
                    const val: i8 = if (is_high)
                        @as(i8, @intCast((byte >> 4) & 0xF)) - 8
                    else
                        @as(i8, @intCast(byte & 0xF)) - 8;

                    output[row * n + col] = @as(f32, @floatFromInt(val)) * scale;
                } else {
                    output[row * n + col] = 0;
                }
            }
        }
    }

    /// Dequantize a single row from 4-bit packed data
    fn dequantizeRow4Bit(packed_data: []const u8, scales: []const f32, output: []f32, row_idx: usize, embed_dim: usize, block_size: usize) void {
        const num_blocks = (embed_dim + block_size - 1) / block_size;
        const row_bytes = (embed_dim + 1) / 2; // Bytes per row (2 values per byte)
        const row_start = row_idx * row_bytes;

        for (0..embed_dim) |col| {
            const block_idx = col / block_size;
            const scale_idx = row_idx * num_blocks + block_idx;
            const scale = if (scale_idx < scales.len) scales[scale_idx] else 1.0;

            const byte_idx = row_start + col / 2;
            const is_high = (col % 2) == 1;

            if (byte_idx < packed_data.len) {
                const byte = packed_data[byte_idx];
                const val: i8 = if (is_high)
                    @as(i8, @intCast((byte >> 4) & 0xF)) - 8
                else
                    @as(i8, @intCast(byte & 0xF)) - 8;

                output[col] = @as(f32, @floatFromInt(val)) * scale;
            } else {
                output[col] = 0;
            }
        }
    }
};

// Helper to convert Zig type to DType
fn zigTypeToDType(comptime T: type) ?DType {
    return switch (T) {
        f32 => .f32,
        f64 => .f64,
        f16 => .f16,
        i8 => .i8,
        i16 => .i16,
        i32 => .i32,
        i64 => .i64,
        u8 => .u8,
        u16 => .u16,
        u32 => .u32,
        u64 => .u64,
        else => null,
    };
}

// Add fromZigType to DType
fn dtypeFromZigType(comptime T: type) ?DType {
    return zigTypeToDType(T);
}

// Tests
test "RuntimeTensor alloc and free" {
    const allocator = std.testing.allocator;

    var tensor = try RuntimeTensor.alloc(allocator, .f32, &.{ 2, 3 });
    defer tensor.deinit();

    try std.testing.expectEqual(@as(usize, 6), tensor.numel);
    try std.testing.expectEqual(@as(usize, 24), tensor.data.len); // 6 * 4 bytes
    try std.testing.expectEqual(DType.f32, tensor.dtype);
}

test "RuntimeTensor fromSlice" {
    const allocator = std.testing.allocator;

    const data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var tensor = try RuntimeTensor.fromSlice(allocator, f32, &data, &.{ 2, 3 });
    defer tensor.deinit();

    const slice = tensor.asConstSlice(f32).?;
    try std.testing.expectEqualSlices(f32, &data, slice);
}

test "Executor basic elementwise" {
    const allocator = std.testing.allocator;
    const builder = @import("builder.zig");
    const types = @import("types.zig");

    // Create a simple Add graph
    var model: types.ModelProto = .{};
    var g: types.GraphProto = .{};
    g.name = "test";

    var node: types.NodeProto = .{};
    node.op_type = "Add";
    node.input = &.{ "A", "B" };
    node.output = &.{"C"};
    g.node = &.{node};

    var input_a: types.ValueInfoProto = .{};
    input_a.name = "A";
    var input_b: types.ValueInfoProto = .{};
    input_b.name = "B";
    g.input = &.{ input_a, input_b };

    var output_c: types.ValueInfoProto = .{};
    output_c.name = "C";
    g.output = &.{output_c};

    model.graph = g;

    // Build runtime graph
    var graph = try builder.buildGraph(allocator, model);
    defer graph.deinit();

    // Create executor
    var executor = try Executor.init(allocator, &graph);
    defer executor.deinit();

    // Set inputs
    try executor.setInputFromSlice("A", f32, &.{ 1, 2, 3, 4 }, &.{4});
    try executor.setInputFromSlice("B", f32, &.{ 10, 20, 30, 40 }, &.{4});

    // Run
    try executor.run();

    // Check output
    const output = executor.getOutput("C").?;
    const result = output.asConstSlice(f32).?;
    try std.testing.expectEqualSlices(f32, &.{ 11, 22, 33, 44 }, result);
}

test "Executor MatMul" {
    const allocator = std.testing.allocator;
    const builder = @import("builder.zig");
    const types = @import("types.zig");

    // Create MatMul graph: [2,3] @ [3,2] -> [2,2]
    var model: types.ModelProto = .{};
    var g: types.GraphProto = .{};

    var node: types.NodeProto = .{};
    node.op_type = "MatMul";
    node.input = &.{ "A", "B" };
    node.output = &.{"C"};
    g.node = &.{node};

    var input_a: types.ValueInfoProto = .{};
    input_a.name = "A";
    var input_b: types.ValueInfoProto = .{};
    input_b.name = "B";
    g.input = &.{ input_a, input_b };

    var output_c: types.ValueInfoProto = .{};
    output_c.name = "C";
    g.output = &.{output_c};

    model.graph = g;

    var graph = try builder.buildGraph(allocator, model);
    defer graph.deinit();

    var executor = try Executor.init(allocator, &graph);
    defer executor.deinit();

    // A = [[1,2,3],[4,5,6]]
    // B = [[1,2],[3,4],[5,6]]
    // C = [[22,28],[49,64]]
    try executor.setInputFromSlice("A", f32, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    try executor.setInputFromSlice("B", f32, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 3, 2 });

    try executor.run();

    const output = executor.getOutput("C").?;
    const result = output.asConstSlice(f32).?;
    try std.testing.expectEqualSlices(f32, &.{ 22, 28, 49, 64 }, result);
}
