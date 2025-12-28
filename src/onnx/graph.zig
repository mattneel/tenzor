//! Runtime Graph for ONNX Execution
//!
//! This is the runtime representation of an ONNX computation graph.
//! Unlike tenzor's comptime graphs, shapes are determined at runtime.

const std = @import("std");
const Allocator = std.mem.Allocator;
const types = @import("types.zig");

/// Data types for tensors (maps to ONNX DataType)
pub const DType = enum(u8) {
    f32 = 1,
    f16 = 10,
    bf16 = 16,
    f64 = 11,
    i8 = 3,
    i16 = 5,
    i32 = 6,
    i64 = 7,
    u8 = 2,
    u16 = 4,
    u32 = 12,
    u64 = 13,
    bool_ = 9,

    pub fn byteSize(self: DType) usize {
        return switch (self) {
            .f32, .i32, .u32 => 4,
            .f16, .bf16, .i16, .u16 => 2,
            .f64, .i64, .u64 => 8,
            .i8, .u8, .bool_ => 1,
        };
    }

    pub fn fromOnnx(onnx_type: types.DataType) ?DType {
        return switch (onnx_type) {
            .float => .f32,
            .float16 => .f16,
            .bfloat16 => .bf16,
            .double => .f64,
            .int8 => .i8,
            .int16 => .i16,
            .int32 => .i32,
            .int64 => .i64,
            .uint8 => .u8,
            .uint16 => .u16,
            .uint32 => .u32,
            .uint64 => .u64,
            .bool => .bool_,
            else => null,
        };
    }

    pub fn fromZigType(comptime T: type) ?DType {
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
            bool => .bool_,
            else => null,
        };
    }

    pub fn ZigType(comptime self: DType) type {
        return switch (self) {
            .f32 => f32,
            .f64 => f64,
            .f16 => f16,
            .bf16 => u16, // bf16 stored as u16
            .i8 => i8,
            .i16 => i16,
            .i32 => i32,
            .i64 => i64,
            .u8 => u8,
            .u16 => u16,
            .u32 => u32,
            .u64 => u64,
            .bool_ => u8, // bool stored as u8
        };
    }
};

/// Metadata about a tensor in the graph
pub const TensorInfo = struct {
    name: []const u8,
    dtype: DType,
    shape: []const i64, // -1 for dynamic dimensions
    owns_shape: bool = false, // true if shape was allocated and needs freeing
};

/// Supported ONNX operations
pub const OpType = enum {
    // Elementwise binary
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,

    // Elementwise unary
    Neg,
    Abs,
    Exp,
    Log,
    Sqrt,
    Pow,
    Sin,
    Cos,
    Ceil,
    Floor,
    Round,
    Relu,
    LeakyRelu,
    Elu,
    Sigmoid,
    Tanh,
    Gelu,
    Silu,
    Softplus,
    Erf,
    Not,

    // Matrix ops
    MatMul,
    Gemm,

    // Normalization
    Softmax,
    LayerNormalization,
    RMSNormalization,
    BatchNormalization,

    // Shape ops
    Reshape,
    Transpose,
    Concat,
    Split,
    Squeeze,
    Unsqueeze,
    Gather,
    Slice,
    Flatten,
    Expand,
    Tile,
    Range,
    NonZero,
    CumSum,
    ScatterND,
    Resize,

    // Reduction
    ReduceSum,
    ReduceMean,
    ReduceMax,
    ReduceMin,
    ReduceProd,
    ReduceL2,

    // Convolution
    Conv,
    ConvTranspose,
    MaxPool,
    AveragePool,
    GlobalAveragePool,

    // Quantization
    DequantizeLinear,
    QuantizeLinear,

    // Comparison
    Equal,
    NotEqual,
    Less,
    LessOrEqual,
    Greater,
    GreaterOrEqual,

    // Misc
    Cast,
    Constant,
    ConstantOfShape,
    Shape,
    Pad,
    Where,
    Clip,

    // Control flow
    If,
    Loop,

    // Random ops
    RandomUniformLike,
    RandomNormalLike,

    // Audio ops
    STFT,

    // Microsoft extensions
    GroupQueryAttention,
    MultiHeadAttention,
    MatMulNBits,
    GatherBlockQuantized,

    // Unknown op
    Unknown,

    pub fn fromString(s: []const u8) OpType {
        const map = std.StaticStringMap(OpType).initComptime(.{
            .{ "Add", .Add },
            .{ "Sub", .Sub },
            .{ "Mul", .Mul },
            .{ "Div", .Div },
            .{ "Max", .Max },
            .{ "Min", .Min },
            .{ "Neg", .Neg },
            .{ "Abs", .Abs },
            .{ "Exp", .Exp },
            .{ "Log", .Log },
            .{ "Sqrt", .Sqrt },
            .{ "Pow", .Pow },
            .{ "Sin", .Sin },
            .{ "Cos", .Cos },
            .{ "Ceil", .Ceil },
            .{ "Floor", .Floor },
            .{ "Round", .Round },
            .{ "Relu", .Relu },
            .{ "LeakyRelu", .LeakyRelu },
            .{ "Elu", .Elu },
            .{ "Sigmoid", .Sigmoid },
            .{ "Tanh", .Tanh },
            .{ "Gelu", .Gelu },
            .{ "Silu", .Silu },
            .{ "Softplus", .Softplus },
            .{ "Erf", .Erf },
            .{ "Not", .Not },
            .{ "MatMul", .MatMul },
            .{ "Gemm", .Gemm },
            .{ "Softmax", .Softmax },
            .{ "LayerNormalization", .LayerNormalization },
            .{ "RMSNormalization", .RMSNormalization },
            .{ "BatchNormalization", .BatchNormalization },
            .{ "Reshape", .Reshape },
            .{ "Transpose", .Transpose },
            .{ "Concat", .Concat },
            .{ "Split", .Split },
            .{ "Squeeze", .Squeeze },
            .{ "Unsqueeze", .Unsqueeze },
            .{ "Gather", .Gather },
            .{ "Slice", .Slice },
            .{ "Flatten", .Flatten },
            .{ "Expand", .Expand },
            .{ "Tile", .Tile },
            .{ "Range", .Range },
            .{ "NonZero", .NonZero },
            .{ "CumSum", .CumSum },
            .{ "ScatterND", .ScatterND },
            .{ "Resize", .Resize },
            .{ "ReduceSum", .ReduceSum },
            .{ "ReduceMean", .ReduceMean },
            .{ "ReduceMax", .ReduceMax },
            .{ "ReduceMin", .ReduceMin },
            .{ "ReduceProd", .ReduceProd },
            .{ "ReduceL2", .ReduceL2 },
            .{ "Conv", .Conv },
            .{ "ConvTranspose", .ConvTranspose },
            .{ "MaxPool", .MaxPool },
            .{ "AveragePool", .AveragePool },
            .{ "GlobalAveragePool", .GlobalAveragePool },
            .{ "DequantizeLinear", .DequantizeLinear },
            .{ "QuantizeLinear", .QuantizeLinear },
            .{ "Cast", .Cast },
            .{ "Constant", .Constant },
            .{ "ConstantOfShape", .ConstantOfShape },
            .{ "Shape", .Shape },
            .{ "Pad", .Pad },
            .{ "Where", .Where },
            .{ "Clip", .Clip },
            .{ "Equal", .Equal },
            .{ "NotEqual", .NotEqual },
            .{ "Less", .Less },
            .{ "LessOrEqual", .LessOrEqual },
            .{ "Greater", .Greater },
            .{ "GreaterOrEqual", .GreaterOrEqual },
            .{ "If", .If },
            .{ "Loop", .Loop },
            .{ "RandomUniformLike", .RandomUniformLike },
            .{ "RandomNormalLike", .RandomNormalLike },
            .{ "STFT", .STFT },
            .{ "GroupQueryAttention", .GroupQueryAttention },
            .{ "MultiHeadAttention", .MultiHeadAttention },
            .{ "MatMulNBits", .MatMulNBits },
            .{ "GatherBlockQuantized", .GatherBlockQuantized },
        });
        return map.get(s) orelse .Unknown;
    }
};

/// A computation node in the graph
pub const Node = struct {
    op_type: OpType,
    op_type_str: []const u8, // Original ONNX op type string (for Unknown ops)
    name: ?[]const u8,
    inputs: []const u32, // indices into graph.tensors
    outputs: []const u32, // indices into graph.tensors
    attributes: Attributes,

    pub const Attributes = union(enum) {
        none: void,
        gemm: GemmAttrs,
        reshape: ReshapeAttrs,
        transpose: TransposeAttrs,
        softmax: AxisAttr,
        concat: AxisAttr,
        gather: AxisAttr,
        squeeze: AxesAttr,
        unsqueeze: AxesAttr,
        conv: ConvAttrs,
        reduce: ReduceAttrs,
        slice: SliceAttrs,
        pad: PadAttrs,
        cast: CastAttrs,
        constant: ConstantAttrs,
        matmul_nbits: MatMulNBitsAttrs,
        gather_block_quantized: GatherBlockQuantizedAttrs,

        pub const GemmAttrs = struct {
            alpha: f32 = 1.0,
            beta: f32 = 1.0,
            transA: bool = false,
            transB: bool = false,
        };

        pub const ReshapeAttrs = struct {
            allow_zero: bool = false,
        };

        pub const TransposeAttrs = struct {
            perm: ?[]const i64 = null,
        };

        pub const AxisAttr = struct {
            axis: i64 = -1,
        };

        pub const AxesAttr = struct {
            axes: ?[]const i64 = null,
        };

        pub const ConvAttrs = struct {
            kernel_shape: ?[]const i64 = null,
            strides: ?[]const i64 = null,
            pads: ?[]const i64 = null,
            dilations: ?[]const i64 = null,
            group: i64 = 1,
            auto_pad: []const u8 = "NOTSET",
        };

        pub const ReduceAttrs = struct {
            axes: ?[]const i64 = null,
            keepdims: bool = true,
            noop_with_empty_axes: bool = false,
        };

        pub const SliceAttrs = struct {
            // Note: starts, ends, axes, steps are typically inputs in newer ONNX
            // but can be attributes in older versions
        };

        pub const PadAttrs = struct {
            mode: []const u8 = "constant",
            // pads and value are typically inputs
        };

        pub const CastAttrs = struct {
            to: DType,
        };

        pub const ConstantAttrs = struct {
            value: ?WeightData = null,
        };

        pub const MatMulNBitsAttrs = struct {
            K: i64 = 0, // size of last dim of A
            N: i64 = 0, // size of last dim of output
            bits: i64 = 4, // quantization bits
            block_size: i64 = 32, // block size for quantization
        };

        pub const GatherBlockQuantizedAttrs = struct {
            block_size: i64 = 32,
            quantize_axis: i64 = 1,
        };
    };
};

/// Weight data (constant tensors)
pub const WeightData = struct {
    dtype: DType,
    shape: []const i64,
    data: []const u8, // raw bytes, interpret by dtype

    // External data fields (if data.len == 0, check these)
    external_location: ?[]const u8 = null, // relative path to external file
    external_offset: usize = 0, // byte offset in file
    external_length: usize = 0, // byte length (0 = infer from shape)

    pub fn isExternal(self: WeightData) bool {
        return self.external_location != null;
    }

    pub fn asFloats(self: WeightData) ?[]const f32 {
        if (self.dtype != .f32) return null;
        const ptr: [*]const f32 = @ptrCast(@alignCast(self.data.ptr));
        return ptr[0 .. self.data.len / 4];
    }

    pub fn asI64s(self: WeightData) ?[]const i64 {
        if (self.dtype != .i64) return null;
        const ptr: [*]const i64 = @ptrCast(@alignCast(self.data.ptr));
        return ptr[0 .. self.data.len / 8];
    }
};

/// Runtime computation graph
pub const Graph = struct {
    allocator: Allocator,

    /// All tensors in the graph (by index)
    tensors: std.ArrayListUnmanaged(TensorInfo),

    /// Map from tensor name to index
    tensor_name_map: std.StringHashMapUnmanaged(u32),

    /// Nodes in topological order (ONNX guarantees this)
    nodes: std.ArrayListUnmanaged(Node),

    /// Constant/weight data by tensor name
    weights: std.StringHashMapUnmanaged(WeightData),

    /// Graph input tensor indices
    inputs: []const u32,

    /// Graph output tensor indices
    outputs: []const u32,

    pub fn init(allocator: Allocator) Graph {
        return .{
            .allocator = allocator,
            .tensors = .empty,
            .tensor_name_map = .empty,
            .nodes = .empty,
            .weights = .empty,
            .inputs = &.{},
            .outputs = &.{},
        };
    }

    pub fn deinit(self: *Graph) void {
        // Free node inputs/outputs slices
        for (self.nodes.items) |node| {
            if (node.inputs.len > 0) {
                self.allocator.free(node.inputs);
            }
            if (node.outputs.len > 0) {
                self.allocator.free(node.outputs);
            }
        }
        // Free owned tensor shapes
        for (self.tensors.items) |tensor| {
            if (tensor.owns_shape and tensor.shape.len > 0) {
                self.allocator.free(tensor.shape);
            }
        }
        self.tensors.deinit(self.allocator);
        self.tensor_name_map.deinit(self.allocator);
        self.nodes.deinit(self.allocator);
        self.weights.deinit(self.allocator);
        if (self.inputs.len > 0) {
            self.allocator.free(self.inputs);
        }
        if (self.outputs.len > 0) {
            self.allocator.free(self.outputs);
        }
        self.* = undefined;
    }

    /// Add a tensor and return its index
    pub fn addTensor(self: *Graph, info: TensorInfo) !u32 {
        const idx: u32 = @intCast(self.tensors.items.len);
        try self.tensors.append(self.allocator, info);
        try self.tensor_name_map.put(self.allocator, info.name, idx);
        return idx;
    }

    /// Get tensor index by name
    pub fn getTensorIndex(self: *const Graph, name: []const u8) ?u32 {
        return self.tensor_name_map.get(name);
    }

    /// Get tensor info by index
    pub fn getTensor(self: *const Graph, idx: u32) ?TensorInfo {
        if (idx >= self.tensors.items.len) return null;
        return self.tensors.items[idx];
    }

    /// Get tensor info by name
    pub fn getTensorByName(self: *const Graph, name: []const u8) ?TensorInfo {
        const idx = self.getTensorIndex(name) orelse return null;
        return self.getTensor(idx);
    }

    /// Add a node to the graph
    pub fn addNode(self: *Graph, node: Node) !void {
        try self.nodes.append(self.allocator, node);
    }

    /// Add weight data
    pub fn addWeight(self: *Graph, name: []const u8, data: WeightData) !void {
        try self.weights.put(self.allocator, name, data);
    }

    /// Get weight data by name
    pub fn getWeight(self: *const Graph, name: []const u8) ?WeightData {
        return self.weights.get(name);
    }

    /// Check if a tensor is a weight (constant)
    pub fn isWeight(self: *const Graph, name: []const u8) bool {
        return self.weights.contains(name);
    }
};

// Tests
test "DType byte sizes" {
    try std.testing.expectEqual(@as(usize, 4), DType.f32.byteSize());
    try std.testing.expectEqual(@as(usize, 2), DType.f16.byteSize());
    try std.testing.expectEqual(@as(usize, 8), DType.f64.byteSize());
    try std.testing.expectEqual(@as(usize, 1), DType.i8.byteSize());
}

test "OpType from string" {
    try std.testing.expectEqual(OpType.Add, OpType.fromString("Add"));
    try std.testing.expectEqual(OpType.MatMul, OpType.fromString("MatMul"));
    try std.testing.expectEqual(OpType.LayerNormalization, OpType.fromString("LayerNormalization"));
    try std.testing.expectEqual(OpType.Unknown, OpType.fromString("CustomOp"));
}

test "Graph basic operations" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);
    defer graph.deinit();

    // Add tensors
    const idx1 = try graph.addTensor(.{
        .name = "input",
        .dtype = .f32,
        .shape = &.{ 1, 3, 224, 224 },
    });
    const idx2 = try graph.addTensor(.{
        .name = "output",
        .dtype = .f32,
        .shape = &.{ 1, 1000 },
    });

    try std.testing.expectEqual(@as(u32, 0), idx1);
    try std.testing.expectEqual(@as(u32, 1), idx2);

    // Lookup by name
    try std.testing.expectEqual(@as(?u32, 0), graph.getTensorIndex("input"));
    try std.testing.expectEqual(@as(?u32, 1), graph.getTensorIndex("output"));
    try std.testing.expectEqual(@as(?u32, null), graph.getTensorIndex("nonexistent"));

    // Lookup tensor info
    const info = graph.getTensorByName("input").?;
    try std.testing.expectEqualStrings("input", info.name);
    try std.testing.expectEqual(DType.f32, info.dtype);
    try std.testing.expectEqualSlices(i64, &.{ 1, 3, 224, 224 }, info.shape);
}

test "Graph with node" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);
    defer graph.deinit();

    const a = try graph.addTensor(.{ .name = "A", .dtype = .f32, .shape = &.{ 2, 3 } });
    const b = try graph.addTensor(.{ .name = "B", .dtype = .f32, .shape = &.{ 3, 4 } });
    const c = try graph.addTensor(.{ .name = "C", .dtype = .f32, .shape = &.{ 2, 4 } });

    // Allocate inputs/outputs so deinit can free them
    const inputs = try allocator.alloc(u32, 2);
    inputs[0] = a;
    inputs[1] = b;
    const outputs = try allocator.alloc(u32, 1);
    outputs[0] = c;

    try graph.addNode(.{
        .op_type = .MatMul,
        .op_type_str = "MatMul",
        .name = "matmul_0",
        .inputs = inputs,
        .outputs = outputs,
        .attributes = .none,
    });

    try std.testing.expectEqual(@as(usize, 1), graph.nodes.items.len);
    try std.testing.expectEqual(OpType.MatMul, graph.nodes.items[0].op_type);
}
