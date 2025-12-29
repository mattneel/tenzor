//! Graph Builder - Converts ONNX proto types to runtime graph
//!
//! Takes a parsed ModelProto and builds a Graph ready for execution.

const std = @import("std");
const Allocator = std.mem.Allocator;
const types = @import("types.zig");
const graph_mod = @import("graph.zig");

const Graph = graph_mod.Graph;
const Node = graph_mod.Node;
const TensorInfo = graph_mod.TensorInfo;
const WeightData = graph_mod.WeightData;
const DType = graph_mod.DType;
const OpType = graph_mod.OpType;

pub const BuildError = error{
    InvalidModel,
    UnsupportedDataType,
    MissingTensor,
    MissingAttribute,
    InvalidAttribute,
    OutOfMemory,
};

/// Build a runtime graph from a parsed ONNX model.
pub fn buildGraph(allocator: Allocator, model: types.ModelProto) BuildError!Graph {
    var g = Graph.init(allocator);
    errdefer g.deinit();

    const onnx_graph = model.graph orelse return BuildError.InvalidModel;

    // 1. Register all initializers (weights/constants)
    for (onnx_graph.initializer) |tensor| {
        const name = tensor.name orelse continue;
        const dtype = DType.fromOnnx(tensor.data_type) orelse continue;

        // Copy dims to graph allocator (tensor.dims points to arena memory)
        const shape = try allocator.dupe(i64, tensor.dims);

        _ = try g.addTensor(.{
            .name = name,
            .dtype = dtype,
            .shape = shape,
            .owns_shape = true, // We allocated this shape, so graph should free it
        });

        // Check for external data
        if (tensor.isExternal()) {
            // Copy external location string
            const ext_loc = if (tensor.getExternalLocation()) |loc|
                try allocator.dupe(u8, loc)
            else
                null;

            try g.addWeight(name, .{
                .dtype = dtype,
                .shape = shape,
                .data = &.{}, // No inline data
                .external_location = ext_loc,
                .external_offset = tensor.getExternalOffset(),
                .external_length = tensor.getExternalLength(),
            });
        } else {
            try g.addWeight(name, .{
                .dtype = dtype,
                .shape = shape,
                .data = extractTensorData(tensor),
            });
        }
    }

    // 2. Register graph inputs (skip those already registered as initializers)
    var input_list: std.ArrayListUnmanaged(u32) = .empty;
    for (onnx_graph.input) |input| {
        const name = input.name orelse continue;

        // Skip if already registered as initializer
        if (g.getTensorIndex(name) != null) continue;

        // Get shape and dtype from ValueInfoProto
        const dtype = getValueInfoDType(input) orelse .f32;
        const shape_result = getValueInfoShape(allocator, input);
        const shape = shape_result.shape;
        const owns_shape = shape_result.allocated;

        const idx = try g.addTensor(.{
            .name = name,
            .dtype = dtype,
            .shape = shape,
            .owns_shape = owns_shape,
        });
        try input_list.append(allocator, idx);
    }
    g.inputs = try input_list.toOwnedSlice(allocator);

    // 3. Convert nodes
    for (onnx_graph.node) |node| {
        const converted = try convertNode(allocator, &g, node);
        try g.addNode(converted);
    }

    // 4. Mark graph outputs
    var output_list: std.ArrayListUnmanaged(u32) = .empty;
    for (onnx_graph.output) |output| {
        const name = output.name orelse continue;
        if (g.getTensorIndex(name)) |idx| {
            try output_list.append(allocator, idx);
        }
    }
    g.outputs = try output_list.toOwnedSlice(allocator);

    return g;
}

fn extractTensorData(tensor: types.TensorProto) []const u8 {
    // Prefer raw_data if available
    if (tensor.raw_data) |data| {
        return data;
    }

    // Otherwise convert from typed arrays (these are slices of bytes in our parsed representation)
    if (tensor.float_data.len > 0) {
        const ptr: [*]const u8 = @ptrCast(tensor.float_data.ptr);
        return ptr[0 .. tensor.float_data.len * 4];
    }
    if (tensor.int64_data.len > 0) {
        const ptr: [*]const u8 = @ptrCast(tensor.int64_data.ptr);
        return ptr[0 .. tensor.int64_data.len * 8];
    }
    if (tensor.int32_data.len > 0) {
        const ptr: [*]const u8 = @ptrCast(tensor.int32_data.ptr);
        return ptr[0 .. tensor.int32_data.len * 4];
    }

    return &[_]u8{};
}

fn getValueInfoDType(info: types.ValueInfoProto) ?DType {
    const type_proto = info.type_info orelse return null;
    const tensor_type = type_proto.tensor_type orelse return null;
    return DType.fromOnnx(tensor_type.elem_type);
}

const ShapeResult = struct {
    shape: []const i64,
    allocated: bool,
};

fn getValueInfoShape(allocator: Allocator, info: types.ValueInfoProto) ShapeResult {
    const type_proto = info.type_info orelse return .{ .shape = &[_]i64{}, .allocated = false };
    const tensor_type = type_proto.tensor_type orelse return .{ .shape = &[_]i64{}, .allocated = false };
    const shape_proto = tensor_type.shape orelse return .{ .shape = &[_]i64{}, .allocated = false };

    var dims: std.ArrayListUnmanaged(i64) = .empty;
    for (shape_proto.dim) |dim| {
        // Use dim_value if set, otherwise -1 for dynamic
        const value = dim.dim_value orelse -1;
        dims.append(allocator, value) catch return .{ .shape = &[_]i64{}, .allocated = false };
    }
    const owned_slice = dims.toOwnedSlice(allocator) catch return .{ .shape = &[_]i64{}, .allocated = false };
    return .{ .shape = owned_slice, .allocated = true };
}

fn convertNode(allocator: Allocator, g: *Graph, node: types.NodeProto) BuildError!Node {
    const op_type_str = node.op_type orelse "Unknown";
    const op_type = OpType.fromString(op_type_str);

    // Resolve input tensor indices
    var input_list: std.ArrayListUnmanaged(u32) = .empty;
    for (node.input) |name| {
        if (name.len == 0) {
            // Empty string means optional input not provided
            continue;
        }
        if (g.getTensorIndex(name)) |idx| {
            try input_list.append(allocator, idx);
        } else {
            // Input not yet registered - create placeholder
            // This can happen when a node outputs to a tensor not in inputs/initializers
            const idx = try g.addTensor(.{
                .name = name,
                .dtype = .f32, // Will be determined during execution
                .shape = &[_]i64{},
            });
            try input_list.append(allocator, idx);
        }
    }

    // Resolve output tensor indices (create if not exist)
    var output_list: std.ArrayListUnmanaged(u32) = .empty;
    for (node.output) |name| {
        if (name.len == 0) continue;

        const idx = g.getTensorIndex(name) orelse blk: {
            // Create new tensor for output
            break :blk try g.addTensor(.{
                .name = name,
                .dtype = .f32, // Will be inferred
                .shape = &[_]i64{},
            });
        };
        try output_list.append(allocator, idx);
    }

    return .{
        .op_type = op_type,
        .op_type_str = op_type_str,
        .name = node.name,
        .inputs = try input_list.toOwnedSlice(allocator),
        .outputs = try output_list.toOwnedSlice(allocator),
        .attributes = try parseAttributes(allocator, op_type, node.attribute),
    };
}

fn parseAttributes(allocator: Allocator, op_type: OpType, attrs: []const types.AttributeProto) BuildError!Node.Attributes {
    return switch (op_type) {
        .Gemm => .{ .gemm = parseGemmAttrs(attrs) },
        .Reshape => .{ .reshape = parseReshapeAttrs(attrs) },
        .Transpose => .{ .transpose = try parseTransposeAttrs(allocator, attrs) },
        .Softmax => .{ .softmax = parseAxisAttr(attrs, -1) },  // ONNX default: last axis
        .Concat => .{ .concat = parseAxisAttr(attrs, 0) },   // ONNX: axis is required, use 0 as fallback
        .Gather => .{ .gather = parseAxisAttr(attrs, 0) },   // ONNX default: axis=0
        .Squeeze => .{ .squeeze = parseAxesAttr(attrs) },
        .Unsqueeze => .{ .unsqueeze = parseAxesAttr(attrs) },
        .ReduceSum, .ReduceMean, .ReduceMax, .ReduceMin, .ReduceL2 => .{ .reduce = parseReduceAttrs(attrs) },
        .Conv, .ConvTranspose => .{ .conv = try parseConvAttrs(allocator, attrs) },
        .Cast => .{ .cast = parseCastAttrs(attrs) orelse return BuildError.MissingAttribute },
        .Constant => .{ .constant = try parseConstantAttrs(allocator, attrs) },
        .MatMulNBits => .{ .matmul_nbits = parseMatMulNBitsAttrs(attrs) },
        .GatherBlockQuantized => .{ .gather_block_quantized = parseGatherBlockQuantizedAttrs(attrs) },
        .MaxPool, .AveragePool => .{ .pool = try parsePoolAttrs(allocator, attrs) },
        .BatchNormalization => .{ .batch_norm = parseBatchNormAttrs(attrs) },
        .Flatten => .{ .flatten = parseFlattenAttrs(attrs) },
        .Split => .{ .split = try parseSplitAttrs(allocator, attrs) },
        else => .none,
    };
}

fn parseMatMulNBitsAttrs(attrs: []const types.AttributeProto) Node.Attributes.MatMulNBitsAttrs {
    var result: Node.Attributes.MatMulNBitsAttrs = .{};
    for (attrs) |attr| {
        const name = attr.name orelse "";
        if (std.mem.eql(u8, name, "K")) {
            result.K = attr.i orelse 0;
        } else if (std.mem.eql(u8, name, "N")) {
            result.N = attr.i orelse 0;
        } else if (std.mem.eql(u8, name, "bits")) {
            result.bits = attr.i orelse 4;
        } else if (std.mem.eql(u8, name, "block_size")) {
            result.block_size = attr.i orelse 32;
        }
    }
    return result;
}

fn parseGatherBlockQuantizedAttrs(attrs: []const types.AttributeProto) Node.Attributes.GatherBlockQuantizedAttrs {
    var result: Node.Attributes.GatherBlockQuantizedAttrs = .{};
    for (attrs) |attr| {
        const name = attr.name orelse "";
        if (std.mem.eql(u8, name, "block_size")) {
            result.block_size = attr.i orelse 32;
        } else if (std.mem.eql(u8, name, "quantize_axis")) {
            result.quantize_axis = attr.i orelse 1;
        }
    }
    return result;
}

fn parseGemmAttrs(attrs: []const types.AttributeProto) Node.Attributes.GemmAttrs {
    var result: Node.Attributes.GemmAttrs = .{};
    for (attrs) |attr| {
        const name = attr.name orelse "";
        if (std.mem.eql(u8, name, "alpha")) {
            result.alpha = attr.f orelse 1.0;
        } else if (std.mem.eql(u8, name, "beta")) {
            result.beta = attr.f orelse 1.0;
        } else if (std.mem.eql(u8, name, "transA")) {
            result.transA = (attr.i orelse 0) != 0;
        } else if (std.mem.eql(u8, name, "transB")) {
            result.transB = (attr.i orelse 0) != 0;
        }
    }
    return result;
}

fn parseReshapeAttrs(attrs: []const types.AttributeProto) Node.Attributes.ReshapeAttrs {
    var result: Node.Attributes.ReshapeAttrs = .{};
    for (attrs) |attr| {
        if (std.mem.eql(u8, attr.name orelse "", "allowzero")) {
            result.allow_zero = (attr.i orelse 0) != 0;
        }
    }
    return result;
}

fn parseTransposeAttrs(allocator: Allocator, attrs: []const types.AttributeProto) BuildError!Node.Attributes.TransposeAttrs {
    var result: Node.Attributes.TransposeAttrs = .{};
    for (attrs) |attr| {
        if (std.mem.eql(u8, attr.name orelse "", "perm")) {
            // Only use ints if the attribute type is actually INTS and data exists
            if (attr.attr_type == .ints and attr.ints.len > 0) {
                // Copy the perm data to our own allocation to ensure it stays valid
                result.perm = try allocator.dupe(i64, attr.ints);
            }
        }
    }
    return result;
}

fn parseAxisAttr(attrs: []const types.AttributeProto, default_axis: i64) Node.Attributes.AxisAttr {
    var result: Node.Attributes.AxisAttr = .{ .axis = default_axis };
    for (attrs) |attr| {
        if (std.mem.eql(u8, attr.name orelse "", "axis")) {
            result.axis = attr.i orelse default_axis;
        }
    }
    return result;
}

fn parseAxesAttr(attrs: []const types.AttributeProto) Node.Attributes.AxesAttr {
    var result: Node.Attributes.AxesAttr = .{};
    for (attrs) |attr| {
        if (std.mem.eql(u8, attr.name orelse "", "axes")) {
            // Only use ints if the attribute type is actually INTS and data exists
            if (attr.attr_type == .ints and attr.ints.len > 0) {
                result.axes = attr.ints;
            }
        }
    }
    return result;
}

fn parseReduceAttrs(attrs: []const types.AttributeProto) Node.Attributes.ReduceAttrs {
    var result: Node.Attributes.ReduceAttrs = .{};
    for (attrs) |attr| {
        const name = attr.name orelse "";
        if (std.mem.eql(u8, name, "axes")) {
            if (attr.attr_type == .ints and attr.ints.len > 0) result.axes = attr.ints;
        } else if (std.mem.eql(u8, name, "keepdims")) {
            result.keepdims = (attr.i orelse 1) != 0;
        } else if (std.mem.eql(u8, name, "noop_with_empty_axes")) {
            result.noop_with_empty_axes = (attr.i orelse 0) != 0;
        }
    }
    return result;
}

fn parseConvAttrs(allocator: Allocator, attrs: []const types.AttributeProto) BuildError!Node.Attributes.ConvAttrs {
    var result: Node.Attributes.ConvAttrs = .{};
    for (attrs) |attr| {
        const name = attr.name orelse "";
        if (std.mem.eql(u8, name, "kernel_shape")) {
            if (attr.attr_type == .ints and attr.ints.len > 0) {
                result.kernel_shape = allocator.dupe(i64, attr.ints) catch return error.OutOfMemory;
            }
        } else if (std.mem.eql(u8, name, "strides")) {
            if (attr.attr_type == .ints and attr.ints.len > 0) {
                result.strides = allocator.dupe(i64, attr.ints) catch return error.OutOfMemory;
            }
        } else if (std.mem.eql(u8, name, "pads")) {
            if (attr.attr_type == .ints and attr.ints.len > 0) {
                result.pads = allocator.dupe(i64, attr.ints) catch return error.OutOfMemory;
            }
        } else if (std.mem.eql(u8, name, "dilations")) {
            if (attr.attr_type == .ints and attr.ints.len > 0) {
                result.dilations = allocator.dupe(i64, attr.ints) catch return error.OutOfMemory;
            }
        } else if (std.mem.eql(u8, name, "group")) {
            result.group = attr.i orelse 1;
        } else if (std.mem.eql(u8, name, "auto_pad")) {
            result.auto_pad = allocator.dupe(u8, attr.s orelse "NOTSET") catch return error.OutOfMemory;
        }
    }
    return result;
}

fn parsePoolAttrs(allocator: Allocator, attrs: []const types.AttributeProto) BuildError!Node.Attributes.PoolAttrs {
    var result: Node.Attributes.PoolAttrs = .{};
    for (attrs) |attr| {
        const name = attr.name orelse "";
        if (std.mem.eql(u8, name, "kernel_shape")) {
            if (attr.attr_type == .ints and attr.ints.len > 0) {
                result.kernel_shape = allocator.dupe(i64, attr.ints) catch return error.OutOfMemory;
            }
        } else if (std.mem.eql(u8, name, "strides")) {
            if (attr.attr_type == .ints and attr.ints.len > 0) {
                result.strides = allocator.dupe(i64, attr.ints) catch return error.OutOfMemory;
            }
        } else if (std.mem.eql(u8, name, "pads")) {
            if (attr.attr_type == .ints and attr.ints.len > 0) {
                result.pads = allocator.dupe(i64, attr.ints) catch return error.OutOfMemory;
            }
        } else if (std.mem.eql(u8, name, "auto_pad")) {
            result.auto_pad = attr.s orelse "NOTSET";
        } else if (std.mem.eql(u8, name, "ceil_mode")) {
            result.ceil_mode = (attr.i orelse 0) != 0;
        } else if (std.mem.eql(u8, name, "count_include_pad")) {
            result.count_include_pad = (attr.i orelse 0) != 0;
        }
    }
    return result;
}

fn parseBatchNormAttrs(attrs: []const types.AttributeProto) Node.Attributes.BatchNormAttrs {
    var result: Node.Attributes.BatchNormAttrs = .{};
    for (attrs) |attr| {
        const name = attr.name orelse "";
        if (std.mem.eql(u8, name, "epsilon")) {
            result.epsilon = attr.f orelse 1e-5;
        } else if (std.mem.eql(u8, name, "momentum")) {
            result.momentum = attr.f orelse 0.9;
        } else if (std.mem.eql(u8, name, "training_mode")) {
            result.training_mode = (attr.i orelse 0) != 0;
        }
    }
    return result;
}

fn parseFlattenAttrs(attrs: []const types.AttributeProto) Node.Attributes.FlattenAttrs {
    var result: Node.Attributes.FlattenAttrs = .{};
    for (attrs) |attr| {
        const name = attr.name orelse "";
        if (std.mem.eql(u8, name, "axis")) {
            result.axis = attr.i orelse 1;
        }
    }
    return result;
}

fn parseSplitAttrs(allocator: Allocator, attrs: []const types.AttributeProto) BuildError!Node.Attributes.SplitAttrs {
    var result: Node.Attributes.SplitAttrs = .{};
    for (attrs) |attr| {
        const name = attr.name orelse "";
        if (std.mem.eql(u8, name, "axis")) {
            result.axis = attr.i orelse 0;
        } else if (std.mem.eql(u8, name, "split")) {
            if (attr.attr_type == .ints and attr.ints.len > 0) {
                result.split = allocator.dupe(i64, attr.ints) catch return error.OutOfMemory;
            }
        } else if (std.mem.eql(u8, name, "num_outputs")) {
            result.num_outputs = attr.i orelse 0;
        }
    }
    return result;
}

fn parseCastAttrs(attrs: []const types.AttributeProto) ?Node.Attributes.CastAttrs {
    for (attrs) |attr| {
        if (std.mem.eql(u8, attr.name orelse "", "to")) {
            const i_val = attr.i orelse continue;
            const onnx_type: types.DataType = @enumFromInt(@as(i32, @intCast(i_val)));
            if (DType.fromOnnx(onnx_type)) |dtype| {
                return .{ .to = dtype };
            }
        }
    }
    return null;
}

fn parseConstantAttrs(allocator: Allocator, attrs: []const types.AttributeProto) BuildError!Node.Attributes.ConstantAttrs {
    for (attrs) |attr| {
        const name = attr.name orelse "";
        if (std.mem.eql(u8, name, "value")) {
            if (attr.t) |tensor| {
                const dtype = DType.fromOnnx(tensor.data_type) orelse continue;
                // Copy shape to owned memory (tensor.dims is from parsing arena)
                const shape = allocator.dupe(i64, tensor.dims) catch return error.OutOfMemory;
                return .{
                    .value = .{
                        .dtype = dtype,
                        .shape = shape,
                        .data = extractTensorData(tensor),
                    },
                };
            }
        }
        // Handle scalar value attributes (value_float, value_int, etc.)
        else if (std.mem.eql(u8, name, "value_float")) {
            if (attr.f) |f| {
                // Create inline scalar data
                const data = allocator.alloc(u8, 4) catch return error.OutOfMemory;
                const ptr: *f32 = @ptrCast(@alignCast(data.ptr));
                ptr.* = f;
                return .{
                    .value = .{
                        .dtype = .f32,
                        .shape = &.{},
                        .data = data,
                    },
                };
            }
        } else if (std.mem.eql(u8, name, "value_int")) {
            if (attr.i) |i| {
                const data = allocator.alloc(u8, 8) catch return error.OutOfMemory;
                const ptr: *i64 = @ptrCast(@alignCast(data.ptr));
                ptr.* = i;
                return .{
                    .value = .{
                        .dtype = .i64,
                        .shape = &.{},
                        .data = data,
                    },
                };
            }
        }
    }
    return .{ .value = null };
}

// Tests
test "build simple graph with Add" {
    const allocator = std.testing.allocator;

    // Create a minimal model proto with Add node
    var model: types.ModelProto = .{};
    var g: types.GraphProto = .{};
    g.name = "test_graph";

    // Create Add node
    var node: types.NodeProto = .{};
    node.op_type = "Add";
    node.name = "add_0";
    node.input = &.{ "A", "B" };
    node.output = &.{"C"};

    g.node = &.{node};

    // Create inputs
    var input_a: types.ValueInfoProto = .{};
    input_a.name = "A";
    var input_b: types.ValueInfoProto = .{};
    input_b.name = "B";
    g.input = &.{ input_a, input_b };

    // Create output
    var output_c: types.ValueInfoProto = .{};
    output_c.name = "C";
    g.output = &.{output_c};

    model.graph = g;

    // Build runtime graph
    var runtime_graph = try buildGraph(allocator, model);
    defer runtime_graph.deinit();

    // Verify structure
    try std.testing.expectEqual(@as(usize, 3), runtime_graph.tensors.items.len); // A, B, C
    try std.testing.expectEqual(@as(usize, 1), runtime_graph.nodes.items.len);
    try std.testing.expectEqual(@as(usize, 2), runtime_graph.inputs.len);
    try std.testing.expectEqual(@as(usize, 1), runtime_graph.outputs.len);

    // Verify node
    const add_node = runtime_graph.nodes.items[0];
    try std.testing.expectEqual(OpType.Add, add_node.op_type);
    try std.testing.expectEqual(@as(usize, 2), add_node.inputs.len);
    try std.testing.expectEqual(@as(usize, 1), add_node.outputs.len);
}

test "build graph with Gemm attributes" {
    const allocator = std.testing.allocator;

    var model: types.ModelProto = .{};
    var g: types.GraphProto = .{};
    g.name = "test_gemm";

    // Create Gemm node with attributes
    var node: types.NodeProto = .{};
    node.op_type = "Gemm";
    node.name = "gemm_0";
    node.input = &.{ "A", "B", "C" };
    node.output = &.{"Y"};

    // Set attributes
    var attr_alpha: types.AttributeProto = .{};
    attr_alpha.name = "alpha";
    attr_alpha.f = 0.5;

    var attr_transB: types.AttributeProto = .{};
    attr_transB.name = "transB";
    attr_transB.i = 1;

    node.attribute = &.{ attr_alpha, attr_transB };

    g.node = &.{node};

    var input_a: types.ValueInfoProto = .{};
    input_a.name = "A";
    var input_b: types.ValueInfoProto = .{};
    input_b.name = "B";
    var input_c: types.ValueInfoProto = .{};
    input_c.name = "C";
    g.input = &.{ input_a, input_b, input_c };

    var output_y: types.ValueInfoProto = .{};
    output_y.name = "Y";
    g.output = &.{output_y};

    model.graph = g;

    var runtime_graph = try buildGraph(allocator, model);
    defer runtime_graph.deinit();

    // Verify Gemm attributes
    const gemm_node = runtime_graph.nodes.items[0];
    try std.testing.expectEqual(OpType.Gemm, gemm_node.op_type);

    const attrs = gemm_node.attributes.gemm;
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), attrs.alpha, 0.001);
    try std.testing.expect(attrs.transB);
    try std.testing.expect(!attrs.transA);
}

test "build graph with initializer" {
    const allocator = std.testing.allocator;

    var model: types.ModelProto = .{};
    var g: types.GraphProto = .{};
    g.name = "test_init";

    // Create weight tensor
    var weight: types.TensorProto = .{};
    weight.name = "W";
    weight.dims = &.{ 3, 4 };
    weight.data_type = .float;
    weight.raw_data = &[_]u8{0} ** 48; // 3*4*4 bytes

    g.initializer = &.{weight};

    // Input W should also be listed in graph.input for valid ONNX
    var input_x: types.ValueInfoProto = .{};
    input_x.name = "X";
    var input_w: types.ValueInfoProto = .{};
    input_w.name = "W";
    g.input = &.{ input_x, input_w };

    // MatMul node
    var node: types.NodeProto = .{};
    node.op_type = "MatMul";
    node.input = &.{ "X", "W" };
    node.output = &.{"Y"};
    g.node = &.{node};

    var output_y: types.ValueInfoProto = .{};
    output_y.name = "Y";
    g.output = &.{output_y};

    model.graph = g;

    var runtime_graph = try buildGraph(allocator, model);
    defer runtime_graph.deinit();

    // W should be in weights, not in inputs
    try std.testing.expect(runtime_graph.isWeight("W"));
    try std.testing.expectEqual(@as(usize, 1), runtime_graph.inputs.len); // Only X

    // W should still be a registered tensor
    const w_idx = runtime_graph.getTensorIndex("W");
    try std.testing.expect(w_idx != null);
}
