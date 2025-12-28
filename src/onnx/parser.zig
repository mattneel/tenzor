//! ONNX protobuf message parser.
//!
//! Parses raw protobuf bytes into ONNX type structures.
//! Uses an arena allocator for all parsed data to enable bulk deallocation.

const std = @import("std");
const proto = @import("proto.zig");
const types = @import("types.zig");

const Decoder = proto.Decoder;
const WireType = proto.WireType;

const DataType = types.DataType;
const AttributeType = types.AttributeType;
const TensorProto = types.TensorProto;
const AttributeProto = types.AttributeProto;
const TypeProto = types.TypeProto;
const TensorShapeProto = types.TensorShapeProto;
const ValueInfoProto = types.ValueInfoProto;
const NodeProto = types.NodeProto;
const GraphProto = types.GraphProto;
const OperatorSetIdProto = types.OperatorSetIdProto;
const ModelProto = types.ModelProto;

/// Parser error types.
pub const ParseError = error{
    EndOfStream,
    VarintTooLong,
    VarintOverflow,
    LengthTooLarge,
    InvalidFieldNumber,
    GroupsNotSupported,
    InvalidPackedData,
    UnexpectedWireType,
    InvalidUtf8,
    OutOfMemory,
};

/// Parse an ONNX model from raw protobuf bytes.
pub fn parseModel(allocator: std.mem.Allocator, data: []const u8) ParseError!ModelProto {
    var decoder = Decoder.init(data);
    return parseModelProto(allocator, &decoder, data.len);
}

// ============================================================================
// ModelProto Parser
// ============================================================================

fn parseModelProto(allocator: std.mem.Allocator, decoder: *Decoder, limit: usize) ParseError!ModelProto {
    var model = ModelProto{};

    var opset_list: std.ArrayList(OperatorSetIdProto) = .empty;

    const end_pos = decoder.pos + limit;
    while (decoder.pos < end_pos) {
        const header = try decoder.readFieldHeader();

        switch (header.field_number) {
            ModelProto.field_numbers.ir_version => {
                model.ir_version = try decoder.readVarintI64();
            },
            ModelProto.field_numbers.producer_name => {
                model.producer_name = try decoder.readString();
            },
            ModelProto.field_numbers.producer_version => {
                model.producer_version = try decoder.readString();
            },
            ModelProto.field_numbers.domain => {
                model.domain = try decoder.readString();
            },
            ModelProto.field_numbers.model_version => {
                model.model_version = try decoder.readVarintI64();
            },
            ModelProto.field_numbers.doc_string => {
                model.doc_string = try decoder.readString();
            },
            ModelProto.field_numbers.graph => {
                const msg_data = try decoder.readLengthDelimited();
                var sub_decoder = Decoder.init(msg_data);
                model.graph = try parseGraphProto(allocator, &sub_decoder, msg_data.len);
            },
            ModelProto.field_numbers.opset_import => {
                const msg_data = try decoder.readLengthDelimited();
                var sub_decoder = Decoder.init(msg_data);
                const opset = try parseOperatorSetIdProto(&sub_decoder, msg_data.len);
                try opset_list.append(allocator, opset);
            },
            else => {
                try decoder.skipField(header.wire_type);
            },
        }
    }

    model.opset_import = try opset_list.toOwnedSlice(allocator);
    return model;
}

// ============================================================================
// OperatorSetIdProto Parser
// ============================================================================

fn parseOperatorSetIdProto(decoder: *Decoder, limit: usize) ParseError!OperatorSetIdProto {
    var opset = OperatorSetIdProto{};

    const end_pos = decoder.pos + limit;
    while (decoder.pos < end_pos) {
        const header = try decoder.readFieldHeader();

        switch (header.field_number) {
            OperatorSetIdProto.field_numbers.domain => {
                opset.domain = try decoder.readString();
            },
            OperatorSetIdProto.field_numbers.version => {
                opset.version = try decoder.readVarintI64();
            },
            else => {
                try decoder.skipField(header.wire_type);
            },
        }
    }

    return opset;
}

// ============================================================================
// GraphProto Parser
// ============================================================================

fn parseGraphProto(allocator: std.mem.Allocator, decoder: *Decoder, limit: usize) ParseError!GraphProto {
    var graph = GraphProto{};

    var node_list: std.ArrayList(NodeProto) = .empty;
    var initializer_list: std.ArrayList(TensorProto) = .empty;
    var input_list: std.ArrayList(ValueInfoProto) = .empty;
    var output_list: std.ArrayList(ValueInfoProto) = .empty;
    var value_info_list: std.ArrayList(ValueInfoProto) = .empty;

    const end_pos = decoder.pos + limit;
    while (decoder.pos < end_pos) {
        const header = try decoder.readFieldHeader();

        switch (header.field_number) {
            GraphProto.field_numbers.node => {
                const msg_data = try decoder.readLengthDelimited();
                var sub_decoder = Decoder.init(msg_data);
                const node = try parseNodeProto(allocator, &sub_decoder, msg_data.len);
                try node_list.append(allocator, node);
            },
            GraphProto.field_numbers.name => {
                graph.name = try decoder.readString();
            },
            GraphProto.field_numbers.initializer => {
                const msg_data = try decoder.readLengthDelimited();
                var sub_decoder = Decoder.init(msg_data);
                const tensor = try parseTensorProto(allocator, &sub_decoder, msg_data.len);
                try initializer_list.append(allocator, tensor);
            },
            GraphProto.field_numbers.doc_string => {
                graph.doc_string = try decoder.readString();
            },
            GraphProto.field_numbers.input => {
                const msg_data = try decoder.readLengthDelimited();
                var sub_decoder = Decoder.init(msg_data);
                const info = try parseValueInfoProto(allocator, &sub_decoder, msg_data.len);
                try input_list.append(allocator, info);
            },
            GraphProto.field_numbers.output => {
                const msg_data = try decoder.readLengthDelimited();
                var sub_decoder = Decoder.init(msg_data);
                const info = try parseValueInfoProto(allocator, &sub_decoder, msg_data.len);
                try output_list.append(allocator, info);
            },
            GraphProto.field_numbers.value_info => {
                const msg_data = try decoder.readLengthDelimited();
                var sub_decoder = Decoder.init(msg_data);
                const info = try parseValueInfoProto(allocator, &sub_decoder, msg_data.len);
                try value_info_list.append(allocator, info);
            },
            else => {
                try decoder.skipField(header.wire_type);
            },
        }
    }

    graph.node = try node_list.toOwnedSlice(allocator);
    graph.initializer = try initializer_list.toOwnedSlice(allocator);
    graph.input = try input_list.toOwnedSlice(allocator);
    graph.output = try output_list.toOwnedSlice(allocator);
    graph.value_info = try value_info_list.toOwnedSlice(allocator);

    return graph;
}

// ============================================================================
// NodeProto Parser
// ============================================================================

fn parseNodeProto(allocator: std.mem.Allocator, decoder: *Decoder, limit: usize) ParseError!NodeProto {
    var node = NodeProto{};

    var input_list: std.ArrayList([]const u8) = .empty;
    var output_list: std.ArrayList([]const u8) = .empty;
    var attr_list: std.ArrayList(AttributeProto) = .empty;

    const end_pos = decoder.pos + limit;
    while (decoder.pos < end_pos) {
        const header = try decoder.readFieldHeader();

        switch (header.field_number) {
            NodeProto.field_numbers.input => {
                const s = try decoder.readString();
                try input_list.append(allocator, s);
            },
            NodeProto.field_numbers.output => {
                const s = try decoder.readString();
                try output_list.append(allocator, s);
            },
            NodeProto.field_numbers.name => {
                node.name = try decoder.readString();
            },
            NodeProto.field_numbers.op_type => {
                node.op_type = try decoder.readString();
            },
            NodeProto.field_numbers.domain => {
                node.domain = try decoder.readString();
            },
            NodeProto.field_numbers.attribute => {
                const msg_data = try decoder.readLengthDelimited();
                var sub_decoder = Decoder.init(msg_data);
                const attr = try parseAttributeProto(allocator, &sub_decoder, msg_data.len);
                try attr_list.append(allocator, attr);
            },
            NodeProto.field_numbers.doc_string => {
                node.doc_string = try decoder.readString();
            },
            else => {
                try decoder.skipField(header.wire_type);
            },
        }
    }

    node.input = try input_list.toOwnedSlice(allocator);
    node.output = try output_list.toOwnedSlice(allocator);
    node.attribute = try attr_list.toOwnedSlice(allocator);

    return node;
}

// ============================================================================
// AttributeProto Parser
// ============================================================================

fn parseAttributeProto(allocator: std.mem.Allocator, decoder: *Decoder, limit: usize) ParseError!AttributeProto {
    var attr = AttributeProto{};

    var floats_list: std.ArrayList(f32) = .empty;
    var ints_list: std.ArrayList(i64) = .empty;
    var strings_list: std.ArrayList([]const u8) = .empty;
    var tensors_list: std.ArrayList(TensorProto) = .empty;

    const end_pos = decoder.pos + limit;
    while (decoder.pos < end_pos) {
        const header = try decoder.readFieldHeader();

        switch (header.field_number) {
            AttributeProto.field_numbers.name => {
                attr.name = try decoder.readString();
            },
            AttributeProto.field_numbers.f => {
                attr.f = try decoder.readFloat();
            },
            AttributeProto.field_numbers.i => {
                attr.i = try decoder.readVarintI64();
            },
            AttributeProto.field_numbers.s => {
                attr.s = try decoder.readLengthDelimited();
            },
            AttributeProto.field_numbers.t => {
                const msg_data = try decoder.readLengthDelimited();
                var sub_decoder = Decoder.init(msg_data);
                attr.t = try parseTensorProto(allocator, &sub_decoder, msg_data.len);
            },
            AttributeProto.field_numbers.floats => {
                // Could be packed or repeated
                if (header.wire_type == .length_delimited) {
                    const values = try decoder.readPackedFloats(allocator);
                    for (values) |v| try floats_list.append(allocator, v);
                } else {
                    try floats_list.append(allocator, try decoder.readFloat());
                }
            },
            AttributeProto.field_numbers.ints => {
                if (header.wire_type == .length_delimited) {
                    const values = try decoder.readPackedInt64s(allocator);
                    for (values) |v| try ints_list.append(allocator, v);
                } else {
                    try ints_list.append(allocator, try decoder.readVarintI64());
                }
            },
            AttributeProto.field_numbers.strings => {
                const s = try decoder.readString();
                try strings_list.append(allocator, s);
            },
            AttributeProto.field_numbers.tensors => {
                const msg_data = try decoder.readLengthDelimited();
                var sub_decoder = Decoder.init(msg_data);
                const tensor = try parseTensorProto(allocator, &sub_decoder, msg_data.len);
                try tensors_list.append(allocator, tensor);
            },
            AttributeProto.field_numbers.attr_type => {
                const value = try decoder.readVarintI32();
                attr.attr_type = @enumFromInt(value);
            },
            else => {
                try decoder.skipField(header.wire_type);
            },
        }
    }

    if (floats_list.items.len > 0) attr.floats = try floats_list.toOwnedSlice(allocator);
    if (ints_list.items.len > 0) attr.ints = try ints_list.toOwnedSlice(allocator);
    if (strings_list.items.len > 0) attr.strings = try strings_list.toOwnedSlice(allocator);
    if (tensors_list.items.len > 0) attr.tensors = try tensors_list.toOwnedSlice(allocator);

    return attr;
}

// ============================================================================
// TensorProto Parser
// ============================================================================

fn parseTensorProto(allocator: std.mem.Allocator, decoder: *Decoder, limit: usize) ParseError!TensorProto {
    var tensor = TensorProto{};

    var dims_list: std.ArrayList(i64) = .empty;
    var external_data_list: std.ArrayList(types.ExternalDataEntry) = .empty;

    const end_pos = decoder.pos + limit;
    while (decoder.pos < end_pos) {
        const header = try decoder.readFieldHeader();

        switch (header.field_number) {
            TensorProto.field_numbers.dims => {
                if (header.wire_type == .length_delimited) {
                    const values = try decoder.readPackedInt64s(allocator);
                    for (values) |v| try dims_list.append(allocator, v);
                } else {
                    try dims_list.append(allocator, try decoder.readVarintI64());
                }
            },
            TensorProto.field_numbers.data_type => {
                const value = try decoder.readVarintI32();
                tensor.data_type = @enumFromInt(value);
            },
            TensorProto.field_numbers.float_data => {
                if (header.wire_type == .length_delimited) {
                    tensor.float_data = try decoder.readPackedFloats(allocator);
                } else {
                    // Single float - shouldn't happen but handle it
                    const f = try decoder.readFloat();
                    const arr = try allocator.alloc(f32, 1);
                    arr[0] = f;
                    tensor.float_data = arr;
                }
            },
            TensorProto.field_numbers.int32_data => {
                if (header.wire_type == .length_delimited) {
                    tensor.int32_data = try decoder.readPackedInt32s(allocator);
                } else {
                    const v = try decoder.readVarintI32();
                    const arr = try allocator.alloc(i32, 1);
                    arr[0] = v;
                    tensor.int32_data = arr;
                }
            },
            TensorProto.field_numbers.int64_data => {
                if (header.wire_type == .length_delimited) {
                    tensor.int64_data = try decoder.readPackedInt64s(allocator);
                } else {
                    const v = try decoder.readVarintI64();
                    const arr = try allocator.alloc(i64, 1);
                    arr[0] = v;
                    tensor.int64_data = arr;
                }
            },
            TensorProto.field_numbers.name => {
                tensor.name = try decoder.readString();
            },
            TensorProto.field_numbers.raw_data => {
                tensor.raw_data = try decoder.readLengthDelimited();
            },
            TensorProto.field_numbers.double_data => {
                if (header.wire_type == .length_delimited) {
                    tensor.double_data = try decoder.readPackedDoubles(allocator);
                } else {
                    const f = try decoder.readDouble();
                    const arr = try allocator.alloc(f64, 1);
                    arr[0] = f;
                    tensor.double_data = arr;
                }
            },
            TensorProto.field_numbers.doc_string => {
                tensor.doc_string = try decoder.readString();
            },
            TensorProto.field_numbers.external_data => {
                // Parse StringStringEntryProto (key-value pair)
                const msg_data = try decoder.readLengthDelimited();
                var sub_decoder = Decoder.init(msg_data);
                const entry = try parseStringStringEntry(allocator, &sub_decoder, msg_data.len);
                try external_data_list.append(allocator, entry);
            },
            TensorProto.field_numbers.data_location => {
                const value = try decoder.readVarintI32();
                tensor.data_location = @enumFromInt(value);
            },
            else => {
                try decoder.skipField(header.wire_type);
            },
        }
    }

    if (dims_list.items.len > 0) tensor.dims = try dims_list.toOwnedSlice(allocator);
    if (external_data_list.items.len > 0) tensor.external_data = try external_data_list.toOwnedSlice(allocator);

    return tensor;
}

/// Parse StringStringEntryProto (key-value pair for external data)
fn parseStringStringEntry(allocator: std.mem.Allocator, decoder: *Decoder, limit: usize) ParseError!types.ExternalDataEntry {
    _ = allocator;
    var entry: types.ExternalDataEntry = .{ .key = "", .value = "" };

    const end_pos = decoder.pos + limit;
    while (decoder.pos < end_pos) {
        const header = try decoder.readFieldHeader();

        switch (header.field_number) {
            1 => { // key
                entry.key = try decoder.readString();
            },
            2 => { // value
                entry.value = try decoder.readString();
            },
            else => {
                try decoder.skipField(header.wire_type);
            },
        }
    }

    return entry;
}

// ============================================================================
// ValueInfoProto Parser
// ============================================================================

fn parseValueInfoProto(allocator: std.mem.Allocator, decoder: *Decoder, limit: usize) ParseError!ValueInfoProto {
    var info = ValueInfoProto{};

    const end_pos = decoder.pos + limit;
    while (decoder.pos < end_pos) {
        const header = try decoder.readFieldHeader();

        switch (header.field_number) {
            ValueInfoProto.field_numbers.name => {
                info.name = try decoder.readString();
            },
            ValueInfoProto.field_numbers.type_info => {
                const msg_data = try decoder.readLengthDelimited();
                var sub_decoder = Decoder.init(msg_data);
                info.type_info = try parseTypeProto(allocator, &sub_decoder, msg_data.len);
            },
            ValueInfoProto.field_numbers.doc_string => {
                info.doc_string = try decoder.readString();
            },
            else => {
                try decoder.skipField(header.wire_type);
            },
        }
    }

    return info;
}

// ============================================================================
// TypeProto Parser
// ============================================================================

fn parseTypeProto(allocator: std.mem.Allocator, decoder: *Decoder, limit: usize) ParseError!TypeProto {
    var type_proto = TypeProto{};

    const end_pos = decoder.pos + limit;
    while (decoder.pos < end_pos) {
        const header = try decoder.readFieldHeader();

        switch (header.field_number) {
            TypeProto.field_numbers.tensor_type => {
                const msg_data = try decoder.readLengthDelimited();
                var sub_decoder = Decoder.init(msg_data);
                type_proto.tensor_type = try parseTensorTypeProto(allocator, &sub_decoder, msg_data.len);
            },
            else => {
                try decoder.skipField(header.wire_type);
            },
        }
    }

    return type_proto;
}

fn parseTensorTypeProto(allocator: std.mem.Allocator, decoder: *Decoder, limit: usize) ParseError!TypeProto.Tensor {
    var tensor = TypeProto.Tensor{};

    const end_pos = decoder.pos + limit;
    while (decoder.pos < end_pos) {
        const header = try decoder.readFieldHeader();

        switch (header.field_number) {
            1 => { // elem_type
                const value = try decoder.readVarintI32();
                tensor.elem_type = @enumFromInt(value);
            },
            2 => { // shape
                const msg_data = try decoder.readLengthDelimited();
                var sub_decoder = Decoder.init(msg_data);
                tensor.shape = try parseTensorShapeProto(allocator, &sub_decoder, msg_data.len);
            },
            else => {
                try decoder.skipField(header.wire_type);
            },
        }
    }

    return tensor;
}

fn parseTensorShapeProto(allocator: std.mem.Allocator, decoder: *Decoder, limit: usize) ParseError!TensorShapeProto {
    var shape = TensorShapeProto{};

    var dim_list: std.ArrayList(TensorShapeProto.Dimension) = .empty;

    const end_pos = decoder.pos + limit;
    while (decoder.pos < end_pos) {
        const header = try decoder.readFieldHeader();

        switch (header.field_number) {
            TensorShapeProto.field_numbers.dim => {
                const msg_data = try decoder.readLengthDelimited();
                var sub_decoder = Decoder.init(msg_data);
                const dim = try parseDimensionProto(&sub_decoder, msg_data.len);
                try dim_list.append(allocator, dim);
            },
            else => {
                try decoder.skipField(header.wire_type);
            },
        }
    }

    shape.dim = try dim_list.toOwnedSlice(allocator);
    return shape;
}

fn parseDimensionProto(decoder: *Decoder, limit: usize) ParseError!TensorShapeProto.Dimension {
    var dim = TensorShapeProto.Dimension{};

    const end_pos = decoder.pos + limit;
    while (decoder.pos < end_pos) {
        const header = try decoder.readFieldHeader();

        switch (header.field_number) {
            TensorShapeProto.Dimension.field_numbers.dim_value => {
                dim.dim_value = try decoder.readVarintI64();
            },
            TensorShapeProto.Dimension.field_numbers.dim_param => {
                dim.dim_param = try decoder.readString();
            },
            else => {
                try decoder.skipField(header.wire_type);
            },
        }
    }

    return dim;
}

// ============================================================================
// Tests
// ============================================================================

test "parse empty model" {
    // Minimal valid model: just ir_version = 1
    // Field 1 (ir_version), varint, value 1: 0x08 0x01
    const data = [_]u8{ 0x08, 0x01 };
    const model = try parseModel(std.testing.allocator, &data);

    try std.testing.expectEqual(@as(i64, 1), model.ir_version);
    try std.testing.expectEqual(@as(?*const GraphProto, null), if (model.graph) |*g| g else null);
}

test "parse model with producer name" {
    // ir_version = 7, producer_name = "test"
    // Field 1 (varint 7): 0x08 0x07
    // Field 2 (string "test"): 0x12 0x04 "test"
    const data = [_]u8{
        0x08, 0x07, // ir_version = 7
        0x12, 0x04, 't', 'e', 's', 't', // producer_name = "test"
    };
    const model = try parseModel(std.testing.allocator, &data);

    try std.testing.expectEqual(@as(i64, 7), model.ir_version);
    try std.testing.expectEqualStrings("test", model.producer_name.?);
}

test "parse opset import" {
    // ir_version = 7, opset_import { domain = "", version = 17 }
    // Field 8 is opset_import (length-delimited message)
    const data = [_]u8{
        0x08, 0x07, // ir_version = 7
        0x42, 0x04, // field 8, length 4
        0x10, 0x11, // nested: field 2 (version) = 17
        0x0a, 0x00, // nested: field 1 (domain) = "" (empty string)
    };
    const model = try parseModel(std.testing.allocator, &data);
    defer std.testing.allocator.free(model.opset_import);

    try std.testing.expectEqual(@as(usize, 1), model.opset_import.len);
    try std.testing.expectEqual(@as(i64, 17), model.opset_import[0].version);
}

test "parse simple graph with node" {
    // This creates a model with a graph containing one Add node
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Build the protobuf manually:
    // Model { ir_version: 8, graph: Graph { node: [Node { op_type: "Add", input: ["a", "b"], output: ["c"] }] } }

    // NodeProto: op_type="Add", input=["a","b"], output=["c"]
    const node_bytes = [_]u8{
        0x0a, 0x01, 'a', // input "a"
        0x0a, 0x01, 'b', // input "b"
        0x12, 0x01, 'c', // output "c"
        0x22, 0x03, 'A', 'd', 'd', // op_type "Add"
    };

    // GraphProto with one node
    var graph_bytes: std.ArrayList(u8) = .empty;
    // Field 1 (node), length-delimited
    try graph_bytes.append(allocator, 0x0a);
    try graph_bytes.append(allocator, @intCast(node_bytes.len));
    try graph_bytes.appendSlice(allocator, &node_bytes);

    // ModelProto
    var model_bytes: std.ArrayList(u8) = .empty;
    try model_bytes.append(allocator, 0x08); // field 1, varint
    try model_bytes.append(allocator, 0x08); // ir_version = 8
    try model_bytes.append(allocator, 0x3a); // field 7, length-delimited
    try model_bytes.append(allocator, @intCast(graph_bytes.items.len));
    try model_bytes.appendSlice(allocator, graph_bytes.items);

    const model = try parseModel(allocator, model_bytes.items);

    try std.testing.expectEqual(@as(i64, 8), model.ir_version);
    try std.testing.expect(model.graph != null);

    const graph = model.graph.?;
    try std.testing.expectEqual(@as(usize, 1), graph.node.len);

    const node = graph.node[0];
    try std.testing.expectEqualStrings("Add", node.op_type.?);
    try std.testing.expectEqual(@as(usize, 2), node.input.len);
    try std.testing.expectEqualStrings("a", node.input[0]);
    try std.testing.expectEqualStrings("b", node.input[1]);
    try std.testing.expectEqual(@as(usize, 1), node.output.len);
    try std.testing.expectEqualStrings("c", node.output[0]);
}

test "parse tensor with dims and data" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // TensorProto: dims=[2,3], data_type=FLOAT, raw_data=<24 bytes>
    const raw_data = [_]u8{
        // 6 floats: 1.0, 2.0, 3.0, 4.0, 5.0, 6.0
        0x00, 0x00, 0x80, 0x3f, // 1.0
        0x00, 0x00, 0x00, 0x40, // 2.0
        0x00, 0x00, 0x40, 0x40, // 3.0
        0x00, 0x00, 0x80, 0x40, // 4.0
        0x00, 0x00, 0xa0, 0x40, // 5.0
        0x00, 0x00, 0xc0, 0x40, // 6.0
    };

    var tensor_bytes: std.ArrayList(u8) = .empty;
    // dims (packed): field 1, length 2, values [2, 3]
    try tensor_bytes.append(allocator, 0x0a); // field 1, length-delimited
    try tensor_bytes.append(allocator, 0x02); // length 2
    try tensor_bytes.append(allocator, 0x02); // varint 2
    try tensor_bytes.append(allocator, 0x03); // varint 3
    // data_type: field 2, varint 1 (FLOAT)
    try tensor_bytes.append(allocator, 0x10); // field 2, varint
    try tensor_bytes.append(allocator, 0x01); // value 1
    // raw_data: field 9, length-delimited
    try tensor_bytes.append(allocator, 0x4a); // field 9, length-delimited
    try tensor_bytes.append(allocator, @intCast(raw_data.len));
    try tensor_bytes.appendSlice(allocator, &raw_data);
    // name: field 8, string "weights"
    try tensor_bytes.appendSlice(allocator, &[_]u8{ 0x42, 0x07, 'w', 'e', 'i', 'g', 'h', 't', 's' });

    var decoder = Decoder.init(tensor_bytes.items);
    const tensor = try parseTensorProto(allocator, &decoder, tensor_bytes.items.len);

    try std.testing.expectEqual(@as(usize, 2), tensor.dims.len);
    try std.testing.expectEqual(@as(i64, 2), tensor.dims[0]);
    try std.testing.expectEqual(@as(i64, 3), tensor.dims[1]);
    try std.testing.expectEqual(DataType.float, tensor.data_type);
    try std.testing.expectEqualStrings("weights", tensor.name.?);
    try std.testing.expectEqual(@as(usize, 24), tensor.raw_data.?.len);
    try std.testing.expectEqual(@as(usize, 6), tensor.numel());
}
