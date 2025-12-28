//! ONNX protobuf message types.
//!
//! These types mirror the ONNX protobuf definitions from:
//! https://github.com/onnx/onnx/blob/main/onnx/onnx.proto3
//!
//! Field numbers match the proto definitions for parsing.

const std = @import("std");
const proto = @import("proto.zig");

const Decoder = proto.Decoder;
const WireType = proto.WireType;

// ============================================================================
// ONNX Data Types
// ============================================================================

/// ONNX tensor element types.
/// Values match TensorProto.DataType enum in onnx.proto3.
pub const DataType = enum(i32) {
    undefined = 0,
    float = 1, // f32
    uint8 = 2,
    int8 = 3,
    uint16 = 4,
    int16 = 5,
    int32 = 6,
    int64 = 7,
    string = 8,
    bool = 9,
    float16 = 10, // f16
    double = 11, // f64
    uint32 = 12,
    uint64 = 13,
    complex64 = 14,
    complex128 = 15,
    bfloat16 = 16, // bf16
    float8e4m3fn = 17,
    float8e4m3fnuz = 18,
    float8e5m2 = 19,
    float8e5m2fnuz = 20,
    uint4 = 21,
    int4 = 22,
    _,

    pub fn byteSize(self: DataType) ?usize {
        return switch (self) {
            .float => 4,
            .uint8 => 1,
            .int8 => 1,
            .uint16 => 2,
            .int16 => 2,
            .int32 => 4,
            .int64 => 8,
            .bool => 1,
            .float16 => 2,
            .double => 8,
            .uint32 => 4,
            .uint64 => 8,
            .bfloat16 => 2,
            else => null,
        };
    }

    pub fn name(self: DataType) []const u8 {
        return switch (self) {
            .undefined => "undefined",
            .float => "float32",
            .uint8 => "uint8",
            .int8 => "int8",
            .uint16 => "uint16",
            .int16 => "int16",
            .int32 => "int32",
            .int64 => "int64",
            .string => "string",
            .bool => "bool",
            .float16 => "float16",
            .double => "float64",
            .uint32 => "uint32",
            .uint64 => "uint64",
            .bfloat16 => "bfloat16",
            else => "unknown",
        };
    }
};

// ============================================================================
// Attribute Types
// ============================================================================

/// Attribute type enum.
/// Values match AttributeProto.AttributeType in onnx.proto3.
pub const AttributeType = enum(i32) {
    undefined = 0,
    float = 1,
    int = 2,
    string = 3,
    tensor = 4,
    graph = 5,
    sparse_tensor = 11,
    type_proto = 13,
    floats = 6,
    ints = 7,
    strings = 8,
    tensors = 9,
    graphs = 10,
    sparse_tensors = 12,
    type_protos = 14,
    _,
};

// ============================================================================
// TensorProto - Field numbers from onnx.proto3
// ============================================================================

/// Data location for TensorProto
pub const DataLocation = enum(i32) {
    default = 0, // Data stored in raw_data or typed arrays
    external = 1, // Data stored in external file
    _,
};

/// External data entry (key-value pair)
pub const ExternalDataEntry = struct {
    key: []const u8,
    value: []const u8,
};

/// Tensor data, used for weights/initializers.
pub const TensorProto = struct {
    // Field 1: repeated int64 dims
    dims: []const i64 = &.{},

    // Field 2: int32 data_type
    data_type: DataType = .undefined,

    // Field 3: TensorProto.Segment segment (not commonly used)

    // Field 4: repeated float float_data (for FLOAT type)
    float_data: []const f32 = &.{},

    // Field 5: repeated int32 int32_data (for INT32, INT16, INT8, UINT16, UINT8, BOOL, FLOAT16)
    int32_data: []const i32 = &.{},

    // Field 6: repeated bytes string_data (for STRING type)
    string_data: []const []const u8 = &.{},

    // Field 7: repeated int64 int64_data (for INT64)
    int64_data: []const i64 = &.{},

    // Field 8: string name
    name: ?[]const u8 = null,

    // Field 12: string doc_string
    doc_string: ?[]const u8 = null,

    // Field 9: bytes raw_data (serialized tensor data)
    raw_data: ?[]const u8 = null,

    // Field 13: repeated StringStringEntryProto external_data
    external_data: []const ExternalDataEntry = &.{},

    // Field 14: int32 data_location
    data_location: DataLocation = .default,

    // Field 10: repeated double double_data (for DOUBLE)
    double_data: []const f64 = &.{},

    // Field 11: repeated uint64 uint64_data (for UINT64, UINT32)
    uint64_data: []const u64 = &.{},

    /// Check if tensor uses external data storage
    pub fn isExternal(self: *const TensorProto) bool {
        return self.data_location == .external or self.external_data.len > 0;
    }

    /// Get external data location (file path)
    pub fn getExternalLocation(self: *const TensorProto) ?[]const u8 {
        for (self.external_data) |entry| {
            if (std.mem.eql(u8, entry.key, "location")) {
                return entry.value;
            }
        }
        return null;
    }

    /// Get external data offset
    pub fn getExternalOffset(self: *const TensorProto) usize {
        for (self.external_data) |entry| {
            if (std.mem.eql(u8, entry.key, "offset")) {
                return std.fmt.parseInt(usize, entry.value, 10) catch 0;
            }
        }
        return 0;
    }

    /// Get external data length (0 means read to end or infer from shape)
    pub fn getExternalLength(self: *const TensorProto) usize {
        for (self.external_data) |entry| {
            if (std.mem.eql(u8, entry.key, "length")) {
                return std.fmt.parseInt(usize, entry.value, 10) catch 0;
            }
        }
        return 0;
    }

    /// Get the raw bytes of the tensor data, regardless of storage format.
    pub fn getRawBytes(self: *const TensorProto) ?[]const u8 {
        if (self.raw_data) |data| return data;
        if (self.float_data.len > 0) return std.mem.sliceAsBytes(self.float_data);
        if (self.int32_data.len > 0) return std.mem.sliceAsBytes(self.int32_data);
        if (self.int64_data.len > 0) return std.mem.sliceAsBytes(self.int64_data);
        if (self.double_data.len > 0) return std.mem.sliceAsBytes(self.double_data);
        if (self.uint64_data.len > 0) return std.mem.sliceAsBytes(self.uint64_data);
        return null;
    }

    /// Get total number of elements.
    pub fn numel(self: *const TensorProto) usize {
        if (self.dims.len == 0) return 0;
        var n: usize = 1;
        for (self.dims) |d| {
            if (d < 0) return 0; // Dynamic dim
            n *= @intCast(d);
        }
        return n;
    }

    pub const field_numbers = struct {
        pub const dims = 1;
        pub const data_type = 2;
        pub const float_data = 4;
        pub const int32_data = 5;
        pub const string_data = 6;
        pub const int64_data = 7;
        pub const name = 8;
        pub const raw_data = 9;
        pub const double_data = 10;
        pub const uint64_data = 11;
        pub const doc_string = 12;
        pub const external_data = 13;
        pub const data_location = 14;
    };
};

// ============================================================================
// AttributeProto - Field numbers from onnx.proto3
// ============================================================================

/// Node attribute (e.g., kernel_shape for Conv).
pub const AttributeProto = struct {
    // Field 1: string name
    name: ?[]const u8 = null,

    // Field 21: string ref_attr_name (for subgraph references)

    // Field 13: string doc_string

    // Field 20: AttributeType type
    attr_type: AttributeType = .undefined,

    // Single values:
    // Field 2: float f
    f: ?f32 = null,
    // Field 3: int64 i
    i: ?i64 = null,
    // Field 4: bytes s
    s: ?[]const u8 = null,
    // Field 5: TensorProto t
    t: ?TensorProto = null,
    // Field 6: GraphProto g (for subgraphs like If/Loop)

    // Repeated values:
    // Field 7: repeated float floats
    floats: []const f32 = &.{},
    // Field 8: repeated int64 ints
    ints: []const i64 = &.{},
    // Field 9: repeated bytes strings
    strings: []const []const u8 = &.{},
    // Field 10: repeated TensorProto tensors
    tensors: []const TensorProto = &.{},
    // Field 11: repeated GraphProto graphs

    pub const field_numbers = struct {
        pub const name = 1;
        pub const f = 2;
        pub const i = 3;
        pub const s = 4;
        pub const t = 5;
        pub const g = 6;
        pub const floats = 7;
        pub const ints = 8;
        pub const strings = 9;
        pub const tensors = 10;
        pub const graphs = 11;
        pub const doc_string = 13;
        pub const attr_type = 20;
        pub const ref_attr_name = 21;
    };
};

// ============================================================================
// ValueInfoProto - Field numbers from onnx.proto3
// ============================================================================

/// Type information for graph inputs/outputs.
pub const TypeProto = struct {
    pub const Tensor = struct {
        // Field 1: int32 elem_type
        elem_type: DataType = .undefined,
        // Field 2: TensorShapeProto shape
        shape: ?TensorShapeProto = null,
    };

    // Field 1: TypeProto.Tensor tensor_type
    tensor_type: ?Tensor = null,

    // Field 4: TypeProto.Sequence sequence_type
    // Field 5: TypeProto.Map map_type
    // Field 8: TypeProto.Optional optional_type
    // Field 9: TypeProto.SparseTensor sparse_tensor_type

    // Field 6: string denotation

    pub const field_numbers = struct {
        pub const tensor_type = 1;
        pub const sequence_type = 4;
        pub const map_type = 5;
        pub const denotation = 6;
        pub const optional_type = 8;
        pub const sparse_tensor_type = 9;
    };
};

/// Shape information.
pub const TensorShapeProto = struct {
    pub const Dimension = struct {
        // Oneof: dim_value (int64) or dim_param (string)
        dim_value: ?i64 = null,
        dim_param: ?[]const u8 = null,
        // Field 3: string denotation

        pub fn getValue(self: *const Dimension) ?i64 {
            return self.dim_value;
        }

        pub fn isDynamic(self: *const Dimension) bool {
            return self.dim_value == null;
        }

        pub const field_numbers = struct {
            pub const dim_value = 1;
            pub const dim_param = 2;
            pub const denotation = 3;
        };
    };

    // Field 1: repeated Dimension dim
    dim: []const Dimension = &.{},

    /// Get shape as i64 array (-1 for dynamic dims).
    pub fn toShape(self: *const TensorShapeProto, allocator: std.mem.Allocator) ![]i64 {
        const shape = try allocator.alloc(i64, self.dim.len);
        for (self.dim, 0..) |d, idx| {
            shape[idx] = d.dim_value orelse -1;
        }
        return shape;
    }

    pub const field_numbers = struct {
        pub const dim = 1;
    };
};

/// Input/output value information.
pub const ValueInfoProto = struct {
    // Field 1: string name
    name: ?[]const u8 = null,

    // Field 2: TypeProto type
    type_info: ?TypeProto = null,

    // Field 3: string doc_string
    doc_string: ?[]const u8 = null,

    pub const field_numbers = struct {
        pub const name = 1;
        pub const type_info = 2;
        pub const doc_string = 3;
    };
};

// ============================================================================
// NodeProto - Field numbers from onnx.proto3
// ============================================================================

/// A single operation node in the graph.
pub const NodeProto = struct {
    // Field 1: repeated string input
    input: []const []const u8 = &.{},

    // Field 2: repeated string output
    output: []const []const u8 = &.{},

    // Field 3: string name
    name: ?[]const u8 = null,

    // Field 4: string op_type
    op_type: ?[]const u8 = null,

    // Field 7: string domain
    domain: ?[]const u8 = null,

    // Field 5: repeated AttributeProto attribute
    attribute: []const AttributeProto = &.{},

    // Field 6: string doc_string
    doc_string: ?[]const u8 = null,

    /// Get attribute by name.
    pub fn getAttribute(self: *const NodeProto, attr_name: []const u8) ?*const AttributeProto {
        for (self.attribute) |*attr| {
            if (attr.name) |name| {
                if (std.mem.eql(u8, name, attr_name)) {
                    return attr;
                }
            }
        }
        return null;
    }

    /// Get int attribute or default.
    pub fn getIntAttr(self: *const NodeProto, attr_name: []const u8, default: i64) i64 {
        if (self.getAttribute(attr_name)) |attr| {
            return attr.i orelse default;
        }
        return default;
    }

    /// Get float attribute or default.
    pub fn getFloatAttr(self: *const NodeProto, attr_name: []const u8, default: f32) f32 {
        if (self.getAttribute(attr_name)) |attr| {
            return attr.f orelse default;
        }
        return default;
    }

    /// Get ints attribute or empty.
    pub fn getIntsAttr(self: *const NodeProto, attr_name: []const u8) []const i64 {
        if (self.getAttribute(attr_name)) |attr| {
            return attr.ints;
        }
        return &.{};
    }

    pub const field_numbers = struct {
        pub const input = 1;
        pub const output = 2;
        pub const name = 3;
        pub const op_type = 4;
        pub const attribute = 5;
        pub const doc_string = 6;
        pub const domain = 7;
    };
};

// ============================================================================
// GraphProto - Field numbers from onnx.proto3
// ============================================================================

/// The computation graph.
pub const GraphProto = struct {
    // Field 1: repeated NodeProto node
    node: []const NodeProto = &.{},

    // Field 2: string name
    name: ?[]const u8 = null,

    // Field 5: repeated TensorProto initializer (weights)
    initializer: []const TensorProto = &.{},

    // Field 15: repeated SparseTensorProto sparse_initializer

    // Field 10: string doc_string
    doc_string: ?[]const u8 = null,

    // Field 11: repeated ValueInfoProto input
    input: []const ValueInfoProto = &.{},

    // Field 12: repeated ValueInfoProto output
    output: []const ValueInfoProto = &.{},

    // Field 13: repeated ValueInfoProto value_info (intermediate shapes)
    value_info: []const ValueInfoProto = &.{},

    // Field 14: repeated TensorAnnotation quantization_annotation

    pub const field_numbers = struct {
        pub const node = 1;
        pub const name = 2;
        pub const initializer = 5;
        pub const doc_string = 10;
        pub const input = 11;
        pub const output = 12;
        pub const value_info = 13;
        pub const quantization_annotation = 14;
        pub const sparse_initializer = 15;
    };
};

// ============================================================================
// OperatorSetIdProto - Field numbers from onnx.proto3
// ============================================================================

/// Operator set version.
pub const OperatorSetIdProto = struct {
    // Field 1: string domain
    domain: ?[]const u8 = null,

    // Field 2: int64 version
    version: i64 = 0,

    pub const field_numbers = struct {
        pub const domain = 1;
        pub const version = 2;
    };
};

// ============================================================================
// ModelProto - Field numbers from onnx.proto3
// ============================================================================

/// The top-level ONNX model.
pub const ModelProto = struct {
    // Field 1: int64 ir_version
    ir_version: i64 = 0,

    // Field 8: repeated OperatorSetIdProto opset_import
    opset_import: []const OperatorSetIdProto = &.{},

    // Field 2: string producer_name
    producer_name: ?[]const u8 = null,

    // Field 3: string producer_version
    producer_version: ?[]const u8 = null,

    // Field 4: string domain
    domain: ?[]const u8 = null,

    // Field 5: int64 model_version
    model_version: i64 = 0,

    // Field 6: string doc_string
    doc_string: ?[]const u8 = null,

    // Field 7: GraphProto graph
    graph: ?GraphProto = null,

    // Field 14: repeated StringStringEntryProto metadata_props
    // Field 10: repeated TrainingInfoProto training_info
    // Field 20: repeated FunctionProto functions

    /// Get the opset version for a domain (empty string = default ONNX domain).
    pub fn getOpsetVersion(self: *const ModelProto, domain_name: []const u8) ?i64 {
        for (self.opset_import) |opset| {
            const opset_domain = opset.domain orelse "";
            if (std.mem.eql(u8, opset_domain, domain_name)) {
                return opset.version;
            }
        }
        return null;
    }

    pub const field_numbers = struct {
        pub const ir_version = 1;
        pub const producer_name = 2;
        pub const producer_version = 3;
        pub const domain = 4;
        pub const model_version = 5;
        pub const doc_string = 6;
        pub const graph = 7;
        pub const opset_import = 8;
        pub const training_info = 10;
        pub const metadata_props = 14;
        pub const functions = 20;
    };
};

// ============================================================================
// Tests
// ============================================================================

test "DataType byte sizes" {
    try std.testing.expectEqual(@as(?usize, 4), DataType.float.byteSize());
    try std.testing.expectEqual(@as(?usize, 2), DataType.float16.byteSize());
    try std.testing.expectEqual(@as(?usize, 1), DataType.int8.byteSize());
    try std.testing.expectEqual(@as(?usize, 8), DataType.int64.byteSize());
    try std.testing.expectEqual(@as(?usize, null), DataType.string.byteSize());
}

test "TensorProto numel" {
    const tensor = TensorProto{
        .dims = &[_]i64{ 2, 3, 4 },
        .data_type = .float,
    };
    try std.testing.expectEqual(@as(usize, 24), tensor.numel());
}

test "NodeProto getAttribute" {
    const attrs = [_]AttributeProto{
        .{ .name = "alpha", .f = 1.0 },
        .{ .name = "beta", .i = 42 },
    };
    const node = NodeProto{
        .op_type = "Gemm",
        .attribute = &attrs,
    };

    try std.testing.expectEqual(@as(f32, 1.0), node.getFloatAttr("alpha", 0.0));
    try std.testing.expectEqual(@as(i64, 42), node.getIntAttr("beta", 0));
    try std.testing.expectEqual(@as(f32, 0.5), node.getFloatAttr("gamma", 0.5));
}
