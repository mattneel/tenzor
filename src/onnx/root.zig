//! ONNX Model Loader for Tenzor
//!
//! Load and execute ONNX computation graphs using tenzor's tensor operations.
//!
//! ```zig
//! const onnx = @import("onnx");
//! const model = try onnx.load(allocator, model_bytes);
//! defer model.deinit(allocator);
//! ```

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const proto = @import("proto.zig");
pub const types = @import("types.zig");
pub const parser = @import("parser.zig");
pub const graph = @import("graph.zig");
pub const builder = @import("builder.zig");
pub const executor = @import("executor.zig");

// Re-export key types from proto types
pub const ModelProto = types.ModelProto;
pub const GraphProto = types.GraphProto;
pub const NodeProto = types.NodeProto;
pub const TensorProto = types.TensorProto;
pub const DataType = types.DataType;
pub const AttributeProto = types.AttributeProto;

// Re-export key types from runtime graph
pub const Graph = graph.Graph;
pub const DType = graph.DType;
pub const OpType = graph.OpType;
pub const Node = graph.Node;
pub const TensorInfo = graph.TensorInfo;
pub const WeightData = graph.WeightData;

// Re-export executor types
pub const Executor = executor.Executor;
pub const RuntimeTensor = executor.RuntimeTensor;

/// Load an ONNX model from raw bytes (returns proto types).
pub fn load(allocator: Allocator, data: []const u8) !ModelProto {
    return parser.parseModel(allocator, data);
}

/// Load an ONNX model from a file path (returns proto types).
pub fn loadFile(allocator: Allocator, path: []const u8) !ModelProto {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    const data = try file.readToEndAlloc(allocator, std.math.maxInt(usize));
    defer allocator.free(data);

    return load(allocator, data);
}

/// Build a runtime graph from raw ONNX bytes.
/// Note: Uses an arena for parsing. The returned Graph owns its memory.
pub fn buildFromBytes(allocator: Allocator, data: []const u8) !Graph {
    // Use arena for parsing proto - it's discarded after buildGraph
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    const model = try parser.parseModel(arena.allocator(), data);
    return builder.buildGraph(allocator, model);
}

/// Build a runtime graph from an ONNX file.
/// Note: Uses an arena for parsing. The returned Graph owns its memory.
pub fn buildFromFile(allocator: Allocator, path: []const u8) !Graph {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    // Use arena for file data and parsing - both discarded after buildGraph
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    const data = try file.readToEndAlloc(arena.allocator(), std.math.maxInt(usize));
    const model = try parser.parseModel(arena.allocator(), data);
    return builder.buildGraph(allocator, model);
}

test {
    _ = proto;
    _ = types;
    _ = parser;
    _ = graph;
    _ = builder;
    _ = executor;
    _ = @import("test_real_model.zig");
}
