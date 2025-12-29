//! Test parsing real ONNX models from Chatterbox

const std = @import("std");
const root = @import("root.zig");
const parser = @import("parser.zig");
const builder = @import("builder.zig");

fn readFile(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    const stat = try file.stat();
    const data = try allocator.alloc(u8, stat.size);
    errdefer allocator.free(data);

    var total_read: usize = 0;
    while (total_read < stat.size) {
        const bytes_read = file.read(data[total_read..]) catch |err| return err;
        if (bytes_read == 0) break;
        total_read += bytes_read;
    }
    return data;
}

test "parse embed_tokens.onnx" {
    const allocator = std.testing.allocator;

    // Load the model file
    const data = readFile(allocator, "tests/models/embed_tokens.onnx") catch |err| {
        std.debug.print("Skipping test: {}\n", .{err});
        return; // Skip if file not found
    };
    defer allocator.free(data);

    std.debug.print("\nModel file size: {} bytes\n", .{data.len});

    // Use arena for parsing - proto types don't have deinit
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    // Parse the model
    const model = try parser.parseModel(arena.allocator(), data);

    // Print model info
    std.debug.print("IR version: {}\n", .{model.ir_version});
    if (model.producer_name) |name| {
        std.debug.print("Producer: {s}\n", .{name});
    }
    if (model.producer_version) |ver| {
        std.debug.print("Producer version: {s}\n", .{ver});
    }

    // Print opset imports
    std.debug.print("Opset imports: {}\n", .{model.opset_import.len});
    for (model.opset_import) |opset| {
        if (opset.domain) |domain| {
            std.debug.print("  Domain: {s}, version: {}\n", .{ domain, opset.version });
        } else {
            std.debug.print("  Default domain, version: {}\n", .{opset.version});
        }
    }

    // Print graph info
    if (model.graph) |graph| {
        if (graph.name) |name| {
            std.debug.print("Graph name: {s}\n", .{name});
        }
        std.debug.print("Inputs: {}\n", .{graph.input.len});
        for (graph.input) |input| {
            if (input.name) |name| {
                std.debug.print("  - {s}\n", .{name});
            }
        }
        std.debug.print("Outputs: {}\n", .{graph.output.len});
        for (graph.output) |output| {
            if (output.name) |name| {
                std.debug.print("  - {s}\n", .{name});
            }
        }
        std.debug.print("Nodes: {}\n", .{graph.node.len});
        std.debug.print("Initializers: {}\n", .{graph.initializer.len});

        // Print first 10 nodes
        const max_nodes = @min(graph.node.len, 10);
        for (graph.node[0..max_nodes]) |node| {
            if (node.op_type) |op| {
                std.debug.print("  Node: {s}", .{op});
                if (node.name) |name| {
                    std.debug.print(" ({s})", .{name});
                }
                std.debug.print("\n", .{});
            }
        }
        if (graph.node.len > 10) {
            std.debug.print("  ... and {} more nodes\n", .{graph.node.len - 10});
        }
    }

    // Verify we parsed something
    try std.testing.expect(model.ir_version > 0);
    try std.testing.expect(model.graph != null);
}

test "build runtime graph from embed_tokens.onnx" {
    const allocator = std.testing.allocator;

    // Load the model file
    const data = readFile(allocator, "tests/models/embed_tokens.onnx") catch |err| {
        std.debug.print("Skipping test: {}\n", .{err});
        return;
    };
    defer allocator.free(data);

    // Build runtime graph
    var graph = try root.buildFromBytes(allocator, data);
    defer graph.deinit();

    std.debug.print("\nRuntime graph:\n", .{});
    std.debug.print("  Tensors: {}\n", .{graph.tensors.items.len});
    std.debug.print("  Nodes: {}\n", .{graph.nodes.items.len});
    std.debug.print("  Inputs: {}\n", .{graph.inputs.len});
    std.debug.print("  Outputs: {}\n", .{graph.outputs.len});
    std.debug.print("  Weights: {}\n", .{graph.weights.count()});

    // Print op type distribution
    var op_counts: std.AutoHashMapUnmanaged(root.OpType, u32) = .empty;
    defer op_counts.deinit(allocator);

    for (graph.nodes.items) |node| {
        const entry = try op_counts.getOrPut(allocator, node.op_type);
        if (!entry.found_existing) {
            entry.value_ptr.* = 0;
        }
        entry.value_ptr.* += 1;
    }

    std.debug.print("  Op distribution:\n", .{});
    var it = op_counts.iterator();
    while (it.next()) |entry| {
        std.debug.print("    {s}: {}\n", .{ @tagName(entry.key_ptr.*), entry.value_ptr.* });
    }

    // Verify we have nodes
    try std.testing.expect(graph.nodes.items.len > 0);
}

test "run speech_encoder_q4.onnx" {
    const allocator = std.testing.allocator;

    const data = readFile(allocator, "/home/autark/models/chatterbox/onnx/speech_encoder_q4.onnx") catch |err| {
        std.debug.print("Skipping test: {}\n", .{err});
        return;
    };
    defer allocator.free(data);

    var graph = try root.buildFromBytes(allocator, data);
    defer graph.deinit();

    std.debug.print("\nSpeech encoder: {} nodes, {} inputs\n", .{ graph.nodes.items.len, graph.inputs.len });

    var exec = try root.Executor.init(allocator, &graph);
    defer exec.deinit();

    try exec.loadWeights();
    try exec.loadExternalWeights("/home/autark/models/chatterbox/onnx");

    // Set input: audio [batch=1, time=16000]
    const input_data = try allocator.alloc(f32, 16000);
    defer allocator.free(input_data);
    @memset(input_data, 0.1);

    const input_name = graph.tensors.items[graph.inputs[0]].name;
    const input_shape = [_]i64{ 1, 16000 };
    try exec.setInputFromSlice(input_name, f32, input_data, &input_shape);

    // Run with debug tracing
    exec.runDebug() catch |err| {
        std.debug.print("Execution failed: {}\n", .{err});
        return err;
    };

    std.debug.print("Speech encoder completed!\n", .{});
}

test "execute embed_tokens.onnx (external weights)" {
    const allocator = std.testing.allocator;

    // Load the model file
    const data = readFile(allocator, "tests/models/embed_tokens.onnx") catch |err| {
        std.debug.print("Skipping test: {}\n", .{err});
        return;
    };
    defer allocator.free(data);

    // Build runtime graph
    var graph = try root.buildFromBytes(allocator, data);
    defer graph.deinit();

    // Create executor
    var exec = try root.Executor.init(allocator, &graph);
    defer exec.deinit();

    // Load inline weights from graph
    try exec.loadWeights();

    // Count external weights
    var external_count: usize = 0;
    var inline_count: usize = 0;
    var wit = graph.weights.iterator();
    while (wit.next()) |entry| {
        if (entry.value_ptr.*.isExternal()) {
            external_count += 1;
            std.debug.print("  External weight: {s} -> {?s}\n", .{
                entry.key_ptr.*,
                entry.value_ptr.*.external_location,
            });
        } else if (entry.value_ptr.*.data.len > 0) {
            inline_count += 1;
        }
    }

    // Try loading external weights (may fail if weight files not present)
    exec.loadExternalWeights("tests/models") catch |err| {
        std.debug.print("  External weights not loaded: {} (expected if weight files missing)\n", .{err});
    };

    // Check that we have the expected structure
    try std.testing.expectEqual(@as(usize, 7), graph.nodes.items.len);
    try std.testing.expectEqual(@as(usize, 1), graph.inputs.len);
    try std.testing.expectEqual(@as(usize, 1), graph.outputs.len);

    std.debug.print("\nSuccessfully parsed embed_tokens.onnx\n", .{});
    std.debug.print("  Nodes: {}\n", .{graph.nodes.items.len});
    std.debug.print("  Inline weights: {}\n", .{inline_count});
    std.debug.print("  External weights: {}\n", .{external_count});
}
