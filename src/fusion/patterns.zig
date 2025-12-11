//! Fusion pattern definitions and matchers.
//!
//! Defines the recognizable fusion patterns and provides matchers
//! to identify them in expression graphs.

const std = @import("std");
const ops = @import("../ops/expr.zig");
const analyzer = @import("analyzer.zig");

const OpTag = ops.OpTag;
const NodeKind = ops.NodeKind;
const FusionPattern = analyzer.FusionPattern;

/// Maximum number of operations in a fused chain.
pub const MAX_CHAIN_OPS = 32;

/// Describes a matched elementwise fusion pattern.
pub fn ElementwiseFusionInfo(comptime Expr: type) type {
    const ops_result = collectOps(Expr);

    return struct {
        /// Number of operations that will be fused
        pub const op_count: usize = ops_result.len;
        /// The operations in the chain, in execution order (fixed-size array)
        pub const operations: [op_count]OpTag = ops_result.ops[0..op_count].*;
        /// Number of distinct input tensors
        pub const input_count: usize = countInputs(Expr);
        /// Output shape
        pub const output_shape = Expr.shape;
        /// Element type
        pub const T = Expr.ElementType;
    };
}

/// Collect operations from expression tree.
fn collectOps(comptime E: type) struct { ops: [MAX_CHAIN_OPS]OpTag, len: usize } {
    var ops_arr: [MAX_CHAIN_OPS]OpTag = undefined;
    var len: usize = 0;
    collectOpsRecursive(E, &ops_arr, &len);
    return .{ .ops = ops_arr, .len = len };
}

fn collectOpsRecursive(comptime E: type, arr: *[MAX_CHAIN_OPS]OpTag, len: *usize) void {
    switch (E.kind) {
        .tensor, .constant => {},
        .unary => {
            collectOpsRecursive(E.InputType, arr, len);
            if (len.* < MAX_CHAIN_OPS) {
                arr[len.*] = E.operation;
                len.* += 1;
            }
        },
        .binary => {
            collectOpsRecursive(E.LhsType, arr, len);
            collectOpsRecursive(E.RhsType, arr, len);
            if (len.* < MAX_CHAIN_OPS) {
                arr[len.*] = E.operation;
                len.* += 1;
            }
        },
        else => {},
    }
}

fn countInputs(comptime E: type) usize {
    return switch (E.kind) {
        .tensor => 1,
        .constant => 0,
        .unary => countInputs(E.InputType),
        .binary => countInputs(E.LhsType) + countInputs(E.RhsType),
        else => 0,
    };
}

/// Describes a matched matmul epilogue fusion pattern.
pub fn MatmulEpilogueInfo(comptime Expr: type) type {
    const epilogue_result = collectEpilogue(Expr);

    return struct {
        /// The base matmul expression type
        pub const MatmulType: ?type = findMatmul(Expr);
        /// Number of epilogue operations
        pub const epilogue_count: usize = epilogue_result.len;
        /// Epilogue operations (bias add, activations) in order
        pub const epilogue_ops: [epilogue_count]OpTag = epilogue_result.ops[0..epilogue_count].*;
        /// Whether there's a bias add
        pub const has_bias: bool = hasBiasAdd(epilogue_result);
        /// Whether there's an activation function
        pub const has_activation: bool = hasActivation(epilogue_result);
        /// Output shape
        pub const output_shape = Expr.shape;
        /// Element type
        pub const T = Expr.ElementType;
    };
}

fn findMatmul(comptime E: type) ?type {
    return switch (E.kind) {
        .matmul => E,
        .unary => findMatmul(E.InputType),
        .binary => blk: {
            if (E.LhsType.kind == .matmul) break :blk E.LhsType;
            if (E.RhsType.kind == .matmul) break :blk E.RhsType;
            if (findMatmul(E.LhsType)) |m| break :blk m;
            if (findMatmul(E.RhsType)) |m| break :blk m;
            break :blk null;
        },
        else => null,
    };
}

fn collectEpilogue(comptime E: type) struct { ops: [8]OpTag, len: usize } {
    var ops_arr: [8]OpTag = undefined;
    var len: usize = 0;
    collectEpilogueRecursive(E, &ops_arr, &len);
    return .{ .ops = ops_arr, .len = len };
}

fn collectEpilogueRecursive(comptime E: type, arr: *[8]OpTag, len: *usize) void {
    switch (E.kind) {
        .matmul => {},
        .unary => {
            collectEpilogueRecursive(E.InputType, arr, len);
            if (len.* < 8) {
                arr[len.*] = E.operation;
                len.* += 1;
            }
        },
        .binary => {
            if (E.LhsType.kind == .matmul or hasMatmulType(E.LhsType)) {
                collectEpilogueRecursive(E.LhsType, arr, len);
            } else if (E.RhsType.kind == .matmul or hasMatmulType(E.RhsType)) {
                collectEpilogueRecursive(E.RhsType, arr, len);
            }
            if (len.* < 8) {
                arr[len.*] = E.operation;
                len.* += 1;
            }
        },
        else => {},
    }
}

fn hasMatmulType(comptime E: type) bool {
    return findMatmul(E) != null;
}

fn hasBiasAdd(epilogue_result: anytype) bool {
    for (epilogue_result.ops[0..epilogue_result.len]) |op| {
        if (op == .add) return true;
    }
    return false;
}

fn hasActivation(epilogue_result: anytype) bool {
    for (epilogue_result.ops[0..epilogue_result.len]) |op| {
        if (op.isActivation()) return true;
    }
    return false;
}

/// Describes a matched reduction fusion pattern.
pub fn ReduceFusionInfo(comptime Expr: type) type {
    const pre_result = collectPreOps(Expr.InputType);

    return struct {
        /// The reduction operation
        pub const reduce_op: OpTag = Expr.operation;
        /// Reduction axes
        pub const axes = Expr.reduction_axes;
        /// Whether keepdims is set
        pub const keep_dims: bool = Expr.keep_dims;
        /// Number of pre-ops
        pub const pre_op_count: usize = pre_result.len;
        /// Pre-reduction elementwise operations
        pub const pre_ops: [pre_op_count]OpTag = pre_result.ops[0..pre_op_count].*;
        /// Input expression type
        pub const InputType = Expr.InputType;
        /// Output shape
        pub const output_shape = Expr.shape;
        /// Element type
        pub const T = Expr.ElementType;
    };
}

fn collectPreOps(comptime E: type) struct { ops: [16]OpTag, len: usize } {
    var ops_arr: [16]OpTag = undefined;
    var len: usize = 0;
    collectPreOpsRecursive(E, &ops_arr, &len);
    return .{ .ops = ops_arr, .len = len };
}

fn collectPreOpsRecursive(comptime E: type, arr: *[16]OpTag, len: *usize) void {
    switch (E.kind) {
        .tensor, .constant => {},
        .unary => {
            collectPreOpsRecursive(E.InputType, arr, len);
            if (len.* < 16) {
                arr[len.*] = E.operation;
                len.* += 1;
            }
        },
        .binary => {
            collectPreOpsRecursive(E.LhsType, arr, len);
            collectPreOpsRecursive(E.RhsType, arr, len);
            if (len.* < 16) {
                arr[len.*] = E.operation;
                len.* += 1;
            }
        },
        else => {},
    }
}

// ============================================================================
// Tests
// ============================================================================

test "elementwise fusion info" {
    const tensor_mod = @import("../core/tensor.zig");
    const Tensor = tensor_mod.Tensor;
    const Vec = Tensor(f32, .{4});

    // relu(exp(x))
    const ExpExpr = ops.UnaryExpr(.exp, Vec);
    const ReluExpr = ops.UnaryExpr(.relu, ExpExpr);

    const info = ElementwiseFusionInfo(ReluExpr);
    try std.testing.expectEqual(@as(usize, 2), info.op_count);
    try std.testing.expectEqual(@as(usize, 1), info.input_count);
    try std.testing.expectEqual(OpTag.exp, info.operations[0]);
    try std.testing.expectEqual(OpTag.relu, info.operations[1]);
}

test "matmul epilogue info" {
    const tensor_mod = @import("../core/tensor.zig");
    const Tensor = tensor_mod.Tensor;

    const A = Tensor(f32, .{ 2, 3 });
    const B = Tensor(f32, .{ 3, 4 });
    const Bias = Tensor(f32, .{4});

    const MatmulExpr = ops.MatmulExpr(A, B);
    const AddExpr = ops.BinaryExpr(.add, MatmulExpr, Bias);
    const ReluExpr = ops.UnaryExpr(.relu, AddExpr);

    const info = MatmulEpilogueInfo(ReluExpr);
    try std.testing.expect(info.MatmulType != null);
    try std.testing.expect(info.has_bias);
    try std.testing.expect(info.has_activation);
    try std.testing.expectEqual(@as(usize, 2), info.epilogue_count);
}

test "reduce fusion info" {
    const tensor_mod = @import("../core/tensor.zig");
    const Tensor = tensor_mod.Tensor;
    const Vec = Tensor(f32, .{4});

    // sum(exp(x))
    const ExpExpr = ops.UnaryExpr(.exp, Vec);
    const SumExpr = ops.ReduceExpr(.sum, ExpExpr, .{}, false);

    const info = ReduceFusionInfo(SumExpr);
    try std.testing.expectEqual(OpTag.sum, info.reduce_op);
    try std.testing.expectEqual(@as(usize, 1), info.pre_op_count);
    try std.testing.expectEqual(OpTag.exp, info.pre_ops[0]);
}
