//! Comptime fusion analysis for expression graphs.
//!
//! Analyzes expression graphs at compile time to identify fusion opportunities.
//! Returns a FusionPlan that describes how to efficiently execute the expression.

const std = @import("std");
const ops = @import("../ops/expr.zig");

const OpTag = ops.OpTag;
const NodeKind = ops.NodeKind;

/// Maximum number of operations in an elementwise chain.
pub const MAX_ELEMENTWISE_OPS = 32;

/// Pattern types that can be fused.
pub const FusionPattern = enum {
    /// Single operation, no fusion possible
    single,
    /// Chain of elementwise operations
    elementwise_chain,
    /// Matmul with fused epilogue (bias, activation)
    matmul_epilogue,
    /// Elementwise operations followed by reduction
    reduce_epilogue,
    /// Full reduction with fused elementwise
    fused_reduce,
};

/// A segment of operations to be executed together.
pub fn FusionSegment(comptime Expr: type) type {
    return struct {
        /// The pattern type for this segment
        pub const pattern: FusionPattern = analyzePattern(Expr);
        /// The expression type this segment represents
        pub const ExprType = Expr;
        /// Number of operations that are fused
        pub const fused_op_count: usize = countFusedOps(Expr);
        /// Whether this segment can be executed in-place
        pub const can_inplace: bool = canExecuteInplace(Expr);
    };
}

/// Analyze an expression graph and produce a fusion plan.
pub fn FusionPlan(comptime Expr: type) type {
    const chain_result = getElementwiseChainInternal(Expr);

    return struct {
        pub const RootExpr = Expr;
        pub const pattern: FusionPattern = analyzePattern(Expr);
        pub const fused_op_count: usize = countFusedOps(Expr);

        /// Number of operations in the elementwise chain.
        pub const elementwise_chain_len: usize = chain_result.len;

        /// Get the elementwise operations array (fixed size).
        pub const elementwise_chain: [chain_result.len]OpTag = chain_result.ops[0..chain_result.len].*;

        /// Check if this plan represents a matmul with epilogue.
        pub fn isMatmulEpilogue() bool {
            return pattern == .matmul_epilogue;
        }

        /// Get the matmul base expression type if this is a matmul epilogue.
        pub fn getMatmulBase() ?type {
            if (pattern != .matmul_epilogue) {
                return null;
            }
            return findMatmulBase(Expr);
        }

        /// Number of epilogue operations.
        pub const epilogue_len: usize = getEpilogueInternal(Expr).len;

        /// Get epilogue operations for matmul (bias add, activations).
        pub const matmul_epilogue: [epilogue_len]OpTag = getEpilogueInternal(Expr).ops[0..epilogue_len].*;
    };
}

/// Internal helper to get elementwise chain.
fn getElementwiseChainInternal(comptime Expr: type) struct { ops: [MAX_ELEMENTWISE_OPS]OpTag, len: usize } {
    const p = analyzePattern(Expr);
    if (p != .elementwise_chain and p != .single) {
        return .{ .ops = undefined, .len = 0 };
    }

    var chain: [MAX_ELEMENTWISE_OPS]OpTag = undefined;
    var len: usize = 0;
    collectElementwiseOps(Expr, &chain, &len);
    return .{ .ops = chain, .len = len };
}

/// Internal helper to get epilogue operations.
fn getEpilogueInternal(comptime Expr: type) struct { ops: [8]OpTag, len: usize } {
    const p = analyzePattern(Expr);
    if (p != .matmul_epilogue) {
        return .{ .ops = undefined, .len = 0 };
    }

    var epilogue: [8]OpTag = undefined;
    var len: usize = 0;
    collectEpilogueOps(Expr, &epilogue, &len);
    return .{ .ops = epilogue, .len = len };
}

/// Analyze the fusion pattern for an expression.
fn analyzePattern(comptime Expr: type) FusionPattern {
    const kind = Expr.kind;

    switch (kind) {
        .tensor, .constant => return .single,

        .unary => {
            // Check if there's a matmul somewhere in the input tree
            if (hasMatmulInChain(Expr.InputType)) {
                return .matmul_epilogue;
            }
            const input_kind = Expr.InputType.kind;
            if (input_kind == .unary or input_kind == .binary) {
                if (isElementwiseChain(Expr)) {
                    return .elementwise_chain;
                }
            }
            return .single;
        },

        .binary => {
            // Check if there's a matmul somewhere in the inputs
            if (hasMatmulInChain(Expr.LhsType) or hasMatmulInChain(Expr.RhsType)) {
                return .matmul_epilogue;
            }

            if (isElementwiseChain(Expr)) {
                return .elementwise_chain;
            }

            return .single;
        },

        .matmul => return .single,

        .reduce => {
            const input_kind = Expr.InputType.kind;
            if (input_kind == .unary or input_kind == .binary) {
                if (isElementwiseChain(Expr.InputType)) {
                    return .reduce_epilogue;
                }
            }
            return .single;
        },

        else => return .single,
    }
}

/// Check if there's a matmul operation in the expression chain.
fn hasMatmulInChain(comptime Expr: type) bool {
    const kind = Expr.kind;
    return switch (kind) {
        .matmul => true,
        .unary => hasMatmulInChain(Expr.InputType),
        .binary => hasMatmulInChain(Expr.LhsType) or hasMatmulInChain(Expr.RhsType),
        else => false,
    };
}

/// Check if an expression is a pure elementwise chain.
fn isElementwiseChain(comptime Expr: type) bool {
    const kind = Expr.kind;

    switch (kind) {
        .tensor, .constant => return true,
        .unary => return isElementwiseChain(Expr.InputType),
        .binary => {
            return isElementwiseOrLeaf(Expr.LhsType) and isElementwiseOrLeaf(Expr.RhsType);
        },
        else => return false,
    }
}

/// Check if expression is elementwise operation or leaf tensor/constant.
fn isElementwiseOrLeaf(comptime Expr: type) bool {
    const kind = Expr.kind;
    return switch (kind) {
        .tensor, .constant => true,
        .unary, .binary => true,
        else => false,
    };
}

/// Count total number of operations that can be fused.
fn countFusedOps(comptime Expr: type) usize {
    const kind = Expr.kind;

    switch (kind) {
        .tensor, .constant => return 0,
        .unary => return 1 + countFusedOps(Expr.InputType),
        .binary => {
            const lhs_count = if (isElementwiseOrLeaf(Expr.LhsType)) countFusedOps(Expr.LhsType) else 0;
            const rhs_count = if (isElementwiseOrLeaf(Expr.RhsType)) countFusedOps(Expr.RhsType) else 0;
            return 1 + lhs_count + rhs_count;
        },
        .matmul => return 1,
        .reduce => return 1 + countFusedOps(Expr.InputType),
        else => return 1,
    }
}

/// Check if expression can be executed in-place (output reuses input buffer).
fn canExecuteInplace(comptime Expr: type) bool {
    const kind = Expr.kind;

    switch (kind) {
        .tensor, .constant => return false,
        .unary => {
            return std.mem.eql(usize, &Expr.shape, &Expr.InputType.shape);
        },
        .binary => {
            const lhs_match = std.mem.eql(usize, &Expr.shape, &Expr.LhsType.shape);
            const rhs_match = std.mem.eql(usize, &Expr.shape, &Expr.RhsType.shape);
            return lhs_match or rhs_match;
        },
        else => return false,
    }
}

/// Collect elementwise operations in chain order.
fn collectElementwiseOps(comptime Expr: type, ops_out: *[MAX_ELEMENTWISE_OPS]OpTag, len: *usize) void {
    const kind = Expr.kind;

    switch (kind) {
        .tensor, .constant => {},
        .unary => {
            collectElementwiseOps(Expr.InputType, ops_out, len);
            if (len.* < MAX_ELEMENTWISE_OPS) {
                ops_out[len.*] = Expr.operation;
                len.* += 1;
            }
        },
        .binary => {
            if (len.* < MAX_ELEMENTWISE_OPS) {
                ops_out[len.*] = Expr.operation;
                len.* += 1;
            }
        },
        else => {},
    }
}

/// Find the matmul base expression in a matmul epilogue pattern.
fn findMatmulBase(comptime Expr: type) ?type {
    const kind = Expr.kind;

    switch (kind) {
        .matmul => return Expr,
        .unary => return findMatmulBase(Expr.InputType),
        .binary => {
            if (Expr.LhsType.kind == .matmul) return Expr.LhsType;
            if (Expr.RhsType.kind == .matmul) return Expr.RhsType;
            if (findMatmulBase(Expr.LhsType)) |t| return t;
            if (findMatmulBase(Expr.RhsType)) |t| return t;
            return null;
        },
        else => return null,
    }
}

/// Collect epilogue operations after matmul.
fn collectEpilogueOps(comptime Expr: type, ops_out: *[8]OpTag, len: *usize) void {
    const kind = Expr.kind;

    switch (kind) {
        .matmul => {}, // Base case - stop here
        .unary => {
            // First recurse into input
            collectEpilogueOps(Expr.InputType, ops_out, len);
            // Then add our operation
            if (len.* < 8) {
                ops_out[len.*] = Expr.operation;
                len.* += 1;
            }
        },
        .binary => {
            // Recurse into the side that has the matmul
            if (hasMatmulInChain(Expr.LhsType)) {
                collectEpilogueOps(Expr.LhsType, ops_out, len);
            } else if (hasMatmulInChain(Expr.RhsType)) {
                collectEpilogueOps(Expr.RhsType, ops_out, len);
            }
            // Then add our operation
            if (len.* < 8) {
                ops_out[len.*] = Expr.operation;
                len.* += 1;
            }
        },
        else => {},
    }
}

// ============================================================================
// Tests
// ============================================================================

test "analyze single tensor" {
    const tensor_mod = @import("../core/tensor.zig");
    const Tensor = tensor_mod.Tensor;
    const Vec = Tensor(f32, .{4});

    const plan = FusionPlan(Vec);
    try std.testing.expectEqual(FusionPattern.single, plan.pattern);
    try std.testing.expectEqual(@as(usize, 0), plan.fused_op_count);
}

test "analyze unary expression" {
    const tensor_mod = @import("../core/tensor.zig");
    const Tensor = tensor_mod.Tensor;
    const Vec = Tensor(f32, .{4});

    const ReluExpr = ops.UnaryExpr(.relu, Vec);
    const plan = FusionPlan(ReluExpr);
    try std.testing.expectEqual(FusionPattern.single, plan.pattern);
    try std.testing.expectEqual(@as(usize, 1), plan.fused_op_count);
}

test "analyze elementwise chain" {
    const tensor_mod = @import("../core/tensor.zig");
    const Tensor = tensor_mod.Tensor;
    const Vec = Tensor(f32, .{4});

    // relu(exp(x))
    const ExpExpr = ops.UnaryExpr(.exp, Vec);
    const ReluExpr = ops.UnaryExpr(.relu, ExpExpr);

    const plan = FusionPlan(ReluExpr);
    try std.testing.expectEqual(FusionPattern.elementwise_chain, plan.pattern);
    try std.testing.expectEqual(@as(usize, 2), plan.fused_op_count);
    try std.testing.expectEqual(@as(usize, 2), plan.elementwise_chain_len);
    try std.testing.expectEqual(OpTag.exp, plan.elementwise_chain[0]);
    try std.testing.expectEqual(OpTag.relu, plan.elementwise_chain[1]);
}

test "analyze matmul epilogue" {
    const tensor_mod = @import("../core/tensor.zig");
    const Tensor = tensor_mod.Tensor;

    const A = Tensor(f32, .{ 2, 3 });
    const B = Tensor(f32, .{ 3, 4 });
    const Bias = Tensor(f32, .{4});

    // matmul + bias + relu
    const MatmulExpr = ops.MatmulExpr(A, B);
    const AddExpr = ops.BinaryExpr(.add, MatmulExpr, Bias);
    const ReluExpr = ops.UnaryExpr(.relu, AddExpr);

    const plan = FusionPlan(ReluExpr);
    try std.testing.expectEqual(FusionPattern.matmul_epilogue, plan.pattern);
    try std.testing.expect(plan.isMatmulEpilogue());
    try std.testing.expectEqual(@as(usize, 2), plan.epilogue_len);
    try std.testing.expectEqual(OpTag.add, plan.matmul_epilogue[0]);
    try std.testing.expectEqual(OpTag.relu, plan.matmul_epilogue[1]);
}

test "analyze binary expression" {
    const tensor_mod = @import("../core/tensor.zig");
    const Tensor = tensor_mod.Tensor;
    const Vec = Tensor(f32, .{4});

    // A binary elementwise operation is an elementwise chain (of length 1)
    const AddExpr = ops.BinaryExpr(.add, Vec, Vec);
    const plan = FusionPlan(AddExpr);
    try std.testing.expectEqual(FusionPattern.elementwise_chain, plan.pattern);
    try std.testing.expectEqual(@as(usize, 1), plan.fused_op_count);
}

test "can execute inplace" {
    const tensor_mod = @import("../core/tensor.zig");
    const Tensor = tensor_mod.Tensor;
    const Vec = Tensor(f32, .{4});

    const ReluExpr = ops.UnaryExpr(.relu, Vec);
    const segment = FusionSegment(ReluExpr);
    try std.testing.expect(segment.can_inplace);
}
