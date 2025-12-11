//! CPU Executor for evaluating expression graphs.
//!
//! Walks the comptime expression graph and executes operations,
//! allocating intermediate tensors as needed.

const std = @import("std");
const core = struct {
    const tensor = @import("../../core/tensor.zig");
    const shape = @import("../../core/shape.zig");
    const strides = @import("../../core/strides.zig");
};
const ops = @import("../../ops/expr.zig");
const kernels = struct {
    const elementwise = @import("kernels/elementwise.zig");
    const matmul = @import("kernels/matmul.zig");
    const reduce = @import("kernels/reduce.zig");
};

const Tensor = core.tensor.Tensor;
const OpTag = ops.OpTag;
const NodeKind = ops.NodeKind;

/// Evaluate an expression and return a concrete tensor.
/// This is the main entry point for executing the expression graph.
pub fn eval(
    comptime Expr: type,
    expr: Expr,
    allocator: std.mem.Allocator,
) !Tensor(Expr.ElementType, Expr.shape) {
    const T = Expr.ElementType;
    const ResultTensor = Tensor(T, Expr.shape);

    // Allocate output tensor
    var result = try ResultTensor.init(allocator);
    errdefer result.deinit();

    // Evaluate expression into output buffer
    try evalInto(Expr, expr, result.slice(), allocator);

    return result;
}

/// Evaluate an expression into an existing output buffer.
/// This allows reuse of allocated memory.
pub fn evalInto(
    comptime Expr: type,
    expr: Expr,
    output: []Expr.ElementType,
    allocator: std.mem.Allocator,
) !void {
    // Dispatch based on expression kind
    switch (Expr.kind) {
        .tensor => {
            // Leaf tensor - copy data
            @memcpy(output, expr.constSlice());
        },
        .constant => {
            // Scalar constant - broadcast to output
            @memset(output, expr.value);
        },
        .unary => {
            try evalUnary(Expr, expr, output, allocator);
        },
        .binary => {
            try evalBinary(Expr, expr, output, allocator);
        },
        .matmul => {
            try evalMatmul(Expr, expr, output, allocator);
        },
        .reduce => {
            try evalReduce(Expr, expr, output, allocator);
        },
        else => {
            @compileError("Unsupported expression kind: " ++ @tagName(Expr.kind));
        },
    }
}

/// Evaluate a unary expression.
fn evalUnary(
    comptime Expr: type,
    expr: Expr,
    output: []Expr.ElementType,
    allocator: std.mem.Allocator,
) !void {
    const T = Expr.ElementType;
    const Input = Expr.InputType;

    // Get input data
    const input_data = try getInputData(Input, expr.input, allocator);
    defer if (input_data.allocated) allocator.free(input_data.data);

    // Apply operation
    kernels.elementwise.unaryOp(Expr.operation, T, input_data.data, output);
}

/// Evaluate a binary expression.
fn evalBinary(
    comptime Expr: type,
    expr: Expr,
    output: []Expr.ElementType,
    allocator: std.mem.Allocator,
) !void {
    const T = Expr.ElementType;
    const Lhs = Expr.LhsType;
    const Rhs = Expr.RhsType;

    // Get input data
    const lhs_data = try getInputData(Lhs, expr.lhs, allocator);
    defer if (lhs_data.allocated) allocator.free(lhs_data.data);

    const rhs_data = try getInputData(Rhs, expr.rhs, allocator);
    defer if (rhs_data.allocated) allocator.free(rhs_data.data);

    // Check if we need broadcasting
    const lhs_numel = computeNumel(Lhs.shape);
    const rhs_numel = computeNumel(Rhs.shape);
    const out_numel = computeNumel(Expr.shape);

    if (lhs_numel == rhs_numel and lhs_numel == out_numel) {
        // No broadcasting needed - direct operation
        kernels.elementwise.binaryOp(Expr.operation, T, lhs_data.data, rhs_data.data, output);
    } else if (lhs_numel == 1) {
        // LHS is scalar, broadcast
        kernels.elementwise.binaryOpScalarLhs(Expr.operation, T, lhs_data.data[0], rhs_data.data, output);
    } else if (rhs_numel == 1) {
        // RHS is scalar, broadcast
        kernels.elementwise.binaryOpScalarRhs(Expr.operation, T, lhs_data.data, rhs_data.data[0], output);
    } else {
        // General broadcasting - need to expand and iterate
        try evalBinaryBroadcast(Expr, T, lhs_data.data, rhs_data.data, output);
    }
}

/// Evaluate binary operation with general broadcasting.
fn evalBinaryBroadcast(
    comptime Expr: type,
    comptime T: type,
    lhs: []const T,
    rhs: []const T,
    output: []T,
) !void {
    const Lhs = Expr.LhsType;
    const Rhs = Expr.RhsType;

    const lhs_strides = computeStrides(Lhs.ndim, Lhs.shape);
    const rhs_strides = computeStrides(Rhs.ndim, Rhs.shape);

    // Compute broadcast strides
    const lhs_bcast = core.strides.broadcastStrides(Lhs.ndim, Expr.ndim, Lhs.shape, lhs_strides);
    const rhs_bcast = core.strides.broadcastStrides(Rhs.ndim, Expr.ndim, Rhs.shape, rhs_strides);

    // Iterate over output
    const out_numel = computeNumel(Expr.shape);
    for (0..out_numel) |flat_idx| {
        // Convert flat index to multi-dimensional
        var idx: [Expr.ndim]usize = undefined;
        var remaining = flat_idx;
        for (0..Expr.ndim) |i| {
            const dim_idx = Expr.ndim - 1 - i;
            idx[dim_idx] = remaining % Expr.shape[dim_idx];
            remaining /= Expr.shape[dim_idx];
        }

        // Compute offsets using broadcast strides
        var lhs_offset: usize = 0;
        var rhs_offset: usize = 0;
        for (0..Expr.ndim) |i| {
            lhs_offset += idx[i] * lhs_bcast[i];
            rhs_offset += idx[i] * rhs_bcast[i];
        }

        output[flat_idx] = kernels.elementwise.applyBinaryScalar(
            Expr.operation,
            T,
            lhs[lhs_offset],
            rhs[rhs_offset],
        );
    }
}

/// Evaluate a matmul expression.
fn evalMatmul(
    comptime Expr: type,
    expr: Expr,
    output: []Expr.ElementType,
    allocator: std.mem.Allocator,
) !void {
    const T = Expr.ElementType;
    const Lhs = Expr.LhsType;
    const Rhs = Expr.RhsType;

    // Get input data
    const lhs_data = try getInputData(Lhs, expr.lhs, allocator);
    defer if (lhs_data.allocated) allocator.free(lhs_data.data);

    const rhs_data = try getInputData(Rhs, expr.rhs, allocator);
    defer if (rhs_data.allocated) allocator.free(rhs_data.data);

    // Determine dimensions based on ranks
    if (Lhs.ndim == 1 and Rhs.ndim == 1) {
        // Dot product: [K] @ [K] -> scalar
        output[0] = kernels.matmul.dotProduct(T, lhs_data.data, rhs_data.data);
    } else if (Lhs.ndim == 1 and Rhs.ndim == 2) {
        // Vector-matrix: [K] @ [K, N] -> [N]
        kernels.matmul.vecMatmul(T, lhs_data.data, rhs_data.data, output, Rhs.shape[0], Rhs.shape[1]);
    } else if (Lhs.ndim == 2 and Rhs.ndim == 1) {
        // Matrix-vector: [M, K] @ [K] -> [M]
        kernels.matmul.matVecmul(T, lhs_data.data, rhs_data.data, output, Lhs.shape[0], Lhs.shape[1]);
    } else if (Lhs.ndim == 2 and Rhs.ndim == 2) {
        // Matrix-matrix: [M, K] @ [K, N] -> [M, N]
        kernels.matmul.matmulTiled(T, lhs_data.data, rhs_data.data, output, Lhs.shape[0], Lhs.shape[1], Rhs.shape[1]);
    } else {
        @compileError("Unsupported matmul dimensions");
    }
}

/// Evaluate a reduce expression.
fn evalReduce(
    comptime Expr: type,
    expr: Expr,
    output: []Expr.ElementType,
    allocator: std.mem.Allocator,
) !void {
    const T = Expr.ElementType;
    const Input = Expr.InputType;

    // Get input data
    const input_data = try getInputData(Input, expr.input, allocator);
    defer if (input_data.allocated) allocator.free(input_data.data);

    // Check if it's a full reduction or axis reduction
    const axes = Expr.reduction_axes;
    if (@typeInfo(@TypeOf(axes)).@"struct".fields.len == 0) {
        // Full reduction - reduce all elements
        output[0] = kernels.reduce.reduceAll(Expr.operation, T, input_data.data);
    } else {
        // Axis reduction
        // For simplicity, only handle single axis for now
        const axis = axes[0];
        kernels.reduce.reduceAxis(
            Expr.operation,
            T,
            Input.ndim,
            input_data.data,
            output,
            Input.shape,
            axis,
            Expr.keep_dims,
        );
    }
}

/// Result of getting input data - tracks whether we allocated.
fn InputData(comptime T: type) type {
    return struct {
        data: []const T,
        allocated: bool,
    };
}

/// Get the data for an input expression, evaluating if necessary.
fn getInputData(
    comptime Input: type,
    input: Input,
    allocator: std.mem.Allocator,
) !InputData(ops.ElementTypeOf(Input)) {
    const T = ops.ElementTypeOf(Input);

    switch (Input.kind) {
        .tensor => {
            // Leaf tensor - just return the data
            return .{ .data = input.constSlice(), .allocated = false };
        },
        .constant => {
            // Scalar constant - allocate and fill
            const data = try allocator.alloc(T, 1);
            data[0] = input.value;
            return .{ .data = data, .allocated = true };
        },
        else => {
            // Need to evaluate the expression
            const numel = computeNumel(Input.shape);
            const data = try allocator.alloc(T, numel);
            errdefer allocator.free(data);
            try evalInto(Input, input, data, allocator);
            return .{ .data = data, .allocated = true };
        },
    }
}

/// Compute number of elements from shape.
fn computeNumel(comptime shape: anytype) usize {
    if (shape.len == 0) return 1;
    var result: usize = 1;
    for (shape) |d| result *= d;
    return result;
}

/// Compute contiguous strides from shape.
fn computeStrides(comptime ndim: usize, comptime shape: [ndim]usize) [ndim]usize {
    if (ndim == 0) return .{};
    var strides: [ndim]usize = undefined;
    var stride: usize = 1;
    for (0..ndim) |i| {
        const idx = ndim - 1 - i;
        strides[idx] = stride;
        stride *= shape[idx];
    }
    return strides;
}

// ============================================================================
// Tests
// ============================================================================

test "eval tensor passthrough" {
    const Vec = Tensor(f32, .{4});
    var input = try Vec.fromSlice(std.testing.allocator, &[_]f32{ 1, 2, 3, 4 });
    defer input.deinit();

    var result = try eval(Vec, input, std.testing.allocator);
    defer result.deinit();

    try std.testing.expectEqualSlices(f32, input.slice(), result.slice());
}

test "eval unary relu" {
    const Vec = Tensor(f32, .{4});
    var input = try Vec.fromSlice(std.testing.allocator, &[_]f32{ -2, -1, 1, 2 });
    defer input.deinit();

    const ReluExpr = ops.UnaryExpr(.relu, Vec);
    const expr = ReluExpr.init(input);

    var result = try eval(ReluExpr, expr, std.testing.allocator);
    defer result.deinit();

    const expected = [_]f32{ 0, 0, 1, 2 };
    for (result.slice(), expected) |got, exp| {
        try std.testing.expectApproxEqAbs(exp, got, 1e-6);
    }
}

test "eval binary add" {
    const Vec = Tensor(f32, .{4});
    var a = try Vec.fromSlice(std.testing.allocator, &[_]f32{ 1, 2, 3, 4 });
    defer a.deinit();
    var b = try Vec.fromSlice(std.testing.allocator, &[_]f32{ 10, 20, 30, 40 });
    defer b.deinit();

    const AddExpr = ops.BinaryExpr(.add, Vec, Vec);
    const expr = AddExpr.init(a, b);

    var result = try eval(AddExpr, expr, std.testing.allocator);
    defer result.deinit();

    const expected = [_]f32{ 11, 22, 33, 44 };
    for (result.slice(), expected) |got, exp| {
        try std.testing.expectEqual(exp, got);
    }
}

test "eval chained expression" {
    const Vec = Tensor(f32, .{4});
    var a = try Vec.fromSlice(std.testing.allocator, &[_]f32{ -1, 2, -3, 4 });
    defer a.deinit();
    var b = try Vec.fromSlice(std.testing.allocator, &[_]f32{ 1, 1, 1, 1 });
    defer b.deinit();

    // expr = relu(a + b) = relu([-1+1, 2+1, -3+1, 4+1]) = relu([0, 3, -2, 5]) = [0, 3, 0, 5]
    const AddExpr = ops.BinaryExpr(.add, Vec, Vec);
    const ReluExpr = ops.UnaryExpr(.relu, AddExpr);

    const add_expr = AddExpr.init(a, b);
    const relu_expr = ReluExpr.init(add_expr);

    var result = try eval(ReluExpr, relu_expr, std.testing.allocator);
    defer result.deinit();

    const expected = [_]f32{ 0, 3, 0, 5 };
    for (result.slice(), expected) |got, exp| {
        try std.testing.expectEqual(exp, got);
    }
}

test "eval matmul 2x2" {
    const Mat = Tensor(f32, .{ 2, 2 });
    var a = try Mat.fromSlice(std.testing.allocator, &[_]f32{ 1, 2, 3, 4 });
    defer a.deinit();
    var b = try Mat.fromSlice(std.testing.allocator, &[_]f32{ 5, 6, 7, 8 });
    defer b.deinit();

    const MatmulExpr = ops.MatmulExpr(Mat, Mat);
    const expr = MatmulExpr.init(a, b);

    var result = try eval(MatmulExpr, expr, std.testing.allocator);
    defer result.deinit();

    // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
    const expected = [_]f32{ 19, 22, 43, 50 };
    for (result.slice(), expected) |got, exp| {
        try std.testing.expectApproxEqAbs(exp, got, 1e-5);
    }
}

test "eval reduce sum" {
    const Vec = Tensor(f32, .{4});
    var input = try Vec.fromSlice(std.testing.allocator, &[_]f32{ 1, 2, 3, 4 });
    defer input.deinit();

    const SumExpr = ops.ReduceExpr(.sum, Vec, .{}, false);
    const expr = SumExpr.init(input);

    var result = try eval(SumExpr, expr, std.testing.allocator);
    defer result.deinit();

    try std.testing.expectApproxEqAbs(@as(f32, 10), result.slice()[0], 1e-6);
}

test "eval full pipeline: matmul + bias + relu" {
    // Common neural network pattern
    const A = Tensor(f32, .{ 2, 3 });
    const B = Tensor(f32, .{ 3, 2 });
    const Bias = Tensor(f32, .{2});

    var a = try A.fromSlice(std.testing.allocator, &[_]f32{ 1, 2, 3, 4, 5, 6 });
    defer a.deinit();
    var b = try B.fromSlice(std.testing.allocator, &[_]f32{ 1, 2, 3, 4, 5, 6 });
    defer b.deinit();
    var bias = try Bias.fromSlice(std.testing.allocator, &[_]f32{ -100, -100 });
    defer bias.deinit();

    // Compute: relu(A @ B + bias)
    const MatmulExpr = ops.MatmulExpr(A, B);
    const AddExpr = ops.BinaryExpr(.add, MatmulExpr, Bias);
    const ReluExpr = ops.UnaryExpr(.relu, AddExpr);

    const mm = MatmulExpr.init(a, b);
    const add_bias = AddExpr.init(mm, bias);
    const activated = ReluExpr.init(add_bias);

    var result = try eval(ReluExpr, activated, std.testing.allocator);
    defer result.deinit();

    // A @ B = [[22, 28], [49, 64]]
    // + bias = [[-78, -72], [-51, -36]]
    // relu = [[0, 0], [0, 0]]
    for (result.slice()) |v| {
        try std.testing.expectEqual(@as(f32, 0), v);
    }
}
