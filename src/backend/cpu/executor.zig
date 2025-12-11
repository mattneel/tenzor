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
    const softmax = @import("kernels/softmax.zig");
    const layernorm = @import("kernels/layernorm.zig");
    const gather = @import("kernels/gather.zig");
    const transpose = @import("kernels/transpose.zig");
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
        .softmax => {
            try evalSoftmax(Expr, expr, output, allocator);
        },
        .layernorm => {
            try evalLayerNorm(Expr, expr, output, allocator);
        },
        .gather => {
            try evalGather(Expr, expr, output, allocator);
        },
        .reshape => {
            try evalReshape(Expr, expr, output, allocator);
        },
        .transpose => {
            try evalTranspose(Expr, expr, output, allocator);
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
        // Batched matmul
        evalBatchedMatmul(T, Lhs, Rhs, Expr, lhs_data.data, rhs_data.data, output);
    }
}

/// Evaluate batched matmul with proper stride computation.
fn evalBatchedMatmul(
    comptime T: type,
    comptime Lhs: type,
    comptime Rhs: type,
    comptime Expr: type,
    lhs: []const T,
    rhs: []const T,
    output: []T,
) void {
    // Extract matrix dimensions (last 2 dims)
    const m = Lhs.shape[Lhs.ndim - 2];
    const k = Lhs.shape[Lhs.ndim - 1];
    const n = Rhs.shape[Rhs.ndim - 1];

    // Number of batch dimensions
    const lhs_batch_dims = Lhs.ndim - 2;
    const rhs_batch_dims = Rhs.ndim - 2;
    const out_batch_dims = Expr.ndim - 2;

    // Special case: 3D with same batch (most common: [B,M,K] @ [B,K,N])
    if (Lhs.ndim == 3 and Rhs.ndim == 3 and Lhs.shape[0] == Rhs.shape[0]) {
        kernels.matmul.batchedMatmul(T, lhs, rhs, output, Lhs.shape[0], m, k, n);
        return;
    }

    // Special case: 3D @ 2D broadcast ([B,M,K] @ [K,N] -> [B,M,N])
    if (Lhs.ndim == 3 and Rhs.ndim == 2) {
        kernels.matmul.batchedMatmulBroadcastB(T, lhs, rhs, output, Lhs.shape[0], m, k, n);
        return;
    }

    // General case: compute batch strides with broadcasting
    const batch_shape = Expr.shape[0..out_batch_dims].*;

    // Compute LHS batch strides
    var lhs_batch_strides: [out_batch_dims]usize = undefined;
    const lhs_mat_size = m * k;
    for (0..out_batch_dims) |i| {
        const lhs_dim_idx = if (i + lhs_batch_dims >= out_batch_dims)
            i + lhs_batch_dims - out_batch_dims
        else
            out_batch_dims; // out of bounds marker

        if (lhs_dim_idx < lhs_batch_dims and Lhs.shape[lhs_dim_idx] > 1) {
            // Compute stride: product of all following batch dims * matrix size
            var stride: usize = lhs_mat_size;
            for (lhs_dim_idx + 1..lhs_batch_dims) |j| {
                stride *= Lhs.shape[j];
            }
            lhs_batch_strides[i] = stride;
        } else {
            lhs_batch_strides[i] = 0; // broadcast
        }
    }

    // Compute RHS batch strides
    var rhs_batch_strides: [out_batch_dims]usize = undefined;
    const rhs_mat_size = k * n;
    for (0..out_batch_dims) |i| {
        const rhs_dim_idx = if (i + rhs_batch_dims >= out_batch_dims)
            i + rhs_batch_dims - out_batch_dims
        else
            out_batch_dims;

        if (rhs_dim_idx < rhs_batch_dims and Rhs.shape[rhs_dim_idx] > 1) {
            var stride: usize = rhs_mat_size;
            for (rhs_dim_idx + 1..rhs_batch_dims) |j| {
                stride *= Rhs.shape[j];
            }
            rhs_batch_strides[i] = stride;
        } else {
            rhs_batch_strides[i] = 0;
        }
    }

    kernels.matmul.batchedMatmulGeneral(
        T,
        out_batch_dims,
        lhs,
        rhs,
        output,
        batch_shape,
        lhs_batch_strides,
        rhs_batch_strides,
        m,
        k,
        n,
    );
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

/// Evaluate a softmax expression.
fn evalSoftmax(
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

    // Call softmax kernel with the axis from the expression
    kernels.softmax.softmaxAxis(
        T,
        input_data.data,
        output,
        Expr.ndim,
        Expr.shape,
        Expr.softmax_axis,
        null, // No mask for now
    );
}

/// Evaluate a layer normalization expression.
fn evalLayerNorm(
    comptime Expr: type,
    expr: Expr,
    output: []Expr.ElementType,
    allocator: std.mem.Allocator,
) !void {
    const T = Expr.ElementType;
    const Input = Expr.InputType;
    const Gamma = Expr.GammaType;
    const Beta = Expr.BetaType;

    // Get input data
    const input_data = try getInputData(Input, expr.input, allocator);
    defer if (input_data.allocated) allocator.free(input_data.data);

    const gamma_data = try getInputData(Gamma, expr.gamma, allocator);
    defer if (gamma_data.allocated) allocator.free(gamma_data.data);

    const beta_data = try getInputData(Beta, expr.beta, allocator);
    defer if (beta_data.allocated) allocator.free(beta_data.data);

    // Call layernorm kernel
    kernels.layernorm.layerNorm(
        T,
        input_data.data,
        gamma_data.data,
        beta_data.data,
        output,
        Expr.ndim,
        Expr.shape,
        Expr.norm_dims,
        kernels.layernorm.default_eps,
    );
}

/// Evaluate a gather/embedding lookup expression.
fn evalGather(
    comptime Expr: type,
    expr: Expr,
    output: []Expr.ElementType,
    allocator: std.mem.Allocator,
) !void {
    const T = Expr.ElementType;
    const IndexT = Expr.IndexType;
    const Table = Expr.TableType;
    const Indices = Expr.IndicesType;

    // Get table data (floats)
    const table_data = try getInputData(Table, expr.table, allocator);
    defer if (table_data.allocated) allocator.free(table_data.data);

    // Get indices data (integers) - need separate handling
    const indices_data = try getIndicesData(Indices, IndexT, expr.indices, allocator);
    defer if (indices_data.allocated) allocator.free(indices_data.data);

    // Call gather kernel
    kernels.gather.gather(
        T,
        IndexT,
        table_data.data,
        indices_data.data,
        output,
        Expr.vocab_size,
        Expr.embedding_dim,
    );
}

/// Get indices data (integer type), separate from getInputData which assumes float.
fn getIndicesData(
    comptime Indices: type,
    comptime IndexT: type,
    indices: Indices,
    allocator: std.mem.Allocator,
) !InputData(IndexT) {
    switch (Indices.kind) {
        .tensor => {
            return .{ .data = indices.constSlice(), .allocated = false };
        },
        else => {
            // Need to evaluate the expression
            const numel = computeNumel(Indices.shape);
            const data = try allocator.alloc(IndexT, numel);
            errdefer allocator.free(data);
            // Note: This assumes indices come from tensors, not computed expressions
            // For now, we only support tensor indices
            @compileError("Gather indices must be tensor type, not computed expression");
        },
    }
}

/// Evaluate a reshape expression.
/// Reshape is a no-op on the data - just a reinterpretation of shape.
fn evalReshape(
    comptime Expr: type,
    expr: Expr,
    output: []Expr.ElementType,
    allocator: std.mem.Allocator,
) !void {
    const Input = Expr.InputType;

    // Get input data
    const input_data = try getInputData(Input, expr.input, allocator);
    defer if (input_data.allocated) allocator.free(input_data.data);

    // Just copy (reshape doesn't change data layout for contiguous tensors)
    @memcpy(output, input_data.data);
}

/// Evaluate a transpose expression.
fn evalTranspose(
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

    // Check for optimized 4D attention transpose [B, S, H, D] -> [B, H, S, D]
    if (Expr.ndim == 4 and
        Expr.permutation[0] == 0 and
        Expr.permutation[1] == 2 and
        Expr.permutation[2] == 1 and
        Expr.permutation[3] == 3)
    {
        const in_shape = Input.shape;
        kernels.transpose.transpose4D_0213(
            T,
            input_data.data,
            output,
            in_shape[0], // B
            in_shape[1], // S
            in_shape[2], // H
            in_shape[3], // D
        );
        return;
    }

    // General case
    kernels.transpose.transpose(
        T,
        Expr.ndim,
        input_data.data,
        output,
        Input.shape,
        Expr.permutation,
    );
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

test "eval batched matmul 3D @ 3D" {
    // [2, 2, 3] @ [2, 3, 2] -> [2, 2, 2]
    const A = Tensor(f32, .{ 2, 2, 3 });
    const B = Tensor(f32, .{ 2, 3, 2 });

    // A[0] = [[1,2,3],[4,5,6]], A[1] = [[7,8,9],[10,11,12]]
    var a = try A.fromSlice(std.testing.allocator, &[_]f32{
        1, 2, 3, 4,  5,  6,
        7, 8, 9, 10, 11, 12,
    });
    defer a.deinit();

    // B[0] = [[1,0],[0,1],[1,1]], B[1] = [[1,1],[1,1],[1,1]]
    var b = try B.fromSlice(std.testing.allocator, &[_]f32{
        1, 0, 0, 1, 1, 1,
        1, 1, 1, 1, 1, 1,
    });
    defer b.deinit();

    const MatmulExpr = ops.MatmulExpr(A, B);
    const expr = MatmulExpr.init(a, b);

    var result = try eval(MatmulExpr, expr, std.testing.allocator);
    defer result.deinit();

    // Check result shape
    try std.testing.expectEqual(@as(usize, 3), MatmulExpr.ndim);
    try std.testing.expectEqual(@as(usize, 2), MatmulExpr.shape[0]);
    try std.testing.expectEqual(@as(usize, 2), MatmulExpr.shape[1]);
    try std.testing.expectEqual(@as(usize, 2), MatmulExpr.shape[2]);

    // A[0] @ B[0] = [[1+0+3, 0+2+3], [4+0+6, 0+5+6]] = [[4,5],[10,11]]
    try std.testing.expectApproxEqAbs(@as(f32, 4), result.slice()[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 5), result.slice()[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 10), result.slice()[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 11), result.slice()[3], 1e-5);

    // A[1] @ B[1] = [[7+8+9, 7+8+9], [10+11+12, 10+11+12]] = [[24,24],[33,33]]
    try std.testing.expectApproxEqAbs(@as(f32, 24), result.slice()[4], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 24), result.slice()[5], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 33), result.slice()[6], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 33), result.slice()[7], 1e-5);
}

test "eval batched matmul 3D @ 2D broadcast" {
    // [2, 3, 4] @ [4, 5] -> [2, 3, 5] (B broadcasts over batch)
    const A = Tensor(f32, .{ 2, 3, 4 });
    const B = Tensor(f32, .{ 4, 5 });

    var a = try A.zeros(std.testing.allocator);
    defer a.deinit();
    // Set some values: first batch first row = [1,1,1,1]
    a.data[0] = 1;
    a.data[1] = 1;
    a.data[2] = 1;
    a.data[3] = 1;

    // B = all ones
    var b = try B.ones(std.testing.allocator);
    defer b.deinit();

    const MatmulExpr = ops.MatmulExpr(A, B);
    const expr = MatmulExpr.init(a, b);

    var result = try eval(MatmulExpr, expr, std.testing.allocator);
    defer result.deinit();

    // Shape should be [2, 3, 5]
    try std.testing.expectEqual(@as(usize, 3), MatmulExpr.ndim);
    try std.testing.expectEqual(@as(usize, 2), MatmulExpr.shape[0]);
    try std.testing.expectEqual(@as(usize, 3), MatmulExpr.shape[1]);
    try std.testing.expectEqual(@as(usize, 5), MatmulExpr.shape[2]);

    // First row of first batch: [1,1,1,1] @ ones(4,5) = [4,4,4,4,4]
    for (0..5) |i| {
        try std.testing.expectApproxEqAbs(@as(f32, 4), result.slice()[i], 1e-5);
    }
}

test "eval batched matmul arctic pattern" {
    // Arctic attention: [B, S, D] @ [D, D] -> [B, S, D]
    // Small version: [2, 4, 8] @ [8, 8] -> [2, 4, 8]
    const Hidden = Tensor(f32, .{ 2, 4, 8 });
    const Weight = Tensor(f32, .{ 8, 8 });

    var hidden = try Hidden.ones(std.testing.allocator);
    defer hidden.deinit();

    // Weight = identity
    var weight = try Weight.zeros(std.testing.allocator);
    defer weight.deinit();
    for (0..8) |i| {
        weight.data[i * 8 + i] = 1.0;
    }

    const MatmulExpr = ops.MatmulExpr(Hidden, Weight);
    const expr = MatmulExpr.init(hidden, weight);

    var result = try eval(MatmulExpr, expr, std.testing.allocator);
    defer result.deinit();

    // Result should equal hidden (since weight is identity)
    try std.testing.expectEqual(@as(usize, 3), MatmulExpr.ndim);
    try std.testing.expectEqual(@as(usize, 2), MatmulExpr.shape[0]);
    try std.testing.expectEqual(@as(usize, 4), MatmulExpr.shape[1]);
    try std.testing.expectEqual(@as(usize, 8), MatmulExpr.shape[2]);

    for (hidden.slice(), result.slice()) |h, r| {
        try std.testing.expectApproxEqAbs(h, r, 1e-5);
    }
}

test "eval softmax 1D" {
    const Vec = Tensor(f32, .{4});
    var input = try Vec.fromSlice(std.testing.allocator, &[_]f32{ 1, 2, 3, 4 });
    defer input.deinit();

    const SoftmaxExpr = ops.SoftmaxExpr(Vec, -1);
    const expr = SoftmaxExpr.init(input);

    var result = try eval(SoftmaxExpr, expr, std.testing.allocator);
    defer result.deinit();

    // Should sum to 1
    var sum: f32 = 0;
    for (result.slice()) |v| sum += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-5);

    // Larger inputs should have higher probabilities
    try std.testing.expect(result.slice()[3] > result.slice()[2]);
    try std.testing.expect(result.slice()[2] > result.slice()[1]);
    try std.testing.expect(result.slice()[1] > result.slice()[0]);
}

test "eval softmax 2D last axis" {
    const Mat = Tensor(f32, .{ 2, 3 });
    var input = try Mat.fromSlice(std.testing.allocator, &[_]f32{
        1, 2, 3, // row 0
        0, 0, 0, // row 1 (uniform)
    });
    defer input.deinit();

    const SoftmaxExpr = ops.SoftmaxExpr(Mat, -1);
    const expr = SoftmaxExpr.init(input);

    var result = try eval(SoftmaxExpr, expr, std.testing.allocator);
    defer result.deinit();

    // Each row should sum to 1
    var row0_sum: f32 = 0;
    var row1_sum: f32 = 0;
    for (result.slice()[0..3]) |v| row0_sum += v;
    for (result.slice()[3..6]) |v| row1_sum += v;

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), row0_sum, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), row1_sum, 1e-5);

    // Row 1 should be uniform
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / 3.0), result.slice()[3], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / 3.0), result.slice()[4], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / 3.0), result.slice()[5], 1e-5);
}

test "eval softmax 4D attention pattern" {
    // [B=1, H=2, S=2, S=2] - typical attention shape
    const Attn = Tensor(f32, .{ 1, 2, 2, 2 });
    var input = try Attn.fromSlice(std.testing.allocator, &[_]f32{
        // Head 0
        1.0, 2.0, // row 0
        3.0, 4.0, // row 1
        // Head 1
        0.0, 0.0, // row 0 (uniform)
        1.0, -1.0, // row 1
    });
    defer input.deinit();

    const SoftmaxExpr = ops.SoftmaxExpr(Attn, -1);
    const expr = SoftmaxExpr.init(input);

    var result = try eval(SoftmaxExpr, expr, std.testing.allocator);
    defer result.deinit();

    // Check each attention row sums to 1
    for (0..4) |row| {
        var row_sum: f32 = 0;
        for (result.slice()[row * 2 ..][0..2]) |v| row_sum += v;
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), row_sum, 1e-5);
    }

    // Head 1, row 0 should be uniform (both 0)
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), result.slice()[4], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), result.slice()[5], 1e-5);
}

test "eval softmax chained after matmul" {
    // Common attention pattern: QK^T -> softmax
    const Q = Tensor(f32, .{ 2, 3 });
    const K = Tensor(f32, .{ 3, 2 });

    var q = try Q.fromSlice(std.testing.allocator, &[_]f32{ 1, 0, 0, 0, 1, 0 });
    defer q.deinit();
    var k = try K.fromSlice(std.testing.allocator, &[_]f32{ 1, 0, 0, 1, 0, 0 });
    defer k.deinit();

    // Compute softmax(Q @ K)
    const MatmulExpr = ops.MatmulExpr(Q, K);
    const SoftmaxExpr = ops.SoftmaxExpr(MatmulExpr, -1);

    const mm = MatmulExpr.init(q, k);
    const softmax_expr = mm.softmax(-1);

    var result = try eval(SoftmaxExpr, softmax_expr, std.testing.allocator);
    defer result.deinit();

    // Result should be [2, 2], each row summing to 1
    try std.testing.expectEqual(@as(usize, 2), SoftmaxExpr.ndim);
    try std.testing.expectEqual(@as(usize, 2), SoftmaxExpr.shape[0]);
    try std.testing.expectEqual(@as(usize, 2), SoftmaxExpr.shape[1]);

    var row0_sum: f32 = 0;
    var row1_sum: f32 = 0;
    row0_sum = result.slice()[0] + result.slice()[1];
    row1_sum = result.slice()[2] + result.slice()[3];

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), row0_sum, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), row1_sum, 1e-5);
}

test "eval layernorm 1D" {
    const Vec = Tensor(f32, .{4});
    var input = try Vec.fromSlice(std.testing.allocator, &[_]f32{ 1, 2, 3, 4 });
    defer input.deinit();
    var gamma = try Vec.fromSlice(std.testing.allocator, &[_]f32{ 1, 1, 1, 1 });
    defer gamma.deinit();
    var beta = try Vec.fromSlice(std.testing.allocator, &[_]f32{ 0, 0, 0, 0 });
    defer beta.deinit();

    const LNExpr = ops.LayerNormExpr(Vec, Vec, Vec, 1);
    const expr = LNExpr.init(input, gamma, beta);

    var result = try eval(LNExpr, expr, std.testing.allocator);
    defer result.deinit();

    // After normalization with gamma=1, beta=0: mean should be ~0
    var mean: f32 = 0;
    for (result.slice()) |v| mean += v;
    mean /= 4;
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), mean, 1e-5);
}

test "eval layernorm 2D transformer pattern" {
    // [B=1, S=2, D=4] -> normalize over D
    const Input = Tensor(f32, .{ 2, 4 });
    const Params = Tensor(f32, .{4});

    var input = try Input.fromSlice(std.testing.allocator, &[_]f32{
        1, 2, 3, 4, // token 0
        10, 20, 30, 40, // token 1 (scaled)
    });
    defer input.deinit();

    var gamma = try Params.fromSlice(std.testing.allocator, &[_]f32{ 1, 1, 1, 1 });
    defer gamma.deinit();
    var beta = try Params.fromSlice(std.testing.allocator, &[_]f32{ 0, 0, 0, 0 });
    defer beta.deinit();

    const LNExpr = ops.LayerNormExpr(Input, Params, Params, 1);
    const expr = LNExpr.init(input, gamma, beta);

    var result = try eval(LNExpr, expr, std.testing.allocator);
    defer result.deinit();

    // Each token should have mean ~0
    var mean0: f32 = 0;
    var mean1: f32 = 0;
    for (result.slice()[0..4]) |v| mean0 += v;
    for (result.slice()[4..8]) |v| mean1 += v;
    mean0 /= 4;
    mean1 /= 4;

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), mean0, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), mean1, 1e-5);

    // Normalized values should be the same (same relative pattern)
    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(result.slice()[i], result.slice()[4 + i], 1e-5);
    }
}

test "eval layernorm chained" {
    // Common pattern: layernorm -> matmul
    const Input = Tensor(f32, .{ 2, 4 });
    const Params = Tensor(f32, .{4});
    const Weight = Tensor(f32, .{ 4, 4 });

    var input = try Input.fromSlice(std.testing.allocator, &[_]f32{
        1, 2, 3, 4,
        5, 6, 7, 8,
    });
    defer input.deinit();

    var gamma = try Params.ones(std.testing.allocator);
    defer gamma.deinit();
    var beta = try Params.zeros(std.testing.allocator);
    defer beta.deinit();

    // Identity weight
    var weight = try Weight.zeros(std.testing.allocator);
    defer weight.deinit();
    for (0..4) |i| {
        weight.data[i * 4 + i] = 1.0;
    }

    // layernorm(input) @ weight
    const LNExpr = ops.LayerNormExpr(Input, Params, Params, 1);
    const MatmulExpr = ops.MatmulExpr(LNExpr, Weight);

    const ln = LNExpr.init(input, gamma, beta);
    const mm = ln.matmul(weight);

    var result = try eval(MatmulExpr, mm, std.testing.allocator);
    defer result.deinit();

    // With identity weight, result should equal layernorm output
    var ln_only = try eval(LNExpr, ln, std.testing.allocator);
    defer ln_only.deinit();

    for (result.slice(), ln_only.slice()) |r, l| {
        try std.testing.expectApproxEqAbs(r, l, 1e-5);
    }
}

test "eval gather basic" {
    // vocab_size=4, embed_dim=3
    const Table = Tensor(f32, .{ 4, 3 });
    const Indices = Tensor(u32, .{3});

    var table = try Table.fromSlice(std.testing.allocator, &[_]f32{
        1, 2, 3, // entry 0
        4, 5, 6, // entry 1
        7, 8, 9, // entry 2
        10, 11, 12, // entry 3
    });
    defer table.deinit();

    var indices = try Indices.fromSlice(std.testing.allocator, &[_]u32{ 0, 2, 1 });
    defer indices.deinit();

    const GatherExpr = ops.GatherExpr(Table, Indices);
    const expr = GatherExpr.init(table, indices);

    // Output should be [3, 3]
    try std.testing.expectEqual(@as(usize, 2), GatherExpr.ndim);
    try std.testing.expectEqual(@as(usize, 3), GatherExpr.shape[0]);
    try std.testing.expectEqual(@as(usize, 3), GatherExpr.shape[1]);

    var result = try eval(GatherExpr, expr, std.testing.allocator);
    defer result.deinit();

    // Entry 0
    try std.testing.expectEqual(@as(f32, 1), result.slice()[0]);
    try std.testing.expectEqual(@as(f32, 2), result.slice()[1]);
    try std.testing.expectEqual(@as(f32, 3), result.slice()[2]);

    // Entry 2
    try std.testing.expectEqual(@as(f32, 7), result.slice()[3]);
    try std.testing.expectEqual(@as(f32, 8), result.slice()[4]);
    try std.testing.expectEqual(@as(f32, 9), result.slice()[5]);

    // Entry 1
    try std.testing.expectEqual(@as(f32, 4), result.slice()[6]);
    try std.testing.expectEqual(@as(f32, 5), result.slice()[7]);
    try std.testing.expectEqual(@as(f32, 6), result.slice()[8]);
}

test "eval gather 2D indices (batch)" {
    // vocab_size=10, embed_dim=4, batch=2, seq_len=3
    const Table = Tensor(f32, .{ 10, 4 });
    const Indices = Tensor(u32, .{ 2, 3 });

    // Table: entry i has value i in all positions
    var table = try Table.zeros(std.testing.allocator);
    defer table.deinit();
    for (0..10) |i| {
        for (0..4) |j| {
            table.data[i * 4 + j] = @floatFromInt(i);
        }
    }

    var indices = try Indices.fromSlice(std.testing.allocator, &[_]u32{
        0, 5, 9, // batch 0
        1, 2, 3, // batch 1
    });
    defer indices.deinit();

    const GatherExpr = ops.GatherExpr(Table, Indices);
    const expr = GatherExpr.init(table, indices);

    // Output should be [2, 3, 4]
    try std.testing.expectEqual(@as(usize, 3), GatherExpr.ndim);
    try std.testing.expectEqual(@as(usize, 2), GatherExpr.shape[0]);
    try std.testing.expectEqual(@as(usize, 3), GatherExpr.shape[1]);
    try std.testing.expectEqual(@as(usize, 4), GatherExpr.shape[2]);

    var result = try eval(GatherExpr, expr, std.testing.allocator);
    defer result.deinit();

    // Batch 0, token 0 (index 0)
    for (0..4) |j| {
        try std.testing.expectEqual(@as(f32, 0), result.slice()[j]);
    }

    // Batch 0, token 1 (index 5)
    for (0..4) |j| {
        try std.testing.expectEqual(@as(f32, 5), result.slice()[4 + j]);
    }

    // Batch 0, token 2 (index 9)
    for (0..4) |j| {
        try std.testing.expectEqual(@as(f32, 9), result.slice()[8 + j]);
    }

    // Batch 1, token 0 (index 1)
    for (0..4) |j| {
        try std.testing.expectEqual(@as(f32, 1), result.slice()[12 + j]);
    }
}

test "eval gather chained with matmul" {
    // Typical transformer: embedding lookup -> linear projection
    const Table = Tensor(f32, .{ 10, 4 });
    const Indices = Tensor(u32, .{2});
    const Weight = Tensor(f32, .{ 4, 4 });

    var table = try Table.zeros(std.testing.allocator);
    defer table.deinit();
    for (0..10) |i| {
        for (0..4) |j| {
            table.data[i * 4 + j] = @floatFromInt(i);
        }
    }

    var indices = try Indices.fromSlice(std.testing.allocator, &[_]u32{ 1, 2 });
    defer indices.deinit();

    // Identity weight
    var weight = try Weight.zeros(std.testing.allocator);
    defer weight.deinit();
    for (0..4) |i| {
        weight.data[i * 4 + i] = 1.0;
    }

    // gather(table, indices) @ weight
    const GatherExpr = ops.GatherExpr(Table, Indices);
    const MatmulExpr = ops.MatmulExpr(GatherExpr, Weight);

    const gather_expr = GatherExpr.init(table, indices);
    const mm = gather_expr.matmul(weight);

    var result = try eval(MatmulExpr, mm, std.testing.allocator);
    defer result.deinit();

    // With identity weight, output should equal gather output
    var gather_only = try eval(GatherExpr, gather_expr, std.testing.allocator);
    defer gather_only.deinit();

    for (result.slice(), gather_only.slice()) |r, g| {
        try std.testing.expectApproxEqAbs(r, g, 1e-5);
    }
}

test "eval reshape" {
    const Vec = Tensor(f32, .{6});
    var input = try Vec.fromSlice(std.testing.allocator, &[_]f32{ 1, 2, 3, 4, 5, 6 });
    defer input.deinit();

    const ReshapeExpr = ops.ReshapeExpr(Vec, .{ 2, 3 });
    const expr = ReshapeExpr.init(input);

    // Shape should change
    try std.testing.expectEqual(@as(usize, 2), ReshapeExpr.ndim);
    try std.testing.expectEqual(@as(usize, 2), ReshapeExpr.shape[0]);
    try std.testing.expectEqual(@as(usize, 3), ReshapeExpr.shape[1]);

    var result = try eval(ReshapeExpr, expr, std.testing.allocator);
    defer result.deinit();

    // Data should be unchanged
    try std.testing.expectEqualSlices(f32, input.slice(), result.slice());
}

test "eval transpose 2D" {
    const Mat = Tensor(f32, .{ 2, 3 });
    var input = try Mat.fromSlice(std.testing.allocator, &[_]f32{
        1, 2, 3,
        4, 5, 6,
    });
    defer input.deinit();

    const TransposeExpr = ops.TransposeExpr(Mat, .{ 1, 0 });
    const expr = TransposeExpr.init(input);

    // Shape should be swapped
    try std.testing.expectEqual(@as(usize, 2), TransposeExpr.ndim);
    try std.testing.expectEqual(@as(usize, 3), TransposeExpr.shape[0]);
    try std.testing.expectEqual(@as(usize, 2), TransposeExpr.shape[1]);

    var result = try eval(TransposeExpr, expr, std.testing.allocator);
    defer result.deinit();

    // Expected: [[1, 4], [2, 5], [3, 6]]
    try std.testing.expectEqual(@as(f32, 1), result.slice()[0]);
    try std.testing.expectEqual(@as(f32, 4), result.slice()[1]);
    try std.testing.expectEqual(@as(f32, 2), result.slice()[2]);
    try std.testing.expectEqual(@as(f32, 5), result.slice()[3]);
    try std.testing.expectEqual(@as(f32, 3), result.slice()[4]);
    try std.testing.expectEqual(@as(f32, 6), result.slice()[5]);
}

test "eval transpose 4D attention pattern" {
    // [B=1, S=2, H=2, D=3] -> [B=1, H=2, S=2, D=3]
    const Input = Tensor(f32, .{ 1, 2, 2, 3 });
    var input = try Input.fromSlice(std.testing.allocator, &[_]f32{
        // S=0, H=0
        1, 2, 3,
        // S=0, H=1
        4, 5, 6,
        // S=1, H=0
        7, 8, 9,
        // S=1, H=1
        10, 11, 12,
    });
    defer input.deinit();

    const TransposeExpr = ops.TransposeExpr(Input, .{ 0, 2, 1, 3 });
    const expr = TransposeExpr.init(input);

    // Shape: [1, 2, 2, 3]
    try std.testing.expectEqual(@as(usize, 4), TransposeExpr.ndim);
    try std.testing.expectEqual(@as(usize, 1), TransposeExpr.shape[0]);
    try std.testing.expectEqual(@as(usize, 2), TransposeExpr.shape[1]); // H moved here
    try std.testing.expectEqual(@as(usize, 2), TransposeExpr.shape[2]); // S moved here
    try std.testing.expectEqual(@as(usize, 3), TransposeExpr.shape[3]);

    var result = try eval(TransposeExpr, expr, std.testing.allocator);
    defer result.deinit();

    // Output layout: H=0 (S=0: 1,2,3; S=1: 7,8,9), H=1 (S=0: 4,5,6; S=1: 10,11,12)
    try std.testing.expectEqual(@as(f32, 1), result.slice()[0]);
    try std.testing.expectEqual(@as(f32, 2), result.slice()[1]);
    try std.testing.expectEqual(@as(f32, 3), result.slice()[2]);
    try std.testing.expectEqual(@as(f32, 7), result.slice()[3]);
    try std.testing.expectEqual(@as(f32, 8), result.slice()[4]);
    try std.testing.expectEqual(@as(f32, 9), result.slice()[5]);
    try std.testing.expectEqual(@as(f32, 4), result.slice()[6]);
    try std.testing.expectEqual(@as(f32, 10), result.slice()[9]);
}

test "eval attention reshape/transpose dance" {
    // This is the critical test the user flagged
    // [B, S, D] -> [B, S, H, D_h] -> [B, H, S, D_h]
    // B=1, S=2, D=6, H=2, D_h=3

    const Hidden = Tensor(f32, .{ 1, 2, 6 });
    var input = try Hidden.fromSlice(std.testing.allocator, &[_]f32{
        // Token 0: heads interleaved [h0_d0, h0_d1, h0_d2, h1_d0, h1_d1, h1_d2]
        1, 2, 3, 4, 5, 6,
        // Token 1
        7, 8, 9, 10, 11, 12,
    });
    defer input.deinit();

    // Step 1: Reshape [1, 2, 6] -> [1, 2, 2, 3]
    const ReshapeExpr = ops.ReshapeExpr(Hidden, .{ 1, 2, 2, 3 });
    const reshaped = ReshapeExpr.init(input);

    // Step 2: Transpose [1, 2, 2, 3] -> [1, 2, 2, 3] with perm [0, 2, 1, 3]
    const TransposeExpr = ops.TransposeExpr(ReshapeExpr, .{ 0, 2, 1, 3 });
    const transposed = reshaped.transpose(.{ 0, 2, 1, 3 });

    var result = try eval(TransposeExpr, transposed, std.testing.allocator);
    defer result.deinit();

    // Shape should be [B=1, H=2, S=2, D_h=3]
    try std.testing.expectEqual(@as(usize, 4), TransposeExpr.ndim);
    try std.testing.expectEqual(@as(usize, 1), TransposeExpr.shape[0]);
    try std.testing.expectEqual(@as(usize, 2), TransposeExpr.shape[1]);
    try std.testing.expectEqual(@as(usize, 2), TransposeExpr.shape[2]);
    try std.testing.expectEqual(@as(usize, 3), TransposeExpr.shape[3]);

    // After reshape+transpose:
    // Head 0: Token 0 (1,2,3), Token 1 (7,8,9)
    // Head 1: Token 0 (4,5,6), Token 1 (10,11,12)
    try std.testing.expectEqual(@as(f32, 1), result.slice()[0]);
    try std.testing.expectEqual(@as(f32, 2), result.slice()[1]);
    try std.testing.expectEqual(@as(f32, 3), result.slice()[2]);
    try std.testing.expectEqual(@as(f32, 7), result.slice()[3]);
    try std.testing.expectEqual(@as(f32, 8), result.slice()[4]);
    try std.testing.expectEqual(@as(f32, 9), result.slice()[5]);
    try std.testing.expectEqual(@as(f32, 4), result.slice()[6]);
    try std.testing.expectEqual(@as(f32, 5), result.slice()[7]);
    try std.testing.expectEqual(@as(f32, 6), result.slice()[8]);
    try std.testing.expectEqual(@as(f32, 10), result.slice()[9]);
    try std.testing.expectEqual(@as(f32, 11), result.slice()[10]);
    try std.testing.expectEqual(@as(f32, 12), result.slice()[11]);
}
