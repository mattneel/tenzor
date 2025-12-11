//! Expression graph system for lazy tensor computation.
//!
//! Operations on tensors don't execute immediately - they build a comptime
//! expression graph encoded in types. The graph is only executed when `.eval()`
//! is called, enabling automatic operation fusion.

const std = @import("std");
const core = struct {
    const dtype = @import("../core/dtype.zig");
    const shape = @import("../core/shape.zig");
    const tensor = @import("../core/tensor.zig");
    const strides = @import("../core/strides.zig");
};

const Shape = core.shape.Shape;
const Tensor = core.tensor.Tensor;

/// Operation tags for all supported operations.
pub const OpTag = enum {
    // Unary elementwise operations
    neg,
    abs,
    exp,
    exp2,
    log,
    log2,
    log10,
    sqrt,
    rsqrt,
    sin,
    cos,
    tan,
    tanh,
    sinh,
    cosh,
    asin,
    acos,
    atan,
    sigmoid,
    relu,
    leaky_relu,
    gelu,
    silu,
    softplus,
    ceil,
    floor,
    round,
    sign,

    // Binary elementwise operations
    add,
    sub,
    mul,
    div,
    pow,
    max,
    min,
    mod,
    atan2,

    // Comparison operations (return same type for masking)
    eq,
    ne,
    lt,
    le,
    gt,
    ge,

    // Matrix operations
    matmul,
    transpose,

    // Reduction operations
    sum,
    prod,
    reduce_max,
    reduce_min,
    mean,
    variance,

    // Shape operations
    reshape,
    broadcast,
    slice,
    concat,
    squeeze,
    unsqueeze,

    // Indexing operations
    gather,
    scatter,

    /// Returns true if this is a unary elementwise operation.
    pub fn isUnary(self: OpTag) bool {
        return switch (self) {
            .neg, .abs, .exp, .exp2, .log, .log2, .log10, .sqrt, .rsqrt, .sin, .cos, .tan, .tanh, .sinh, .cosh, .asin, .acos, .atan, .sigmoid, .relu, .leaky_relu, .gelu, .silu, .softplus, .ceil, .floor, .round, .sign => true,
            else => false,
        };
    }

    /// Returns true if this is a binary elementwise operation.
    pub fn isBinary(self: OpTag) bool {
        return switch (self) {
            .add, .sub, .mul, .div, .pow, .max, .min, .mod, .atan2, .eq, .ne, .lt, .le, .gt, .ge => true,
            else => false,
        };
    }

    /// Returns true if this is an elementwise operation (unary or binary).
    pub fn isElementwise(self: OpTag) bool {
        return self.isUnary() or self.isBinary();
    }

    /// Returns true if this is a reduction operation.
    pub fn isReduction(self: OpTag) bool {
        return switch (self) {
            .sum, .prod, .reduce_max, .reduce_min, .mean, .variance => true,
            else => false,
        };
    }

    /// Returns true if this operation is fuseable into an elementwise chain.
    pub fn isFuseable(self: OpTag) bool {
        return self.isElementwise();
    }

    /// Returns true if this is an activation function.
    pub fn isActivation(self: OpTag) bool {
        return switch (self) {
            .relu, .leaky_relu, .gelu, .silu, .sigmoid, .tanh, .softplus => true,
            else => false,
        };
    }
};

/// Node kind for expression traversal.
pub const NodeKind = enum {
    tensor, // Leaf: concrete tensor
    constant, // Leaf: comptime constant
    unary, // Unary operation
    binary, // Binary operation
    matmul, // Matrix multiplication
    reduce, // Reduction
    reshape, // Shape manipulation
    transpose, // Axis permutation
    softmax, // Softmax normalization
    layernorm, // Layer normalization
};

/// Marker trait to identify expression types.
pub fn isExprType(comptime T: type) bool {
    const info = @typeInfo(T);
    // Only structs can be expression types
    if (info != .@"struct") return false;
    return @hasDecl(T, "ExpressionMarker") and T.ExpressionMarker;
}

/// Marker trait to identify tensor types (leaves in the expression graph).
pub fn isTensorType(comptime T: type) bool {
    const info = @typeInfo(T);
    // Only structs can be tensor types
    if (info != .@"struct") return false;
    return @hasDecl(T, "ElementType") and @hasDecl(T, "shape") and @hasDecl(T, "numel");
}

/// Get the element type from any expression or tensor type.
pub fn ElementTypeOf(comptime T: type) type {
    if (@hasDecl(T, "ElementType")) {
        return T.ElementType;
    } else if (@hasDecl(T, "T")) {
        return T.T;
    } else {
        @compileError("Cannot determine element type of " ++ @typeName(T));
    }
}

/// Get the shape from any expression or tensor type.
pub fn ShapeOf(comptime T: type) [RankOf(T)]usize {
    if (@hasDecl(T, "shape")) {
        return T.shape;
    } else {
        @compileError("Cannot determine shape of " ++ @typeName(T));
    }
}

/// Get the rank (number of dimensions) from any expression or tensor type.
pub fn RankOf(comptime T: type) usize {
    if (@hasDecl(T, "ndim")) {
        return T.ndim;
    } else {
        @compileError("Cannot determine rank of " ++ @typeName(T));
    }
}

/// Check if an expression type represents a scalar.
pub fn isScalarExpr(comptime T: type) bool {
    return RankOf(T) == 0;
}

/// Check if an expression type represents a vector.
pub fn isVectorExpr(comptime T: type) bool {
    return RankOf(T) == 1;
}

/// Check if an expression type represents a matrix.
pub fn isMatrixExpr(comptime T: type) bool {
    return RankOf(T) == 2;
}

/// Unary expression type.
pub fn UnaryExpr(comptime op: OpTag, comptime Input: type) type {
    comptime {
        if (!op.isUnary()) {
            @compileError("Expected unary operation, got " ++ @tagName(op));
        }
    }

    return struct {
        pub const ExpressionMarker = true;
        pub const kind: NodeKind = .unary;
        pub const operation: OpTag = op;
        pub const InputType = Input;
        pub const ElementType = ElementTypeOf(Input);
        pub const ndim = RankOf(Input);
        pub const shape = ShapeOf(Input);

        input: Input,

        const Self = @This();

        pub fn init(input: Input) Self {
            return .{ .input = input };
        }

        // Unary operations
        pub fn neg(self: Self) UnaryExpr(.neg, Self) {
            return UnaryExpr(.neg, Self).init(self);
        }

        pub fn abs(self: Self) UnaryExpr(.abs, Self) {
            return UnaryExpr(.abs, Self).init(self);
        }

        pub fn @"exp"(self: Self) UnaryExpr(.exp, Self) {
            return UnaryExpr(.exp, Self).init(self);
        }

        pub fn @"log"(self: Self) UnaryExpr(.log, Self) {
            return UnaryExpr(.log, Self).init(self);
        }

        pub fn @"sqrt"(self: Self) UnaryExpr(.sqrt, Self) {
            return UnaryExpr(.sqrt, Self).init(self);
        }

        pub fn rsqrt(self: Self) UnaryExpr(.rsqrt, Self) {
            return UnaryExpr(.rsqrt, Self).init(self);
        }

        pub fn @"sin"(self: Self) UnaryExpr(.sin, Self) {
            return UnaryExpr(.sin, Self).init(self);
        }

        pub fn @"cos"(self: Self) UnaryExpr(.cos, Self) {
            return UnaryExpr(.cos, Self).init(self);
        }

        pub fn @"tanh"(self: Self) UnaryExpr(.tanh, Self) {
            return UnaryExpr(.tanh, Self).init(self);
        }

        pub fn sigmoid(self: Self) UnaryExpr(.sigmoid, Self) {
            return UnaryExpr(.sigmoid, Self).init(self);
        }

        pub fn relu(self: Self) UnaryExpr(.relu, Self) {
            return UnaryExpr(.relu, Self).init(self);
        }

        pub fn gelu(self: Self) UnaryExpr(.gelu, Self) {
            return UnaryExpr(.gelu, Self).init(self);
        }

        pub fn silu(self: Self) UnaryExpr(.silu, Self) {
            return UnaryExpr(.silu, Self).init(self);
        }

        // Binary operations
        pub fn add(self: Self, other: anytype) BinaryExpr(.add, Self, AsExpr(@TypeOf(other))) {
            return BinaryExpr(.add, Self, AsExpr(@TypeOf(other))).init(self, asExpr(other));
        }

        pub fn sub(self: Self, other: anytype) BinaryExpr(.sub, Self, AsExpr(@TypeOf(other))) {
            return BinaryExpr(.sub, Self, AsExpr(@TypeOf(other))).init(self, asExpr(other));
        }

        pub fn mul(self: Self, other: anytype) BinaryExpr(.mul, Self, AsExpr(@TypeOf(other))) {
            return BinaryExpr(.mul, Self, AsExpr(@TypeOf(other))).init(self, asExpr(other));
        }

        pub fn div(self: Self, other: anytype) BinaryExpr(.div, Self, AsExpr(@TypeOf(other))) {
            return BinaryExpr(.div, Self, AsExpr(@TypeOf(other))).init(self, asExpr(other));
        }

        pub fn @"pow"(self: Self, other: anytype) BinaryExpr(.pow, Self, AsExpr(@TypeOf(other))) {
            return BinaryExpr(.pow, Self, AsExpr(@TypeOf(other))).init(self, asExpr(other));
        }

        pub fn maximum(self: Self, other: anytype) BinaryExpr(.max, Self, AsExpr(@TypeOf(other))) {
            return BinaryExpr(.max, Self, AsExpr(@TypeOf(other))).init(self, asExpr(other));
        }

        pub fn minimum(self: Self, other: anytype) BinaryExpr(.min, Self, AsExpr(@TypeOf(other))) {
            return BinaryExpr(.min, Self, AsExpr(@TypeOf(other))).init(self, asExpr(other));
        }

        // Matrix operations
        pub fn matmul(self: Self, other: anytype) MatmulExpr(Self, @TypeOf(other)) {
            return MatmulExpr(Self, @TypeOf(other)).init(self, other);
        }

        // Reduction operations
        pub fn sum(self: Self) ReduceExpr(.sum, Self, .{}, false) {
            return ReduceExpr(.sum, Self, .{}, false).init(self);
        }

        pub fn mean(self: Self) ReduceExpr(.mean, Self, .{}, false) {
            return ReduceExpr(.mean, Self, .{}, false).init(self);
        }

        pub fn prod(self: Self) ReduceExpr(.prod, Self, .{}, false) {
            return ReduceExpr(.prod, Self, .{}, false).init(self);
        }

        pub fn reduceMax(self: Self) ReduceExpr(.reduce_max, Self, .{}, false) {
            return ReduceExpr(.reduce_max, Self, .{}, false).init(self);
        }

        pub fn reduceMin(self: Self) ReduceExpr(.reduce_min, Self, .{}, false) {
            return ReduceExpr(.reduce_min, Self, .{}, false).init(self);
        }

        // Softmax (default axis=-1)
        pub fn softmax(self: Self, comptime axis: isize) SoftmaxExpr(Self, axis) {
            return SoftmaxExpr(Self, axis).init(self);
        }
    };
}

/// Binary expression type.
pub fn BinaryExpr(comptime op: OpTag, comptime Lhs: type, comptime Rhs: type) type {
    comptime {
        if (!op.isBinary()) {
            @compileError("Expected binary operation, got " ++ @tagName(op));
        }

        // Type check
        if (ElementTypeOf(Lhs) != ElementTypeOf(Rhs)) {
            @compileError(std.fmt.comptimePrint(
                "Type mismatch in binary operation: {s} vs {s}",
                .{ @typeName(ElementTypeOf(Lhs)), @typeName(ElementTypeOf(Rhs)) },
            ));
        }

        // Shape check (broadcast compatibility)
        if (!core.shape.broadcastCompatible(
            Shape(ShapeOf(Lhs)),
            Shape(ShapeOf(Rhs)),
        )) {
            @compileError(std.fmt.comptimePrint(
                "Shapes not broadcast compatible: {any} vs {any}",
                .{ ShapeOf(Lhs), ShapeOf(Rhs) },
            ));
        }
    }

    const ResultShape = core.shape.BroadcastShape(Shape(ShapeOf(Lhs)), Shape(ShapeOf(Rhs)));

    return struct {
        pub const ExpressionMarker = true;
        pub const kind: NodeKind = .binary;
        pub const operation: OpTag = op;
        pub const LhsType = Lhs;
        pub const RhsType = Rhs;
        pub const ElementType = ElementTypeOf(Lhs);
        pub const ndim = ResultShape.ndim;
        pub const shape = ResultShape.dimensions;

        lhs: Lhs,
        rhs: Rhs,

        const Self = @This();

        pub fn init(lhs: Lhs, rhs: Rhs) Self {
            return .{ .lhs = lhs, .rhs = rhs };
        }

        // Unary operations
        pub fn neg(self: Self) UnaryExpr(.neg, Self) {
            return UnaryExpr(.neg, Self).init(self);
        }

        pub fn abs(self: Self) UnaryExpr(.abs, Self) {
            return UnaryExpr(.abs, Self).init(self);
        }

        pub fn @"exp"(self: Self) UnaryExpr(.exp, Self) {
            return UnaryExpr(.exp, Self).init(self);
        }

        pub fn @"log"(self: Self) UnaryExpr(.log, Self) {
            return UnaryExpr(.log, Self).init(self);
        }

        pub fn @"sqrt"(self: Self) UnaryExpr(.sqrt, Self) {
            return UnaryExpr(.sqrt, Self).init(self);
        }

        pub fn rsqrt(self: Self) UnaryExpr(.rsqrt, Self) {
            return UnaryExpr(.rsqrt, Self).init(self);
        }

        pub fn @"sin"(self: Self) UnaryExpr(.sin, Self) {
            return UnaryExpr(.sin, Self).init(self);
        }

        pub fn @"cos"(self: Self) UnaryExpr(.cos, Self) {
            return UnaryExpr(.cos, Self).init(self);
        }

        pub fn @"tanh"(self: Self) UnaryExpr(.tanh, Self) {
            return UnaryExpr(.tanh, Self).init(self);
        }

        pub fn sigmoid(self: Self) UnaryExpr(.sigmoid, Self) {
            return UnaryExpr(.sigmoid, Self).init(self);
        }

        pub fn relu(self: Self) UnaryExpr(.relu, Self) {
            return UnaryExpr(.relu, Self).init(self);
        }

        pub fn gelu(self: Self) UnaryExpr(.gelu, Self) {
            return UnaryExpr(.gelu, Self).init(self);
        }

        pub fn silu(self: Self) UnaryExpr(.silu, Self) {
            return UnaryExpr(.silu, Self).init(self);
        }

        // Binary operations
        pub fn add(self: Self, other: anytype) BinaryExpr(.add, Self, AsExpr(@TypeOf(other))) {
            return BinaryExpr(.add, Self, AsExpr(@TypeOf(other))).init(self, asExpr(other));
        }

        pub fn sub(self: Self, other: anytype) BinaryExpr(.sub, Self, AsExpr(@TypeOf(other))) {
            return BinaryExpr(.sub, Self, AsExpr(@TypeOf(other))).init(self, asExpr(other));
        }

        pub fn mul(self: Self, other: anytype) BinaryExpr(.mul, Self, AsExpr(@TypeOf(other))) {
            return BinaryExpr(.mul, Self, AsExpr(@TypeOf(other))).init(self, asExpr(other));
        }

        pub fn div(self: Self, other: anytype) BinaryExpr(.div, Self, AsExpr(@TypeOf(other))) {
            return BinaryExpr(.div, Self, AsExpr(@TypeOf(other))).init(self, asExpr(other));
        }

        pub fn @"pow"(self: Self, other: anytype) BinaryExpr(.pow, Self, AsExpr(@TypeOf(other))) {
            return BinaryExpr(.pow, Self, AsExpr(@TypeOf(other))).init(self, asExpr(other));
        }

        pub fn maximum(self: Self, other: anytype) BinaryExpr(.max, Self, AsExpr(@TypeOf(other))) {
            return BinaryExpr(.max, Self, AsExpr(@TypeOf(other))).init(self, asExpr(other));
        }

        pub fn minimum(self: Self, other: anytype) BinaryExpr(.min, Self, AsExpr(@TypeOf(other))) {
            return BinaryExpr(.min, Self, AsExpr(@TypeOf(other))).init(self, asExpr(other));
        }

        // Matrix operations
        pub fn matmul(self: Self, other: anytype) MatmulExpr(Self, @TypeOf(other)) {
            return MatmulExpr(Self, @TypeOf(other)).init(self, other);
        }

        // Reduction operations
        pub fn sum(self: Self) ReduceExpr(.sum, Self, .{}, false) {
            return ReduceExpr(.sum, Self, .{}, false).init(self);
        }

        pub fn mean(self: Self) ReduceExpr(.mean, Self, .{}, false) {
            return ReduceExpr(.mean, Self, .{}, false).init(self);
        }

        pub fn prod(self: Self) ReduceExpr(.prod, Self, .{}, false) {
            return ReduceExpr(.prod, Self, .{}, false).init(self);
        }

        pub fn reduceMax(self: Self) ReduceExpr(.reduce_max, Self, .{}, false) {
            return ReduceExpr(.reduce_max, Self, .{}, false).init(self);
        }

        pub fn reduceMin(self: Self) ReduceExpr(.reduce_min, Self, .{}, false) {
            return ReduceExpr(.reduce_min, Self, .{}, false).init(self);
        }

        // Softmax
        pub fn softmax(self: Self, comptime axis: isize) SoftmaxExpr(Self, axis) {
            return SoftmaxExpr(Self, axis).init(self);
        }
    };
}

/// Matrix multiplication expression type.
pub fn MatmulExpr(comptime Lhs: type, comptime Rhs: type) type {
    comptime {
        // Type check
        if (ElementTypeOf(Lhs) != ElementTypeOf(Rhs)) {
            @compileError(std.fmt.comptimePrint(
                "Type mismatch in matmul: {s} vs {s}",
                .{ @typeName(ElementTypeOf(Lhs)), @typeName(ElementTypeOf(Rhs)) },
            ));
        }

        // Shape check
        if (!core.shape.matmulCompatible(Shape(ShapeOf(Lhs)), Shape(ShapeOf(Rhs)))) {
            @compileError(std.fmt.comptimePrint(
                "Shapes not compatible for matmul: {any} @ {any}",
                .{ ShapeOf(Lhs), ShapeOf(Rhs) },
            ));
        }
    }

    const ResultShape = core.shape.MatmulShape(Shape(ShapeOf(Lhs)), Shape(ShapeOf(Rhs)));

    return struct {
        pub const ExpressionMarker = true;
        pub const kind: NodeKind = .matmul;
        pub const operation: OpTag = .matmul;
        pub const LhsType = Lhs;
        pub const RhsType = Rhs;
        pub const ElementType = ElementTypeOf(Lhs);
        pub const ndim = ResultShape.ndim;
        pub const shape = ResultShape.dimensions;

        lhs: Lhs,
        rhs: Rhs,

        const Self = @This();

        pub fn init(lhs: Lhs, rhs: Rhs) Self {
            return .{ .lhs = lhs, .rhs = rhs };
        }

        // Unary operations (for chaining after matmul)
        pub fn neg(self: Self) UnaryExpr(.neg, Self) {
            return UnaryExpr(.neg, Self).init(self);
        }

        pub fn relu(self: Self) UnaryExpr(.relu, Self) {
            return UnaryExpr(.relu, Self).init(self);
        }

        pub fn sigmoid(self: Self) UnaryExpr(.sigmoid, Self) {
            return UnaryExpr(.sigmoid, Self).init(self);
        }

        pub fn @"tanh"(self: Self) UnaryExpr(.tanh, Self) {
            return UnaryExpr(.tanh, Self).init(self);
        }

        pub fn gelu(self: Self) UnaryExpr(.gelu, Self) {
            return UnaryExpr(.gelu, Self).init(self);
        }

        // Binary operations
        pub fn add(self: Self, other: anytype) BinaryExpr(.add, Self, AsExpr(@TypeOf(other))) {
            return BinaryExpr(.add, Self, AsExpr(@TypeOf(other))).init(self, asExpr(other));
        }

        pub fn sub(self: Self, other: anytype) BinaryExpr(.sub, Self, AsExpr(@TypeOf(other))) {
            return BinaryExpr(.sub, Self, AsExpr(@TypeOf(other))).init(self, asExpr(other));
        }

        pub fn mul(self: Self, other: anytype) BinaryExpr(.mul, Self, AsExpr(@TypeOf(other))) {
            return BinaryExpr(.mul, Self, AsExpr(@TypeOf(other))).init(self, asExpr(other));
        }

        // Reduction operations
        pub fn sum(self: Self) ReduceExpr(.sum, Self, .{}, false) {
            return ReduceExpr(.sum, Self, .{}, false).init(self);
        }

        pub fn mean(self: Self) ReduceExpr(.mean, Self, .{}, false) {
            return ReduceExpr(.mean, Self, .{}, false).init(self);
        }

        // Softmax
        pub fn softmax(self: Self, comptime axis: isize) SoftmaxExpr(Self, axis) {
            return SoftmaxExpr(Self, axis).init(self);
        }
    };
}

/// Reduction expression type.
pub fn ReduceExpr(
    comptime op: OpTag,
    comptime Input: type,
    comptime axes: anytype,
    comptime keepdims: bool,
) type {
    comptime {
        if (!op.isReduction()) {
            @compileError("Expected reduction operation, got " ++ @tagName(op));
        }
    }

    const ResultShape = core.shape.ReduceShape(Shape(ShapeOf(Input)), axes, keepdims);

    return struct {
        pub const ExpressionMarker = true;
        pub const kind: NodeKind = .reduce;
        pub const operation: OpTag = op;
        pub const InputType = Input;
        pub const ElementType = ElementTypeOf(Input);
        pub const ndim = ResultShape.ndim;
        pub const shape = ResultShape.dimensions;
        pub const reduction_axes = axes;
        pub const keep_dims = keepdims;

        input: Input,

        const Self = @This();

        pub fn init(input: Input) Self {
            return .{ .input = input };
        }

        // Can chain more operations on reduction result
        pub fn add(self: Self, other: anytype) BinaryExpr(.add, Self, AsExpr(@TypeOf(other))) {
            return BinaryExpr(.add, Self, AsExpr(@TypeOf(other))).init(self, asExpr(other));
        }

        pub fn mul(self: Self, other: anytype) BinaryExpr(.mul, Self, AsExpr(@TypeOf(other))) {
            return BinaryExpr(.mul, Self, AsExpr(@TypeOf(other))).init(self, asExpr(other));
        }

        pub fn div(self: Self, other: anytype) BinaryExpr(.div, Self, AsExpr(@TypeOf(other))) {
            return BinaryExpr(.div, Self, AsExpr(@TypeOf(other))).init(self, asExpr(other));
        }

        pub fn @"sqrt"(self: Self) UnaryExpr(.sqrt, Self) {
            return UnaryExpr(.sqrt, Self).init(self);
        }

        // Softmax
        pub fn softmax(self: Self, comptime axis: isize) SoftmaxExpr(Self, axis) {
            return SoftmaxExpr(Self, axis).init(self);
        }
    };
}

/// Softmax expression type.
/// Computes softmax along a specified axis with numerical stability.
/// softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
pub fn SoftmaxExpr(comptime Input: type, comptime axis: isize) type {
    const normalized_axis = core.shape.normalizeAxis(RankOf(Input), axis);

    return struct {
        pub const ExpressionMarker = true;
        pub const kind: NodeKind = .softmax;
        pub const InputType = Input;
        pub const ElementType = ElementTypeOf(Input);
        pub const ndim = RankOf(Input);
        pub const shape = ShapeOf(Input);
        pub const softmax_axis: usize = normalized_axis;

        input: Input,

        const Self = @This();

        pub fn init(input: Input) Self {
            return .{ .input = input };
        }

        // Can chain operations after softmax
        pub fn matmul(self: Self, other: anytype) MatmulExpr(Self, @TypeOf(other)) {
            return MatmulExpr(Self, @TypeOf(other)).init(self, other);
        }

        pub fn mul(self: Self, other: anytype) BinaryExpr(.mul, Self, AsExpr(@TypeOf(other))) {
            return BinaryExpr(.mul, Self, AsExpr(@TypeOf(other))).init(self, asExpr(other));
        }

        pub fn add(self: Self, other: anytype) BinaryExpr(.add, Self, AsExpr(@TypeOf(other))) {
            return BinaryExpr(.add, Self, AsExpr(@TypeOf(other))).init(self, asExpr(other));
        }
    };
}

/// Layer normalization expression type.
/// Computes: (x - mean) / sqrt(var + eps) * gamma + beta
/// Applied over the last `normalized_dims` dimensions.
/// For transformer: input [B, S, D], gamma [D], beta [D], normalized_dims=1
pub fn LayerNormExpr(
    comptime Input: type,
    comptime Gamma: type,
    comptime Beta: type,
    comptime normalized_dims: usize,
) type {
    comptime {
        // Type check
        if (ElementTypeOf(Input) != ElementTypeOf(Gamma) or ElementTypeOf(Input) != ElementTypeOf(Beta)) {
            @compileError("Type mismatch in layernorm");
        }

        // Shape check: gamma and beta should match the last `normalized_dims` of input
        const input_shape = ShapeOf(Input);
        const gamma_shape = ShapeOf(Gamma);
        const beta_shape = ShapeOf(Beta);

        if (RankOf(Gamma) != normalized_dims or RankOf(Beta) != normalized_dims) {
            @compileError(std.fmt.comptimePrint(
                "Gamma/beta rank must equal normalized_dims. Got gamma={d}, beta={d}, normalized_dims={d}",
                .{ RankOf(Gamma), RankOf(Beta), normalized_dims },
            ));
        }

        // Check dimensions match
        for (0..normalized_dims) |i| {
            const input_dim = input_shape[RankOf(Input) - normalized_dims + i];
            if (gamma_shape[i] != input_dim or beta_shape[i] != input_dim) {
                @compileError("Gamma/beta shape must match last dimensions of input");
            }
        }
    }

    return struct {
        pub const ExpressionMarker = true;
        pub const kind: NodeKind = .layernorm;
        pub const InputType = Input;
        pub const GammaType = Gamma;
        pub const BetaType = Beta;
        pub const ElementType = ElementTypeOf(Input);
        pub const ndim = RankOf(Input);
        pub const shape = ShapeOf(Input);
        pub const norm_dims: usize = normalized_dims;

        input: Input,
        gamma: Gamma,
        beta: Beta,

        const Self = @This();

        pub fn init(input: Input, gamma: Gamma, beta: Beta) Self {
            return .{ .input = input, .gamma = gamma, .beta = beta };
        }

        // Can chain operations after layernorm
        pub fn matmul(self: Self, other: anytype) MatmulExpr(Self, @TypeOf(other)) {
            return MatmulExpr(Self, @TypeOf(other)).init(self, other);
        }

        pub fn mul(self: Self, other: anytype) BinaryExpr(.mul, Self, AsExpr(@TypeOf(other))) {
            return BinaryExpr(.mul, Self, AsExpr(@TypeOf(other))).init(self, asExpr(other));
        }

        pub fn add(self: Self, other: anytype) BinaryExpr(.add, Self, AsExpr(@TypeOf(other))) {
            return BinaryExpr(.add, Self, AsExpr(@TypeOf(other))).init(self, asExpr(other));
        }

        pub fn softmax(self: Self, comptime axis: isize) SoftmaxExpr(Self, axis) {
            return SoftmaxExpr(Self, axis).init(self);
        }
    };
}

/// Scalar constant expression (for broadcasting scalars).
pub fn ScalarExpr(comptime T: type) type {
    return struct {
        pub const ExpressionMarker = true;
        pub const kind: NodeKind = .constant;
        pub const ElementType = T;
        pub const ndim: usize = 0;
        pub const shape: [0]usize = .{};

        value: T,

        const Self = @This();

        pub fn init(value: T) Self {
            return .{ .value = value };
        }

        // Scalars can participate in binary ops
        pub fn add(self: Self, other: anytype) BinaryExpr(.add, Self, AsExpr(@TypeOf(other))) {
            return BinaryExpr(.add, Self, AsExpr(@TypeOf(other))).init(self, asExpr(other));
        }

        pub fn mul(self: Self, other: anytype) BinaryExpr(.mul, Self, AsExpr(@TypeOf(other))) {
            return BinaryExpr(.mul, Self, AsExpr(@TypeOf(other))).init(self, asExpr(other));
        }
    };
}

/// Convert a value to an expression type.
/// - Expressions pass through unchanged
/// - Scalars become ScalarExpr
/// - Tensors become TensorRef
pub fn AsExpr(comptime T: type) type {
    if (isExprType(T)) {
        return T;
    } else if (isTensorType(T)) {
        return T; // Tensors are already expression-like
    } else if (@typeInfo(T) == .int or @typeInfo(T) == .float or @typeInfo(T) == .comptime_int or @typeInfo(T) == .comptime_float) {
        // Scalar type - determine the actual type
        const ActualT = if (@typeInfo(T) == .comptime_int) i64 else if (@typeInfo(T) == .comptime_float) f64 else T;
        return ScalarExpr(ActualT);
    } else {
        @compileError("Cannot convert " ++ @typeName(T) ++ " to expression");
    }
}

/// Runtime conversion to expression.
pub fn asExpr(value: anytype) AsExpr(@TypeOf(value)) {
    const T = @TypeOf(value);
    if (isExprType(T) or isTensorType(T)) {
        return value;
    } else {
        return AsExpr(T).init(value);
    }
}

// ============================================================================
// Tests
// ============================================================================

test "OpTag properties" {
    try std.testing.expect(OpTag.relu.isUnary());
    try std.testing.expect(OpTag.exp.isUnary());
    try std.testing.expect(!OpTag.add.isUnary());

    try std.testing.expect(OpTag.add.isBinary());
    try std.testing.expect(OpTag.mul.isBinary());
    try std.testing.expect(!OpTag.relu.isBinary());

    try std.testing.expect(OpTag.relu.isElementwise());
    try std.testing.expect(OpTag.add.isElementwise());
    try std.testing.expect(!OpTag.matmul.isElementwise());

    try std.testing.expect(OpTag.sum.isReduction());
    try std.testing.expect(!OpTag.add.isReduction());
}

test "UnaryExpr shape preservation" {
    const Vec = Tensor(f32, .{4});
    const UnaryType = UnaryExpr(.relu, Vec);

    try std.testing.expectEqual(@as(usize, 1), UnaryType.ndim);
    try std.testing.expectEqual(@as(usize, 4), UnaryType.shape[0]);
    try std.testing.expectEqual(f32, UnaryType.ElementType);
}

test "BinaryExpr broadcasting" {
    const Mat = Tensor(f32, .{ 3, 4 });
    const Vec = Tensor(f32, .{4});
    const AddType = BinaryExpr(.add, Mat, Vec);

    // Result should be broadcast shape
    try std.testing.expectEqual(@as(usize, 2), AddType.ndim);
    try std.testing.expectEqual(@as(usize, 3), AddType.shape[0]);
    try std.testing.expectEqual(@as(usize, 4), AddType.shape[1]);
}

test "MatmulExpr shape" {
    const A = Tensor(f32, .{ 3, 4 });
    const B = Tensor(f32, .{ 4, 5 });
    const MulType = MatmulExpr(A, B);

    try std.testing.expectEqual(@as(usize, 2), MulType.ndim);
    try std.testing.expectEqual(@as(usize, 3), MulType.shape[0]);
    try std.testing.expectEqual(@as(usize, 5), MulType.shape[1]);
}

test "ReduceExpr shape" {
    const Mat = Tensor(f32, .{ 3, 4 });

    // Full reduction
    const SumType = ReduceExpr(.sum, Mat, .{}, false);
    try std.testing.expectEqual(@as(usize, 0), SumType.ndim);
}

test "Expression chaining types" {
    const Vec = Tensor(f32, .{4});

    // Chain: relu -> add -> exp
    const E1 = UnaryExpr(.relu, Vec);
    const E2 = BinaryExpr(.add, E1, Vec);
    const E3 = UnaryExpr(.exp, E2);

    try std.testing.expectEqual(@as(usize, 1), E3.ndim);
    try std.testing.expectEqual(@as(usize, 4), E3.shape[0]);
    try std.testing.expect(E3.ExpressionMarker);
}

test "ScalarExpr" {
    const S = ScalarExpr(f32);
    try std.testing.expectEqual(@as(usize, 0), S.ndim);
    try std.testing.expectEqual(f32, S.ElementType);
}

test "AsExpr conversion" {
    // Scalar int
    try std.testing.expectEqual(ScalarExpr(i64), AsExpr(@TypeOf(42)));

    // Scalar float
    try std.testing.expectEqual(ScalarExpr(f64), AsExpr(@TypeOf(3.14)));

    // Tensor passes through
    const Vec = Tensor(f32, .{4});
    try std.testing.expectEqual(Vec, AsExpr(Vec));
}
