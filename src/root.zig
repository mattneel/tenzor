//! tenzor - A comptime tensor library for Zig
//!
//! Zero-cost abstractions, automatic operation fusion, and CPU SIMD execution.
//! Shape errors become compile errors, not runtime panics.

// Core types
pub const core = struct {
    pub const dtype = @import("core/dtype.zig");
    pub const shape = @import("core/shape.zig");
    pub const strides = @import("core/strides.zig");
    pub const tensor = @import("core/tensor.zig");
};

// Operations and expression graph
pub const ops = struct {
    pub const expr = @import("ops/expr.zig");
};

// Backend implementations
pub const backend = struct {
    pub const cpu = struct {
        pub const simd = @import("backend/cpu/simd.zig");
        pub const executor = @import("backend/cpu/executor.zig");
        pub const threading = @import("backend/cpu/threading.zig");
        pub const kernels = struct {
            pub const elementwise = @import("backend/cpu/kernels/elementwise.zig");
            pub const matmul = @import("backend/cpu/kernels/matmul.zig");
            pub const reduce = @import("backend/cpu/kernels/reduce.zig");
            pub const softmax = @import("backend/cpu/kernels/softmax.zig");
            pub const layernorm = @import("backend/cpu/kernels/layernorm.zig");
            pub const gather = @import("backend/cpu/kernels/gather.zig");
            pub const transpose = @import("backend/cpu/kernels/transpose.zig");
            pub const conv2d = @import("backend/cpu/kernels/conv2d.zig");
            pub const maxpool = @import("backend/cpu/kernels/maxpool.zig");
        };
    };
};

// I/O and file formats
pub const io = struct {
    pub const safetensors = @import("io/safetensors.zig");
    pub const mnist = @import("io/mnist.zig");
};

// Model implementations
pub const model = struct {
    pub const arctic = @import("model/arctic.zig");
    pub const lenet = @import("model/lenet.zig");
};

// Neural network training utilities
pub const nn = struct {
    pub const loss = @import("nn/loss.zig");
    pub const optim = @import("nn/optim.zig");
    pub const init = @import("nn/init.zig");
};

// Fusion engine
pub const fusion = struct {
    pub const analyzer = @import("fusion/analyzer.zig");
    pub const patterns = @import("fusion/patterns.zig");
    pub const codegen = @import("fusion/codegen.zig");
};

// Memory management
pub const memory = struct {
    pub const allocator_mod = @import("memory/allocator.zig");
    pub const pool = @import("memory/pool.zig");
    pub const TensorAllocator = allocator_mod.TensorAllocator;
    pub const ComputeArena = allocator_mod.ComputeArena;
    pub const BufferPool = pool.BufferPool;
};

// Re-export commonly used types at top level
pub const DType = core.dtype.DType;
pub const Shape = core.shape.Shape;
pub const Tensor = core.tensor.Tensor;

// Expression types
pub const OpTag = ops.expr.OpTag;
pub const UnaryExpr = ops.expr.UnaryExpr;
pub const BinaryExpr = ops.expr.BinaryExpr;
pub const MatmulExpr = ops.expr.MatmulExpr;
pub const ReduceExpr = ops.expr.ReduceExpr;
pub const ScalarExpr = ops.expr.ScalarExpr;

// Executor functions
pub const eval = backend.cpu.executor.eval;
pub const evalInto = backend.cpu.executor.evalInto;

// ============================================================================
// Tests - import all test modules
// ============================================================================

test {
    // Core tests
    _ = core.dtype;
    _ = core.shape;
    _ = core.strides;
    _ = core.tensor;

    // Ops tests
    _ = ops.expr;

    // Backend tests
    _ = backend.cpu.simd;
    _ = backend.cpu.executor;
    _ = backend.cpu.threading;
    _ = backend.cpu.kernels.elementwise;
    _ = backend.cpu.kernels.matmul;
    _ = backend.cpu.kernels.reduce;
    _ = backend.cpu.kernels.softmax;
    _ = backend.cpu.kernels.layernorm;
    _ = backend.cpu.kernels.gather;
    _ = backend.cpu.kernels.transpose;
    _ = backend.cpu.kernels.conv2d;
    _ = backend.cpu.kernels.maxpool;

    // I/O tests
    _ = io.safetensors;
    _ = io.mnist;

    // Model tests
    _ = model.arctic;
    _ = model.lenet;

    // Neural network training tests
    _ = nn.loss;
    _ = nn.optim;
    _ = nn.init;

    // Integration tests
    _ = @import("tests/arctic_integration_test.zig");
    _ = @import("tests/lenet_test.zig");

    // Fusion tests
    _ = fusion.analyzer;
    _ = fusion.patterns;
    _ = fusion.codegen;

    // Memory tests
    _ = memory.allocator_mod;
    _ = memory.pool;
}

// ============================================================================
// Basic usage example (comptime verification)
// ============================================================================

test "basic tensor creation" {
    const std = @import("std");

    const Vec = Tensor(f32, .{4});
    var vec = try Vec.zeros(std.testing.allocator);
    defer vec.deinit();

    vec.set(.{0}, 1.0);
    vec.set(.{1}, 2.0);
    vec.set(.{2}, 3.0);
    vec.set(.{3}, 4.0);

    try std.testing.expectEqual(@as(f32, 1.0), vec.get(.{0}));
    try std.testing.expectEqual(@as(f32, 10.0), vec.get(.{0}) + vec.get(.{1}) + vec.get(.{2}) + vec.get(.{3}));
}

test "expression type chaining" {
    // This tests that expression types chain correctly at comptime
    const Vec = Tensor(f32, .{4});
    const Mat = Tensor(f32, .{ 3, 4 });

    // Unary chain
    const E1 = UnaryExpr(.relu, Vec);
    const E2 = UnaryExpr(.exp, E1);
    try @import("std").testing.expectEqual(@as(usize, 1), E2.ndim);

    // Binary with broadcasting
    const E3 = BinaryExpr(.add, Mat, Vec);
    try @import("std").testing.expectEqual(@as(usize, 2), E3.ndim);

    // Matmul
    const A = Tensor(f32, .{ 3, 4 });
    const B = Tensor(f32, .{ 4, 5 });
    const E4 = MatmulExpr(A, B);
    try @import("std").testing.expectEqual(@as(usize, 3), E4.shape[0]);
    try @import("std").testing.expectEqual(@as(usize, 5), E4.shape[1]);
}

test "matmul followed by activation" {
    // Common pattern: matmul -> bias -> relu
    const A = Tensor(f32, .{ 64, 128 });
    const B = Tensor(f32, .{ 128, 256 });
    const Bias = Tensor(f32, .{256});

    // Type chain: Matmul -> Add(bias) -> Relu
    const MM = MatmulExpr(A, B);
    const WithBias = BinaryExpr(.add, MM, Bias);
    const Activated = UnaryExpr(.relu, WithBias);

    try @import("std").testing.expectEqual(@as(usize, 2), Activated.ndim);
    try @import("std").testing.expectEqual(@as(usize, 64), Activated.shape[0]);
    try @import("std").testing.expectEqual(@as(usize, 256), Activated.shape[1]);
}

test "end-to-end: build expression, evaluate, get result" {
    const std = @import("std");
    const allocator = std.testing.allocator;

    // Create input tensors
    const Vec = Tensor(f32, .{4});
    var a = try Vec.fromSlice(allocator, &[_]f32{ -2, -1, 1, 2 });
    defer a.deinit();
    var b = try Vec.fromSlice(allocator, &[_]f32{ 1, 1, 1, 1 });
    defer b.deinit();

    // Build expression: relu(a + b)
    // a + b = [-1, 0, 2, 3]
    // relu = [0, 0, 2, 3]
    const AddExpr = BinaryExpr(.add, Vec, Vec);
    const ReluExpr = UnaryExpr(.relu, AddExpr);

    const expr = ReluExpr.init(AddExpr.init(a, b));

    // Evaluate
    var result = try eval(ReluExpr, expr, allocator);
    defer result.deinit();

    // Check result
    try std.testing.expectEqual(@as(f32, 0), result.get(.{0}));
    try std.testing.expectEqual(@as(f32, 0), result.get(.{1}));
    try std.testing.expectEqual(@as(f32, 2), result.get(.{2}));
    try std.testing.expectEqual(@as(f32, 3), result.get(.{3}));
}
