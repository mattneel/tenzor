# Custom Operations

Extend Tenzor with your own operations.

## Adding Unary Operations

### 1. Define the Operation Tag

```zig
// In ops/unary.zig
pub const UnaryOpTag = enum {
    neg,
    exp,
    log,
    sqrt,
    sin,
    cos,
    tanh,
    relu,
    sigmoid,
    gelu,      // Add new op
    softplus,  // Add new op
};
```

### 2. Implement the Scalar Function

```zig
pub fn applyUnary(comptime op: UnaryOpTag, x: anytype) @TypeOf(x) {
    return switch (op) {
        // ... existing ops ...
        .gelu => blk: {
            // GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            const sqrt_2_pi = 0.7978845608;
            const coeff = 0.044715;
            const x3 = x * x * x;
            const inner = sqrt_2_pi * (x + coeff * x3);
            break :blk x * 0.5 * (1.0 + @tanh(inner));
        },
        .softplus => @log(1.0 + @exp(x)),
    };
}
```

### 3. Add SIMD Kernel (Optional)

```zig
// In backend/cpu/simd_kernels.zig
pub fn geluSimd(comptime T: type, comptime len: comptime_int, v: @Vector(len, T)) @Vector(len, T) {
    const sqrt_2_pi: @Vector(len, T) = @splat(0.7978845608);
    const coeff: @Vector(len, T) = @splat(0.044715);
    const half: @Vector(len, T) = @splat(0.5);
    const one: @Vector(len, T) = @splat(1.0);

    const x3 = v * v * v;
    const inner = sqrt_2_pi * (v + coeff * x3);
    const tanh_inner = tanhSimd(T, len, inner);
    return v * half * (one + tanh_inner);
}
```

### 4. Register in Dispatch

```zig
// In backend/cpu/dispatch.zig
fn dispatchUnary(comptime op: UnaryOpTag, ...) void {
    switch (op) {
        // ... existing ops ...
        .gelu => geluKernel(input, output),
        .softplus => softplusKernel(input, output),
    }
}
```

### 5. Add Expression Method

```zig
// In tensor.zig or expression types
pub fn gelu(self: Self) UnaryExpr(.gelu, Self) {
    return .{ .input = self };
}
```

## Adding Binary Operations

### 1. Define Operation Tag

```zig
pub const BinaryOpTag = enum {
    add,
    sub,
    mul,
    div,
    pow,
    max,
    min,
    // Add new ops
    atan2,
    copysign,
};
```

### 2. Implement Scalar Function

```zig
pub fn applyBinary(comptime op: BinaryOpTag, a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    return switch (op) {
        // ... existing ops ...
        .atan2 => std.math.atan2(a, b),
        .copysign => @copysign(a, b),
    };
}
```

### 3. Add Methods

```zig
pub fn atan2(self: Self, other: anytype) BinaryExpr(.atan2, Self, @TypeOf(other)) {
    return .{ .lhs = self, .rhs = other };
}
```

## Adding Reduction Operations

### 1. Define Operation Tag

```zig
pub const ReduceOpTag = enum {
    sum,
    prod,
    max,
    min,
    mean,
    // Add new ops
    variance,
    l2_norm,
};
```

### 2. Implement Reduction Logic

```zig
pub fn applyReduce(
    comptime op: ReduceOpTag,
    comptime T: type,
    data: []const T,
) T {
    return switch (op) {
        // ... existing ops ...
        .variance => blk: {
            var sum: T = 0;
            var sum_sq: T = 0;
            for (data) |x| {
                sum += x;
                sum_sq += x * x;
            }
            const n: T = @floatFromInt(data.len);
            const mean = sum / n;
            break :blk sum_sq / n - mean * mean;
        },
        .l2_norm => blk: {
            var sum_sq: T = 0;
            for (data) |x| {
                sum_sq += x * x;
            }
            break :blk @sqrt(sum_sq);
        },
    };
}
```

## Custom Expression Types

Create entirely new expression node types:

```zig
pub fn ConvExpr(
    comptime Input: type,
    comptime Kernel: type,
    comptime padding: [2]usize,
    comptime stride: [2]usize,
) type {
    return struct {
        pub const kind = .conv2d;
        pub const ElementType = Input.ElementType;
        pub const shape = computeConvShape(Input.shape, Kernel.shape, padding, stride);

        input: Input,
        kernel: Kernel,

        pub fn eval(self: @This(), allocator: std.mem.Allocator) !ResultType {
            // Implement convolution
        }
    };
}
```

## Custom Kernels

### Register Custom Kernel

```zig
pub fn registerKernel(
    comptime name: []const u8,
    comptime kernel_fn: anytype,
) void {
    // Custom kernel registration
}

// Usage
registerKernel("my_custom_op", struct {
    fn kernel(input: []const f32, output: []f32) void {
        for (input, output) |in, *out| {
            out.* = customComputation(in);
        }
    }
}.kernel);
```

### Fused Kernel

```zig
pub fn fusedGeluAdd(
    comptime T: type,
    a: []const T,
    b: []const T,
    output: []T,
) void {
    const vec_len = simd.suggestVectorLength(T);
    var i: usize = 0;

    while (i + vec_len <= a.len) : (i += vec_len) {
        const va = simd.load(T, a[i..]);
        const vb = simd.load(T, b[i..]);
        const gelu_a = geluSimd(T, vec_len, va);
        const result = gelu_a + vb;
        simd.store(T, result, output[i..]);
    }

    // Scalar remainder
    while (i < a.len) : (i += 1) {
        output[i] = applyUnary(.gelu, a[i]) + b[i];
    }
}
```

## Testing Custom Operations

```zig
test "custom gelu operation" {
    const Tensor = @import("tenzor").Tensor;

    const T = Tensor(f32, .{4});
    var input = T{};
    input.data = .{ -1.0, 0.0, 1.0, 2.0 };

    const result = input.gelu().eval(std.testing.allocator);
    defer result.deinit();

    // GELU(-1) ≈ -0.158
    // GELU(0) = 0
    // GELU(1) ≈ 0.841
    // GELU(2) ≈ 1.954
    try std.testing.expectApproxEqAbs(-0.158, result.data[0], 0.01);
    try std.testing.expectApproxEqAbs(0.0, result.data[1], 0.01);
    try std.testing.expectApproxEqAbs(0.841, result.data[2], 0.01);
}
```

## Best Practices

### 1. Follow Existing Patterns

```zig
// Match existing function signatures
pub fn myOp(self: Self) UnaryExpr(.my_op, Self) {
    return .{ .input = self };
}
```

### 2. Provide SIMD Implementations

```zig
// Scalar fallback + SIMD fast path
if (comptime simd.isSupported(T)) {
    myOpSimd(input, output);
} else {
    myOpScalar(input, output);
}
```

### 3. Validate at Compile Time

```zig
comptime {
    if (!isValidForMyOp(Input)) {
        @compileError("Invalid input type for myOp");
    }
}
```

### 4. Document Shape Requirements

```zig
/// Applies myOp element-wise.
///
/// Input: Any shape
/// Output: Same shape as input
pub fn myOp(self: Self) ...
```

## Next Steps

- [Broadcasting](./broadcasting.md) - Broadcasting rules
- [Shape Algebra](./shape-algebra.md) - Shape computations
