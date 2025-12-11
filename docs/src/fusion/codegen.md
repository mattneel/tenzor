# Code Generation

The fusion engine generates optimized kernels at compile time.

## Overview

```
FusionPlan → Kernel Type → Instantiated Code
```

All code generation happens at compile time using Zig's comptime features.

## Kernel Generation

### Elementwise Chain Kernel

```zig
pub fn FusedElementwiseKernel(comptime info: ElementwiseFusionInfo) type {
    const ops = info.ops[0..info.len];

    return struct {
        pub fn execute(
            comptime T: type,
            input: []const T,
            output: []T,
        ) void {
            const vec_len = simd.suggestVectorLength(T);
            var i: usize = 0;

            // SIMD loop with all ops inlined
            while (i + vec_len <= input.len) : (i += vec_len) {
                var v = simd.load(T, input[i..]);

                // Unroll all operations at compile time
                inline for (ops) |op| {
                    v = switch (op) {
                        .neg => simd.neg(T, v),
                        .exp => simd.exp(T, v),
                        .relu => simd.relu(T, v),
                        .sigmoid => simd.sigmoid(T, v),
                        // ... etc
                    };
                }

                simd.store(T, v, output[i..]);
            }

            // Scalar remainder
            while (i < input.len) : (i += 1) {
                var x = input[i];
                inline for (ops) |op| {
                    x = applyScalar(op, x);
                }
                output[i] = x;
            }
        }
    };
}
```

### Matmul Epilogue Kernel

```zig
pub fn FusedMatmulEpilogueKernel(comptime info: MatmulEpilogueInfo) type {
    return struct {
        pub fn execute(
            comptime T: type,
            comptime M: usize,
            comptime K: usize,
            comptime N: usize,
            a: *const [M * K]T,
            b: *const [K * N]T,
            c: *[M * N]T,
            bias: if (info.has_bias) *const [N]T else void,
        ) void {
            const vec_len = simd.suggestVectorLength(T);

            for (0..M) |i| {
                var j: usize = 0;
                while (j + vec_len <= N) : (j += vec_len) {
                    // Compute matmul tile
                    var acc: @Vector(vec_len, T) = @splat(0);
                    for (0..K) |k| {
                        const a_val: @Vector(vec_len, T) = @splat(a[i * K + k]);
                        const b_vec = simd.load(T, b[k * N + j..]);
                        acc += a_val * b_vec;
                    }

                    // Fused bias add
                    if (info.has_bias) {
                        const bias_vec = simd.load(T, bias[j..]);
                        acc = acc + bias_vec;
                    }

                    // Fused activation
                    if (info.activation) |act| {
                        acc = switch (act) {
                            .relu => simd.relu(T, acc),
                            .gelu => simd.gelu(T, acc),
                            .sigmoid => simd.sigmoid(T, acc),
                            else => acc,
                        };
                    }

                    simd.store(T, acc, c[i * N + j..]);
                }
            }
        }
    };
}
```

## Compile-Time Specialization

Kernels are specialized for exact configurations:

```zig
// At compile time, this becomes a specialized function
const Kernel = FusedElementwiseKernel(.{
    .ops = .{ .exp, .mul, .relu },
    .len = 3,
});

// Equivalent to writing by hand:
fn specialized_exp_mul_relu(input: []const f32, aux: []const f32, output: []f32) void {
    for (input, aux, output) |x, y, *out| {
        out.* = @max(0, @exp(x) * y);
    }
}
```

## Inline Expansion

The `inline for` ensures no loop overhead:

```zig
// This compile-time loop:
inline for (ops) |op| {
    v = applyOp(op, v);
}

// Becomes (for ops = [.exp, .relu]):
v = simd.exp(T, v);
v = simd.relu(T, v);
```

## Kernel Selection

The executor selects kernels based on fusion plan:

```zig
fn executeWithFusion(comptime Expr: type, expr: Expr, output: []Expr.ElementType) void {
    const plan = comptime fusion.analyzer.analyze(Expr);

    switch (plan.pattern) {
        .elementwise_chain => {
            const Kernel = FusedElementwiseKernel(plan.elementwise_chain);
            Kernel.execute(Expr.ElementType, getInput(expr), output);
        },
        .matmul_epilogue => {
            const Kernel = FusedMatmulEpilogueKernel(plan.matmul_epilogue);
            Kernel.execute(...);
        },
        .single => {
            // Fall back to standard execution
            executeStandard(Expr, expr, output);
        },
    }
}
```

## Generated Code Quality

The generated code is as efficient as hand-written:

1. **No runtime dispatch** - All decisions made at compile time
2. **Full inlining** - No function call overhead
3. **SIMD optimization** - Vector operations used throughout
4. **No allocations** - All buffers pre-sized

## Viewing Generated Code

Use `@compileLog` or check assembly:

```zig
comptime {
    const info = ElementwiseFusionInfo{ .ops = .{ .exp, .relu }, .len = 2 };
    const Kernel = FusedElementwiseKernel(info);
    @compileLog("Generated kernel type:", @typeName(Kernel));
}
```

Or compile with `-femit-asm` to inspect assembly.

## Next Steps

- [Architecture](../backend/architecture.md) - Backend overview
- [SIMD Optimization](../backend/simd.md) - Vectorization details
