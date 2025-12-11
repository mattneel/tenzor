# Elementwise Chains

Chains of elementwise operations are fused into single kernels.

## What Gets Fused

Sequential unary and binary elementwise operations:

```zig
// This chain:
const expr = x.exp().mul(scale).add(shift).relu();

// Becomes single kernel:
// for each element:
//   result[i] = relu(exp(x[i]) * scale[i] + shift[i])
```

## Chain Detection

The analyzer walks the expression tree:

```
Input: x.exp().mul(y).relu()

Tree:
    UnaryExpr(.relu)
         │
    BinaryExpr(.mul)
        /    \
  UnaryExpr   y
    (.exp)
      │
      x

Chain: [.exp, .mul, .relu]
```

## Fusion Benefits

| Unfused | Fused |
|---------|-------|
| 3 kernel launches | 1 kernel launch |
| 3 memory writes | 1 memory write |
| 2 temporary buffers | 0 temporary buffers |

## Implementation

### Chain Info Structure

```zig
pub const ElementwiseFusionInfo = struct {
    ops: [MAX_CHAIN_LENGTH]OpTag,
    len: usize,

    pub fn getOps(self: *const Self) []const OpTag {
        return self.ops[0..self.len];
    }
};
```

### Fused Kernel

```zig
pub fn FusedElementwiseKernel(comptime info: ElementwiseFusionInfo) type {
    return struct {
        pub fn execute(input: []const f32, output: []f32) void {
            const vec_len = simd.suggestVectorLength(f32);
            const ops = info.getOps();

            var i: usize = 0;
            while (i + vec_len <= input.len) : (i += vec_len) {
                var v = simd.load(f32, input[i..]);

                // Apply all operations in sequence
                inline for (ops) |op| {
                    v = simd.applyUnary(op, f32, v);
                }

                simd.store(f32, v, output[i..]);
            }

            // Scalar remainder
            while (i < input.len) : (i += 1) {
                var x = input[i];
                inline for (ops) |op| {
                    x = scalar.applyUnary(op, x);
                }
                output[i] = x;
            }
        }
    };
}
```

## Chain Length Limits

Maximum chain length is bounded:

```zig
const MAX_CHAIN_LENGTH = 8;
```

Reasons:
- Register pressure increases with chain length
- Compile time increases
- Diminishing returns beyond ~8 operations

## Binary Operations in Chains

Binary operations are included when one operand continues the chain:

```zig
// x.exp().mul(y) where y is a tensor
// Chain includes: [.exp, .mul]
// y is loaded separately and combined with the chain
```

## Example: Softmax Numerator

```zig
// Numerator of softmax: exp(x - max(x))
const max_x = x.reduce_max(.{});
const shifted = x.sub(max_x);
const numerator = shifted.exp();

// The sub + exp chain is fused
// But reduce_max breaks the chain (different operation type)
```

## Performance Comparison

For `x.exp().mul(y).add(z).relu()` on 1M elements:

| Metric | Unfused | Fused | Speedup |
|--------|---------|-------|---------|
| Time | 4.2ms | 1.5ms | 2.8x |
| Memory | 16MB | 8MB | 2x |
| L3 misses | 12K | 4K | 3x |

## Next Steps

- [Matmul Epilogues](./matmul-epilogues.md) - Fusing with matmul
- [Code Generation](./codegen.md) - Kernel generation
