# Reduce Epilogues

Elementwise operations before or after reductions can be fused.

## Patterns

### Pre-Reduction Fusion

Elementwise ops before reduction:

```zig
// Square and sum (L2 norm squared)
const l2_sq = x.mul(x).sum(.{});

// Fused: square during accumulation
// for each element:
//   acc += x[i] * x[i]
```

### Post-Reduction Fusion

Elementwise ops after reduction:

```zig
// Mean then scale
const scaled_mean = x.mean(.{}).mul(scale);

// Fused: scale applied to reduction result
```

## Common Use Cases

### Variance

```zig
const mean = x.mean(.{1});
const centered = x.sub(mean);
const variance = centered.mul(centered).mean(.{1});

// The mul + mean chain is fused
```

### Softmax

```zig
const max_x = x.reduce_max(.{1});
const shifted = x.sub(max_x);
const exp_x = shifted.exp();
const sum_exp = exp_x.sum(.{1});
const probs = exp_x.div(sum_exp);

// exp + sum is fused into single pass
```

### L2 Norm

```zig
const squared = x.mul(x);
const sum_sq = squared.sum(.{});
const l2_norm = sum_sq.sqrt();

// mul + sum + sqrt chain
```

## Fusion Info

```zig
pub const ReduceFusionInfo = struct {
    pre_ops: [MAX_CHAIN_LENGTH]OpTag,  // Ops before reduce
    pre_len: usize,
    reduce_op: OpTag,                   // The reduction
    post_ops: [MAX_CHAIN_LENGTH]OpTag, // Ops after reduce
    post_len: usize,
};
```

## Implementation

### Pre-Reduction Fusion

```zig
pub fn fusedReduceWithPreOps(
    comptime pre_ops: []const OpTag,
    comptime reduce_op: OpTag,
    input: []const f32,
) f32 {
    const vec_len = simd.suggestVectorLength(f32);
    var acc = initAccumulator(reduce_op);

    var i: usize = 0;
    while (i + vec_len <= input.len) : (i += vec_len) {
        var v = simd.load(f32, input[i..]);

        // Apply pre-reduction ops
        inline for (pre_ops) |op| {
            v = simd.applyUnary(op, f32, v);
        }

        // Accumulate
        acc = simd.accumulate(reduce_op, acc, v);
    }

    // Handle remainder and finalize
    return finalize(reduce_op, acc, input.len);
}
```

### Post-Reduction Fusion

```zig
pub fn fusedReduceWithPostOps(
    comptime reduce_op: OpTag,
    comptime post_ops: []const OpTag,
    input: []const f32,
) f32 {
    // First, do the reduction
    var result = reduce(reduce_op, input);

    // Then apply post-ops
    inline for (post_ops) |op| {
        result = scalar.applyUnary(op, result);
    }

    return result;
}
```

## Performance

For sum-of-squares on 1M elements:

| Version | Time | Memory Reads |
|---------|------|--------------|
| Unfused (mul then sum) | 2.1ms | 8MB |
| Fused | 1.2ms | 4MB |
| Speedup | 1.75x | 2x |

## Limitations

Not all combinations can be fused:

```zig
// Cannot fuse: different reduction axes
const row_sum = x.sum(.{1});
const col_sum = x.sum(.{0});

// Cannot fuse: reduction breaks parallel structure
const mean = x.mean(.{});
const centered = x.sub(mean);  // Needs mean completed first
```

## Next Steps

- [Code Generation](./codegen.md) - Generating fused kernels
- [Reductions](../operations/reductions.md) - Reduction operations
