# Reductions

Reductions aggregate values along specified dimensions.

## Available Reductions

| Method | Description |
|--------|-------------|
| `.sum(axes)` | Sum of elements |
| `.mean(axes)` | Mean of elements |
| `.prod(axes)` | Product of elements |
| `.reduce_max(axes)` | Maximum element |
| `.reduce_min(axes)` | Minimum element |
| `.variance(axes)` | Variance of elements |

## Axis Specification

### Full Reduction

Empty axes reduces all dimensions to a scalar:

```zig
const A = tz.Tensor(f32, .{ 3, 4 });
var a = A{};

const total = a.sum(.{});  // Shape: {} (scalar)

comptime {
    std.debug.assert(total.ndim == 0);
    std.debug.assert(total.numel() == 1);
}
```

### Single Axis

Reduce along one dimension:

```zig
const A = tz.Tensor(f32, .{ 3, 4 });
var a = A{};

const row_sum = a.sum(.{0});   // Shape: { 4 } - sum along rows
const col_sum = a.sum(.{1});   // Shape: { 3 } - sum along columns
```

### Multiple Axes

Reduce along multiple dimensions:

```zig
const A = tz.Tensor(f32, .{ 2, 3, 4 });
var a = A{};

const partial = a.sum(.{ 0, 2 });  // Shape: { 3 }
```

## Reduction Operations

### sum() - Sum

Add all elements along specified axes.

```zig
const s = x.sum(.{});       // Total sum
const row_s = x.sum(.{0});  // Sum each column
const col_s = x.sum(.{1});  // Sum each row
```

**Use cases:**
- Loss aggregation
- Probability normalization denominators
- Counting (with masks)

### mean() - Mean

Compute average along specified axes.

```zig
const m = x.mean(.{});      // Global mean
const row_m = x.mean(.{0}); // Mean of each column
```

Equivalent to: `sum(x) / count`

**Use cases:**
- Batch normalization
- Average pooling
- Mean squared error

### prod() - Product

Multiply all elements along specified axes.

```zig
const p = x.prod(.{});      // Total product
```

**Use cases:**
- Probability chains (in log space: sum of logs)
- Determinant computation

### reduce_max() - Maximum

Find maximum element along specified axes.

```zig
const m = x.reduce_max(.{});      // Global maximum
const row_m = x.reduce_max(.{0}); // Max of each column
```

**Use cases:**
- Softmax stability (subtract max)
- Max pooling
- Argmax approximation

### reduce_min() - Minimum

Find minimum element along specified axes.

```zig
const m = x.reduce_min(.{});      // Global minimum
const row_m = x.reduce_min(.{0}); // Min of each column
```

**Use cases:**
- Clipping bounds
- Min pooling

### variance() - Variance

Compute variance along specified axes.

```zig
const v = x.variance(.{});      // Global variance
const row_v = x.variance(.{0}); // Variance of each column
```

Computed as: `mean((x - mean(x))^2)`

**Use cases:**
- Normalization
- Statistical analysis

## Shape Transformation

Reductions change tensor shape:

```zig
const A = tz.Tensor(f32, .{ 2, 3, 4 });

// Full reduction
const full = a.sum(.{});           // Shape: {}

// Single axis
const axis_0 = a.sum(.{0});        // Shape: { 3, 4 }
const axis_1 = a.sum(.{1});        // Shape: { 2, 4 }
const axis_2 = a.sum(.{2});        // Shape: { 2, 3 }

// Multiple axes
const axes_01 = a.sum(.{ 0, 1 });  // Shape: { 4 }
const axes_12 = a.sum(.{ 1, 2 });  // Shape: { 2 }
```

## Common Patterns

### Softmax

```zig
const max_x = x.reduce_max(.{1});   // Max per row
const shifted = x.sub(max_x);        // Numerical stability
const exp_x = shifted.exp();
const sum_exp = exp_x.sum(.{1});    // Sum per row
const softmax = exp_x.div(sum_exp); // Normalize
```

### Layer Normalization

```zig
const mean = x.mean(.{-1});           // Mean over last dim
const centered = x.sub(mean);
const variance = centered.mul(centered).mean(.{-1});
const std = variance.add(epsilon).sqrt();
const normalized = centered.div(std);
const output = normalized.mul(gamma).add(beta);
```

### Mean Squared Error

```zig
const diff = predicted.sub(target);
const squared = diff.mul(diff);
const mse = squared.mean(.{});
```

### Cross-Entropy Loss

```zig
// log_probs: log probabilities, targets: one-hot
const ce = log_probs.mul(targets).sum(.{1}).neg().mean(.{});
```

## SIMD Implementation

Reductions use SIMD with tree reduction:

```zig
// Conceptual: reducing 8 elements
const vec = simd.load(f32, data[0..8]);
const sum = @reduce(.Add, vec);  // Hardware-accelerated
```

For longer arrays, combine SIMD reduction with scalar accumulation.

## Chaining After Reduction

Reduction results support further operations:

```zig
const loss = error.mul(error)  // Square
    .sum(.{})                   // Sum
    .sqrt();                    // Square root of sum

const normalized = x.div(x.sum(.{1}));  // Normalize rows
```

## Next Steps

- [Fusion Engine](../fusion/overview.md) - Reduce epilogue fusion
- [SIMD Optimization](../backend/simd.md) - Vectorized reductions
