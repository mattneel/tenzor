# Fusion Engine Overview

The fusion engine optimizes expression graphs by combining multiple operations into single kernels.

## Why Fusion Matters

Without fusion, each operation:
1. Reads from memory
2. Computes
3. Writes to memory

```
// Without fusion: 3 memory round-trips
temp1 = matmul(x, w)     // Read x,w → Compute → Write temp1
temp2 = add(temp1, b)    // Read temp1,b → Compute → Write temp2
output = relu(temp2)     // Read temp2 → Compute → Write output
```

With fusion, operations share data in registers:

```
// With fusion: 1 memory round-trip
output = fused_matmul_add_relu(x, w, b)  // Read x,w,b → Compute all → Write output
```

## Fusion Patterns

Tenzor detects and optimizes these patterns:

### Elementwise Chains

Sequential unary/binary operations:

```zig
// Detected pattern
const expr = x.exp().mul(y).add(z).relu();

// Fused into single kernel that:
// 1. Loads x, y, z
// 2. Computes exp(x) * y + z
// 3. Applies relu
// 4. Stores result
```

### Matmul Epilogues

Operations following matrix multiplication:

```zig
// Detected pattern
const expr = x.matmul(w).add(bias).gelu();

// Fused: matmul computes tiles, then immediately:
// - Adds bias
// - Applies activation
// Before writing to memory
```

### Reduce Epilogues

Operations before or after reductions:

```zig
// Detected pattern
const expr = x.exp().sum(.{1});

// Fused: exp and sum in single pass
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Expression Type                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Analyzer                                  │
│   - Pattern matching                                         │
│   - Chain detection                                          │
│   - Fusion plan generation                                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Codegen                                   │
│   - Fused kernel generation                                  │
│   - SIMD optimization                                        │
└─────────────────────────────────────────────────────────────┘
```

## Compile-Time Analysis

Fusion happens at compile time:

```zig
const expr = input.matmul(weights).add(bias).relu();

comptime {
    const Expr = @TypeOf(expr);
    const plan = fusion.analyzer.analyze(Expr);

    // plan.pattern == .matmul_epilogue
    // plan.matmul_epilogue contains:
    //   - has_bias: true
    //   - activation: .relu
}
```

## FusionPlan

The analyzer produces a fusion plan:

```zig
pub const FusionPlan = struct {
    pattern: FusionPattern,

    // For elementwise chains
    elementwise_chain: [MAX_CHAIN_LENGTH]OpTag,
    chain_length: usize,

    // For matmul epilogues
    matmul_epilogue: MatmulEpilogueInfo,
};

pub const FusionPattern = enum {
    single,              // No fusion opportunity
    elementwise_chain,   // Chain of elementwise ops
    matmul_epilogue,     // Matmul + bias + activation
    reduce_epilogue,     // Elementwise + reduce
};
```

## Benefits

| Metric | Without Fusion | With Fusion |
|--------|---------------|-------------|
| Memory bandwidth | 3x | 1x |
| Cache misses | High | Low |
| Kernel launches | 3 | 1 |
| Register pressure | Low | Moderate |

## Limitations

Fusion is not always possible:
- Complex data dependencies
- Operations with different parallelism
- Very long chains (register pressure)

## Next Steps

- [Pattern Detection](./patterns.md) - How patterns are detected
- [Code Generation](./codegen.md) - Fused kernel generation
