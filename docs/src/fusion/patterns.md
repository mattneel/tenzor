# Pattern Detection

The fusion analyzer examines expression types to detect optimization opportunities.

## Pattern Types

### Single Operation

No fusion possible or needed:

```zig
const expr = tensor.relu();  // Single operation
// Pattern: .single
```

### Elementwise Chain

Sequential elementwise operations:

```zig
const expr = x.exp().mul(y).relu();
// Pattern: .elementwise_chain
// Chain: [.exp, .mul, .relu]
```

### Matmul Epilogue

Matmul followed by bias and/or activation:

```zig
const expr = x.matmul(w).add(bias).gelu();
// Pattern: .matmul_epilogue
// Epilogue: { has_bias: true, activation: .gelu }
```

### Reduce Epilogue

Elementwise followed by reduction:

```zig
const expr = x.exp().sum(.{1});
// Pattern: .reduce_epilogue
```

## Detection Algorithm

### Elementwise Chain Detection

```zig
fn collectElementwiseChain(comptime Expr: type) ChainResult {
    var ops: [MAX_CHAIN_LENGTH]OpTag = undefined;
    var len: usize = 0;

    // Walk the expression tree
    var current = Expr;
    while (true) {
        if (current.kind == .unary and current.operation.isElementwise()) {
            ops[len] = current.operation;
            len += 1;
            current = current.InputType;
        } else if (current.kind == .binary and current.operation.isElementwise()) {
            ops[len] = current.operation;
            len += 1;
            // Continue with one branch
            current = current.LhsType;
        } else {
            break;
        }
    }

    return .{ .ops = ops, .len = len };
}
```

### Matmul Epilogue Detection

```zig
fn hasMatmulInChain(comptime Expr: type) bool {
    // Check if matmul exists in the expression tree
    if (Expr.kind == .matmul) return true;

    if (Expr.kind == .unary) {
        return hasMatmulInChain(Expr.InputType);
    }

    if (Expr.kind == .binary) {
        return hasMatmulInChain(Expr.LhsType) or
               hasMatmulInChain(Expr.RhsType);
    }

    return false;
}

fn collectEpilogueOps(comptime Expr: type) EpilogueResult {
    var has_bias = false;
    var activation: ?OpTag = null;

    // Check for bias (binary add after matmul)
    if (Expr.kind == .binary and Expr.operation == .add) {
        if (hasMatmulInChain(Expr.LhsType)) {
            has_bias = true;
        }
    }

    // Check for activation
    if (Expr.kind == .unary and Expr.operation.isActivation()) {
        activation = Expr.operation;
    }

    return .{ .has_bias = has_bias, .activation = activation };
}
```

## Pattern Priority

When multiple patterns match, priority determines selection:

1. **matmul_epilogue** - Highest impact
2. **reduce_epilogue** - High impact
3. **elementwise_chain** - Moderate impact
4. **single** - No fusion

## Analysis Output

```zig
pub fn analyze(comptime Expr: type) FusionPlan {
    // Check for matmul epilogue pattern
    if (hasMatmulInChain(Expr)) {
        const epilogue = collectEpilogueOps(Expr);
        if (epilogue.has_bias or epilogue.activation != null) {
            return .{
                .pattern = .matmul_epilogue,
                .matmul_epilogue = epilogue,
            };
        }
    }

    // Check for elementwise chain
    const chain = collectElementwiseChain(Expr);
    if (chain.len > 1) {
        return .{
            .pattern = .elementwise_chain,
            .elementwise_chain = chain.ops,
            .chain_length = chain.len,
        };
    }

    // Default: single operation
    return .{ .pattern = .single };
}
```

## Example Analysis

```zig
// Expression: input.matmul(weights).add(bias).relu()

// Type structure:
// UnaryExpr(.relu,
//   BinaryExpr(.add,
//     MatmulExpr(input, weights),
//     bias
//   )
// )

// Analysis:
// 1. hasMatmulInChain → true (found MatmulExpr)
// 2. collectEpilogueOps:
//    - BinaryExpr(.add) after matmul → has_bias = true
//    - UnaryExpr(.relu) is activation → activation = .relu
// 3. Result: .matmul_epilogue with { has_bias: true, activation: .relu }
```

## Next Steps

- [Elementwise Chains](./elementwise-chains.md) - Chain fusion details
- [Matmul Epilogues](./matmul-epilogues.md) - Matmul fusion details
- [Code Generation](./codegen.md) - Generating fused kernels
