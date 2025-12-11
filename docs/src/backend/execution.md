# Execution

The executor transforms expression graphs into computed results.

## Execution Model

```
Expression Graph (Types)
         │
         ▼
    ┌─────────┐
    │  Analyze │ ← Compile-time
    └─────────┘
         │
         ▼
    ┌─────────┐
    │ Dispatch │ ← Runtime
    └─────────┘
         │
         ▼
    ┌─────────┐
    │ Execute  │
    └─────────┘
         │
         ▼
      Result
```

## The Executor

```zig
// src/backend/cpu/executor.zig
pub fn eval(comptime Expr: type, expr: Expr, allocator: std.mem.Allocator) ![]Expr.ElementType {
    const result = try allocator.alloc(Expr.ElementType, Expr.numel());
    errdefer allocator.free(result);

    try evalInto(Expr, expr, result);
    return result;
}

pub fn evalInto(comptime Expr: type, expr: Expr, output: []Expr.ElementType) !void {
    // Dispatch based on expression type
    switch (Expr.kind) {
        .tensor => evalTensor(expr, output),
        .unary => evalUnary(Expr, expr, output),
        .binary => evalBinary(Expr, expr, output),
        .matmul => evalMatmul(Expr, expr, output),
        .reduce => evalReduce(Expr, expr, output),
        // ...
    }
}
```

## Dispatch by Expression Kind

### Tensor (Leaf)

Direct copy from tensor data:

```zig
fn evalTensor(tensor: anytype, output: []@TypeOf(tensor).ElementType) void {
    @memcpy(output, &tensor.data);
}
```

### Unary Expression

```zig
fn evalUnary(comptime Expr: type, expr: Expr, output: []Expr.ElementType) !void {
    // First, evaluate the input expression
    var input_buf: [Expr.InputType.numel()]Expr.ElementType = undefined;
    try evalInto(Expr.InputType, expr.input, &input_buf);

    // Apply the unary operation
    kernels.elementwise.unaryOp(
        Expr.operation,
        Expr.ElementType,
        &input_buf,
        output,
    );
}
```

### Binary Expression

```zig
fn evalBinary(comptime Expr: type, expr: Expr, output: []Expr.ElementType) !void {
    // Evaluate both inputs
    var lhs_buf: [Expr.LhsType.numel()]Expr.ElementType = undefined;
    var rhs_buf: [Expr.RhsType.numel()]Expr.ElementType = undefined;

    try evalInto(Expr.LhsType, expr.lhs, &lhs_buf);
    try evalInto(Expr.RhsType, expr.rhs, &rhs_buf);

    // Apply binary operation with broadcasting
    kernels.elementwise.binaryOp(
        Expr.operation,
        Expr.ElementType,
        &lhs_buf,
        &rhs_buf,
        output,
    );
}
```

### Matmul Expression

```zig
fn evalMatmul(comptime Expr: type, expr: Expr, output: []Expr.ElementType) !void {
    var lhs_buf: [Expr.LhsType.numel()]Expr.ElementType = undefined;
    var rhs_buf: [Expr.RhsType.numel()]Expr.ElementType = undefined;

    try evalInto(Expr.LhsType, expr.lhs, &lhs_buf);
    try evalInto(Expr.RhsType, expr.rhs, &rhs_buf);

    kernels.matmul.multiply(
        Expr.ElementType,
        Expr.LhsType.shape[0],
        Expr.LhsType.shape[1],
        Expr.RhsType.shape[1],
        &lhs_buf,
        &rhs_buf,
        output,
    );
}
```

### Reduce Expression

```zig
fn evalReduce(comptime Expr: type, expr: Expr, output: []Expr.ElementType) !void {
    var input_buf: [Expr.InputType.numel()]Expr.ElementType = undefined;
    try evalInto(Expr.InputType, expr.input, &input_buf);

    kernels.reduce.reduce(
        Expr.operation,
        Expr.ElementType,
        &input_buf,
        output,
        Expr.reduction_axes,
    );
}
```

## Recursive Evaluation

Expressions are evaluated bottom-up:

```zig
// Expression: input.matmul(weights).add(bias).relu()
//
// Evaluation order:
// 1. Evaluate input tensor
// 2. Evaluate weights tensor
// 3. Compute matmul(input, weights)
// 4. Evaluate bias tensor
// 5. Compute add(matmul_result, bias)
// 6. Compute relu(add_result)
```

## Buffer Management

### Stack Allocation

For expressions with known small sizes:

```zig
var buffer: [256]f32 = undefined;
```

### Dynamic Allocation

For larger or unknown sizes:

```zig
const buffer = try allocator.alloc(f32, size);
defer allocator.free(buffer);
```

## Fusion Integration

The executor can use fused kernels:

```zig
fn evalWithFusion(comptime Expr: type, expr: Expr, output: []Expr.ElementType) !void {
    const pattern = fusion.analyzer.analyze(Expr);

    switch (pattern) {
        .elementwise_chain => evalFusedChain(Expr, expr, output),
        .matmul_epilogue => evalFusedMatmul(Expr, expr, output),
        else => evalStandard(Expr, expr, output),
    }
}
```

## Error Handling

Allocation failures are propagated:

```zig
pub fn eval(expr: anytype, allocator: Allocator) ![]ElementType {
    const result = try allocator.alloc(ElementType, numel);
    errdefer allocator.free(result);  // Clean up on error

    try evalInto(expr, result);  // May fail
    return result;
}
```

## Next Steps

- [eval and evalInto](./eval.md) - API details
- [Expression Dispatch](./dispatch.md) - Type-based dispatch
