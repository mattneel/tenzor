# Expression Dispatch

The executor uses type-based dispatch to select the appropriate evaluation strategy.

## Dispatch Mechanism

At compile time, the expression type determines the execution path:

```zig
pub fn evalInto(comptime Expr: type, expr: Expr, output: []Expr.ElementType) !void {
    switch (Expr.kind) {
        .tensor => evalTensor(expr, output),
        .constant => evalConstant(Expr, output),
        .unary => evalUnary(Expr, expr, output),
        .binary => evalBinary(Expr, expr, output),
        .matmul => evalMatmul(Expr, expr, output),
        .reduce => evalReduce(Expr, expr, output),
        .reshape => evalReshape(Expr, expr, output),
        .transpose => evalTranspose(Expr, expr, output),
    }
}
```

## NodeKind Enum

Expression kinds are defined in the type system:

```zig
pub const NodeKind = enum {
    tensor,     // Leaf: concrete tensor with data
    constant,   // Leaf: compile-time constant
    unary,      // Single-input operation
    binary,     // Two-input operation
    matmul,     // Matrix multiplication
    reduce,     // Aggregation
    reshape,    // Shape manipulation
    transpose,  // Axis permutation
};
```

## Dispatch by Kind

### Tensor Dispatch

Direct data access:

```zig
fn evalTensor(tensor: anytype, output: []@TypeOf(tensor).ElementType) void {
    const T = @TypeOf(tensor);
    @memcpy(output[0..T.numel()], &tensor.data);
}
```

### Constant Dispatch

Broadcast scalar value:

```zig
fn evalConstant(comptime Expr: type, output: []Expr.ElementType) void {
    @memset(output, Expr.value);
}
```

### Unary Dispatch

Evaluate input, then apply operation:

```zig
fn evalUnary(comptime Expr: type, expr: Expr, output: []Expr.ElementType) !void {
    const Input = Expr.InputType;
    var input_buf: [Input.numel()]Expr.ElementType = undefined;

    // Recursive evaluation
    try evalInto(Input, expr.input, &input_buf);

    // Apply operation based on Expr.operation
    kernels.elementwise.unaryOp(
        Expr.operation,  // .relu, .exp, .log, etc.
        Expr.ElementType,
        &input_buf,
        output,
    );
}
```

### Binary Dispatch

Evaluate both inputs, apply with broadcasting:

```zig
fn evalBinary(comptime Expr: type, expr: Expr, output: []Expr.ElementType) !void {
    const Lhs = Expr.LhsType;
    const Rhs = Expr.RhsType;

    var lhs_buf: [Lhs.numel()]Expr.ElementType = undefined;
    var rhs_buf: [Rhs.numel()]Expr.ElementType = undefined;

    try evalInto(Lhs, expr.lhs, &lhs_buf);
    try evalInto(Rhs, expr.rhs, &rhs_buf);

    // Handle broadcasting if shapes differ
    if (Lhs.numel() == Rhs.numel()) {
        kernels.elementwise.binaryOp(
            Expr.operation,
            Expr.ElementType,
            &lhs_buf,
            &rhs_buf,
            output,
        );
    } else {
        kernels.elementwise.binaryOpBroadcast(
            Expr.operation,
            Expr.ElementType,
            &lhs_buf,
            Lhs.shape,
            &rhs_buf,
            Rhs.shape,
            output,
        );
    }
}
```

### Matmul Dispatch

Matrix multiplication with shape handling:

```zig
fn evalMatmul(comptime Expr: type, expr: Expr, output: []Expr.ElementType) !void {
    const Lhs = Expr.LhsType;
    const Rhs = Expr.RhsType;

    var lhs_buf: [Lhs.numel()]Expr.ElementType = undefined;
    var rhs_buf: [Rhs.numel()]Expr.ElementType = undefined;

    try evalInto(Lhs, expr.lhs, &lhs_buf);
    try evalInto(Rhs, expr.rhs, &rhs_buf);

    kernels.matmul.multiply(
        Expr.ElementType,
        Lhs.shape[0],  // M
        Lhs.shape[1],  // K
        Rhs.shape[1],  // N
        &lhs_buf,
        &rhs_buf,
        output,
    );
}
```

### Reduce Dispatch

```zig
fn evalReduce(comptime Expr: type, expr: Expr, output: []Expr.ElementType) !void {
    const Input = Expr.InputType;
    var input_buf: [Input.numel()]Expr.ElementType = undefined;

    try evalInto(Input, expr.input, &input_buf);

    kernels.reduce.reduce(
        Expr.operation,
        Expr.ElementType,
        &input_buf,
        Input.shape,
        output,
        Expr.reduction_axes,
        Expr.keep_dims,
    );
}
```

## Compile-Time Type Extraction

The dispatch uses compile-time type information:

```zig
comptime {
    // From expression type, we know:
    _ = Expr.kind;           // Which dispatch path
    _ = Expr.operation;      // Which operation
    _ = Expr.ElementType;    // Data type
    _ = Expr.shape;          // Result shape
    _ = Expr.numel();        // Output size

    // For composite expressions:
    _ = Expr.InputType;      // Input expression type (unary)
    _ = Expr.LhsType;        // Left input type (binary)
    _ = Expr.RhsType;        // Right input type (binary)
}
```

## Fusion-Aware Dispatch

When fusion is enabled:

```zig
fn evalWithFusion(comptime Expr: type, expr: Expr, output: []Expr.ElementType) !void {
    const plan = comptime fusion.analyzer.analyze(Expr);

    switch (plan.pattern) {
        .single => evalStandard(Expr, expr, output),
        .elementwise_chain => {
            const chain = plan.elementwise_chain;
            evalFusedChain(chain, expr, output);
        },
        .matmul_epilogue => {
            const epilogue = plan.matmul_epilogue;
            evalFusedMatmul(epilogue, expr, output);
        },
        .reduce_epilogue => evalFusedReduce(plan, expr, output),
    }
}
```

## Recursive Descent

Evaluation proceeds recursively:

```
evalInto(relu(add(matmul(A, B), C)))
│
├─► evalUnary(.relu, ...)
│   │
│   └─► evalBinary(.add, ...)
│       │
│       ├─► evalMatmul(A, B)
│       │   │
│       │   ├─► evalTensor(A)
│       │   └─► evalTensor(B)
│       │
│       └─► evalTensor(C)
│
└─► Apply relu kernel
```

## Next Steps

- [Fusion Engine](../fusion/overview.md) - Optimized dispatch
- [Vectorized Kernels](./vectorized-kernels.md) - Kernel implementations
