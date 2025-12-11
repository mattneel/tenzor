# eval and evalInto

The primary APIs for evaluating expression graphs.

## eval()

Allocates and returns the result.

```zig
pub fn eval(
    comptime Expr: type,
    expr: Expr,
    allocator: std.mem.Allocator,
) ![]Expr.ElementType
```

### Usage

```zig
const tz = @import("tenzor");

const A = tz.Tensor(f32, .{ 3, 4 });
var a = A{};
// ... initialize a.data ...

const expr = a.relu().add(other);

const result = try tz.eval(expr, allocator);
defer allocator.free(result);

// Use result...
```

### Behavior

1. Allocates buffer of size `Expr.numel()` elements
2. Evaluates expression into buffer
3. Returns buffer (caller owns memory)
4. On error, cleans up allocation

### Error Handling

```zig
const result = tz.eval(expr, allocator) catch |err| {
    switch (err) {
        error.OutOfMemory => // Handle allocation failure
    }
};
```

## evalInto()

Evaluates into existing buffer.

```zig
pub fn evalInto(
    comptime Expr: type,
    expr: Expr,
    output: []Expr.ElementType,
) void
```

### Usage

```zig
// Stack-allocated buffer
var buffer: [12]f32 = undefined;
tz.evalInto(expr, &buffer);

// Pre-allocated slice
const output = try allocator.alloc(f32, expr.numel());
tz.evalInto(expr, output);
```

### Requirements

- `output.len` must equal `Expr.numel()`
- Buffer must be properly aligned

### Use Cases

**Avoiding allocation:**
```zig
var workspace: [1024]f32 = undefined;
tz.evalInto(small_expr, workspace[0..small_expr.numel()]);
```

**Reusing buffers:**
```zig
const buffer = try allocator.alloc(f32, max_size);
defer allocator.free(buffer);

for (expressions) |expr| {
    tz.evalInto(expr, buffer[0..expr.numel()]);
    // Process buffer...
}
```

**In-place operations:**
```zig
// Evaluate directly into tensor data
var tensor = Tensor{};
tz.evalInto(expr, &tensor.data);
```

## Comparison

| Feature | eval() | evalInto() |
|---------|--------|------------|
| Allocates | Yes | No |
| Returns | Slice | Void |
| Errors | OutOfMemory | None |
| Use when | Need new buffer | Have buffer |

## Expression Type Requirements

Both functions require:

```zig
comptime {
    // Expression must have these
    _ = Expr.ElementType;  // Element type
    _ = Expr.numel();      // Number of elements
    _ = Expr.kind;         // Expression kind for dispatch
}
```

## Performance Considerations

### eval() Overhead

- Allocation: ~100-1000 cycles
- Deallocation: ~100-500 cycles
- For large tensors, negligible relative to computation

### evalInto() Benefits

- Zero allocation overhead
- Cache-warm buffers for repeated use
- Predictable memory usage

### Recommendation

Use `evalInto()` when:
- Hot loops with repeated evaluation
- Memory-constrained environments
- Predictable buffer sizes
- Real-time requirements

Use `eval()` when:
- One-time evaluation
- Unknown output size at call site
- Convenience over performance

## Advanced Patterns

### Batch Processing

```zig
const batch_size = 32;
const buffers = try allocator.alloc([output_size]f32, batch_size);
defer allocator.free(buffers);

for (inputs, buffers) |input, *buffer| {
    const expr = buildExpr(input);
    tz.evalInto(expr, buffer);
}
```

### Pipeline with Reuse

```zig
var temp1: [size]f32 = undefined;
var temp2: [size]f32 = undefined;

for (0..iterations) |_| {
    tz.evalInto(stage1_expr, &temp1);
    tz.evalInto(stage2_expr, &temp2);
    // Use temp2...
}
```

## Next Steps

- [Expression Dispatch](./dispatch.md) - How expressions are evaluated
- [Memory Management](../memory/allocator.md) - Allocation strategies
