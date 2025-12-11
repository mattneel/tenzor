# Lazy Evaluation

Tenzor uses lazy evaluation: operations build expression graphs without computing results until explicitly requested.

## Eager vs Lazy

### Eager Evaluation (Other Libraries)

```python
# NumPy-style: each operation executes immediately
a = tensor_a + tensor_b   # Allocates and computes
b = a * tensor_c          # Allocates and computes
c = relu(b)               # Allocates and computes
# Three allocations, three kernel launches
```

### Lazy Evaluation (Tenzor)

```zig
// Tenzor: operations build a graph
const a = tensor_a.add(tensor_b);  // No computation
const b = a.mul(tensor_c);          // No computation
const c = b.relu();                 // No computation

const result = tz.eval(c, allocator);  // One fused computation
// One allocation, one optimized kernel
```

## Benefits of Lazy Evaluation

### 1. Operation Fusion

Multiple operations become a single kernel:

```zig
// Without fusion: 3 memory passes
// temp1 = matmul(x, w)
// temp2 = temp1 + bias
// output = relu(temp2)

// With fusion: 1 memory pass
const expr = x.matmul(w).add(bias).relu();
const output = tz.eval(expr, allocator);
```

### 2. Dead Code Elimination

Unused computations are never executed:

```zig
const a = tensor.exp();    // Built but...
const b = tensor.log();    // Built but...
const c = tensor.relu();   // ...only this is used

const result = tz.eval(c, allocator);  // a and b never computed
```

### 3. Memory Optimization

Intermediate buffers can be reused:

```zig
const a = big_tensor.exp();
const b = a.log();          // Can reuse a's buffer
const c = b.sqrt();         // Can reuse b's buffer
```

### 4. Compile-Time Analysis

The full computation graph is visible at compile time:

```zig
const expr = input.matmul(w1).relu().matmul(w2);

comptime {
    // Can analyze the entire graph
    const fusion_plan = tz.fusion.analyzer.analyze(expr);
}
```

## Triggering Evaluation

### eval()

Allocates and returns the result:

```zig
const expr = tensor.relu().add(other);
const result = try tz.eval(expr, allocator);
defer allocator.free(result);

// result is []f32 with the computed values
```

### evalInto()

Writes to a provided buffer:

```zig
const expr = tensor.relu();
var buffer: [100]f32 = undefined;

tz.evalInto(expr, &buffer);
// buffer now contains the result
```

## Evaluation Order

When an expression is evaluated:

1. **Graph Analysis** - Determine execution order
2. **Fusion Detection** - Find fuseable patterns
3. **Buffer Allocation** - Allocate intermediates
4. **Kernel Execution** - Run optimized kernels
5. **Result Return** - Copy to output

## When Does Computation Happen?

| Action | Computation? |
|--------|-------------|
| `a.add(b)` | No |
| `a.relu()` | No |
| `a.matmul(b)` | No |
| `tz.eval(expr)` | Yes |
| `tz.evalInto(expr, buf)` | Yes |

## Expression Lifetime

Expressions are values, not references:

```zig
const expr = blk: {
    const a = tensor.relu();
    const b = a.add(other);
    break :blk b;  // Expression escapes the block
};

// expr is still valid - it's a type containing the graph
const result = tz.eval(expr, allocator);
```

## Forcing Materialization

Sometimes you need intermediate results:

```zig
// Need the same intermediate twice
const intermediate = tensor.matmul(weights);

const branch_a = intermediate.add(bias_a).relu();
const branch_b = intermediate.add(bias_b).sigmoid();

// Evaluate intermediate first to avoid recomputation
const inter_data = try tz.eval(intermediate, allocator);
defer allocator.free(inter_data);

// Then use the data for both branches
```

## Debugging Expressions

Print expression structure at compile time:

```zig
const expr = tensor.relu().add(other).mul(scale);

comptime {
    @compileLog("Expression kind:", expr.kind);
    @compileLog("Operation:", expr.operation);
    @compileLog("Shape:", expr.shape);
}
```

## Performance Implications

### Good: Chained Operations

```zig
// Excellent: one fused kernel
const result = tz.eval(
    input.matmul(w).add(b).relu().mul(scale),
    allocator,
);
```

### Avoid: Repeated Evaluation

```zig
// Bad: evaluates the same subexpression multiple times
const expr = expensive_computation();
const a = tz.eval(expr, allocator);
const b = tz.eval(expr, allocator);  // Recomputes!

// Better: evaluate once, reuse data
const data = tz.eval(expr, allocator);
// Use data multiple times
```

## Next Steps

- [Type-Level Encoding](./type-level-encoding.md) - How the graph is represented
- [Fusion Engine](../fusion/overview.md) - How operations are optimized
- [Execution](../backend/execution.md) - How evaluation works
