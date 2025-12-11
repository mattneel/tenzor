# Quick Start

This guide will have you running tensor computations in under 5 minutes.

## A Simple Neural Network Layer

Let's implement a fully-connected layer with ReLU activation:

```zig
const std = @import("std");
const tz = @import("tenzor");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Define our tensor types with compile-time shapes
    const Input = tz.Tensor(f32, .{ 1, 784 });    // Batch of 1, 784 features
    const Weights = tz.Tensor(f32, .{ 784, 128 }); // 784 -> 128 neurons
    const Bias = tz.Tensor(f32, .{128});           // 128 biases

    // Create tensors with data
    var input = Input{};
    var weights = Weights{};
    var bias = Bias{};

    // Initialize with some values
    @memset(&input.data, 0.1);
    @memset(&weights.data, 0.01);
    @memset(&bias.data, 0.0);

    // Build the expression graph: output = relu(input @ weights + bias)
    const linear = input.matmul(weights);
    const biased = linear.add(bias);
    const activated = biased.relu();

    // Evaluate the expression
    const result = try tz.eval(activated, allocator);
    defer allocator.free(result);

    std.debug.print("Output shape: [{}, {}]\n", .{ 1, 128 });
    std.debug.print("First 5 values: ", .{});
    for (result[0..5]) |v| {
        std.debug.print("{d:.4} ", .{v});
    }
    std.debug.print("\n", .{});
}
```

## Understanding the Flow

### 1. Type-Level Shapes

```zig
const Input = tz.Tensor(f32, .{ 1, 784 });
```

This creates a *type*, not a value. The shape `{1, 784}` is encoded in the type system, enabling compile-time validation.

### 2. Expression Building

```zig
const linear = input.matmul(weights);
const biased = linear.add(bias);
const activated = biased.relu();
```

No computation happens here. Each operation returns a new *type* that encodes the operation and its inputs.

### 3. Evaluation

```zig
const result = try tz.eval(activated, allocator);
```

Only when you call `eval` does the computation execute. The fusion engine analyzes the expression graph and generates optimized code.

## Compile-Time Safety

Try changing the weight dimensions:

```zig
const Weights = tz.Tensor(f32, .{ 784, 128 });  // Correct
const Weights = tz.Tensor(f32, .{ 100, 128 });  // Wrong!
```

With the wrong dimensions, you get a compile error:

```
error: Matmul inner dimensions must match: 784 vs 100
```

## Method Chaining

Operations can be chained fluently:

```zig
const output = input
    .matmul(weights)
    .add(bias)
    .relu()
    .mul(scale);
```

Each method returns a new expression type that can be further composed.

## Common Patterns

### Element-wise Operations

```zig
const a = tensor_a.add(tensor_b);
const b = tensor_a.mul(tensor_b);
const c = tensor_a.sub(tensor_b);
const d = tensor_a.div(tensor_b);
```

### Unary Transformations

```zig
const e = tensor.exp();
const l = tensor.log();
const s = tensor.sqrt();
const r = tensor.relu();
```

### Reductions

```zig
const sum = tensor.sum(.{});           // Full reduction
const row_sum = tensor.sum(.{1});       // Sum along axis 1
const mean = tensor.mean(.{0});         // Mean along axis 0
```

### Matrix Operations

```zig
const product = matrix_a.matmul(matrix_b);
```

## Memory Management

Tenzor is designed for predictable memory usage:

```zig
// Option 1: Let eval allocate
const result = try tz.eval(expr, allocator);
defer allocator.free(result);

// Option 2: Provide your own buffer
var buffer: [128]f32 = undefined;
tz.evalInto(expr, &buffer);
```

## Next Steps

- [Your First Tensor](./first-tensor.md) - Deep dive into tensor creation
- [Expression Graphs](../core/expression-graphs.md) - Understand lazy evaluation
- [Operations](../operations/overview.md) - Full operation reference
