# Tenzor

A compile-time tensor library for Zig with zero-cost abstractions.

## Features

- **Compile-time shape checking** - Shape errors caught at compile time, not runtime
- **Zero-cost abstractions** - Expression graphs with no runtime overhead
- **Lazy evaluation** - Operations fuse automatically for optimal performance
- **SIMD acceleration** - Vectorized kernels for all element-wise operations
- **Multi-threaded** - Parallel execution with work-stealing thread pool
- **No dependencies** - Pure Zig, no external libraries required

## Quick Start

```zig
const std = @import("std");
const tenzor = @import("tenzor");
const Tensor = tenzor.Tensor;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Define tensor types with compile-time shapes
    const Mat = Tensor(f32, .{ 2, 3 });
    const Vec = Tensor(f32, .{3});

    // Create tensors
    var matrix = Mat.init(.{
        1, 2, 3,
        4, 5, 6,
    });
    var bias = Vec.init(.{ 0.1, 0.2, 0.3 });

    // Build expression graph (lazy)
    const expr = matrix.add(bias).mul(bias).exp();

    // Evaluate (fused execution)
    const result = try expr.eval(allocator);
    defer result.deinit(allocator);

    std.debug.print("Result: {any}\n", .{result.data});
}
```

## Installation

Add to `build.zig.zon`:

```zig
.dependencies = .{
    .tenzor = .{
        .url = "https://github.com/your-org/tenzor/archive/refs/tags/v0.1.0.tar.gz",
        .hash = "...",
    },
},
```

Then in `build.zig`:

```zig
const tenzor = b.dependency("tenzor", .{
    .target = target,
    .optimize = optimize,
});
exe.root_module.addImport("tenzor", tenzor.module("tenzor"));
```

## Operations

### Element-wise

```zig
// Unary
tensor.neg()       // -x
tensor.exp()       // e^x
tensor.log()       // ln(x)
tensor.sqrt()      // sqrt(x)
tensor.sin()       // sin(x)
tensor.cos()       // cos(x)
tensor.tanh()      // tanh(x)
tensor.relu()      // max(0, x)
tensor.sigmoid()   // 1 / (1 + e^-x)

// Binary (with broadcasting)
a.add(b)           // a + b
a.sub(b)           // a - b
a.mul(b)           // a * b
a.div(b)           // a / b
a.pow(b)           // a^b
a.maximum(b)       // max(a, b)
a.minimum(b)       // min(a, b)
```

### Matrix Operations

```zig
// Matrix multiplication
const C = A.matmul(B);           // [M, K] @ [K, N] -> [M, N]

// Batched matmul
const C = A.matmul(B);           // [B, M, K] @ [B, K, N] -> [B, M, N]

// Transpose
const AT = A.transpose();         // Swap last two dims
const P = A.transposeAxes(.{2, 0, 1});  // Permute dims
```

### Reductions

```zig
tensor.sum(.{0}, false)     // Sum over axis 0
tensor.sum(.{}, false)      // Sum all elements (scalar)
tensor.mean(.{1}, true)     // Mean over axis 1, keep dims
tensor.max(.{-1}, false)    // Max over last axis
tensor.min(.{0, 1}, false)  // Min over axes 0 and 1
tensor.prod(.{}, false)     // Product of all elements
```

## Broadcasting

NumPy-compatible broadcasting:

```zig
const A = Tensor(f32, .{ 3, 4 });   // [3, 4]
const B = Tensor(f32, .{4});        // [4] -> broadcasts to [3, 4]

const C = a.add(b);  // Shape: [3, 4]
```

## Lazy Evaluation & Fusion

Operations build expression graphs that fuse on evaluation:

```zig
// These operations fuse into a single pass
const result = input
    .matmul(weights)    // Matrix multiply
    .add(bias)          // Fused epilogue
    .relu()             // Fused activation
    .eval(allocator);   // Single materialization
```

Fusion patterns:
- **Elementwise chains**: `a.add(b).mul(c).exp()` -> single kernel
- **Matmul epilogues**: `matmul(A, B).add(bias).relu()` -> fused
- **Reduce epilogues**: `tensor.sum(...).mul(scale)` -> fused

## Compile-Time Safety

Shape errors are caught at compile time:

```zig
const A = Tensor(f32, .{ 3, 4 });
const B = Tensor(f32, .{ 5, 6 });

// Compile error: "Shapes not broadcastable"
const C = a.add(b);

const D = Tensor(f32, .{ 4, 5 });

// Compile error: "Matmul inner dimension mismatch"
const E = a.matmul(d);  // [3,4] @ [4,5] is valid, but [3,4] @ [5,6] is not
```

## Memory Management

### Pool Allocator

Reuse allocations across operations:

```zig
var pool = tenzor.memory.PoolAllocator.init(page_allocator);
defer pool.deinit();

for (batches) |batch| {
    const result = try expr.eval(pool.allocator());
    process(result);
    pool.clear();  // Return to pool, don't free
}
```

### Arena Pattern

For expression trees with many intermediates:

```zig
var arena = std.heap.ArenaAllocator.init(page_allocator);
defer arena.deinit();

const result = try complex_expr.eval(arena.allocator());
// All intermediates freed together
```

## Threading

Automatic parallelization for large tensors:

```zig
const pool = try tenzor.backend.cpu.threading.ThreadPool.create(allocator, .{});
defer pool.destroy();

// Operations automatically parallelize when beneficial
// Threshold: 8192+ elements
```

## Performance

| Operation | Optimization |
|-----------|--------------|
| Element-wise | SIMD vectorized (AVX2/AVX-512/NEON) |
| Matmul | Tiled + SIMD + threaded |
| Reductions | Parallel reduce + SIMD |
| Memory | Buffer pooling, cache-aligned |

## Documentation

Full documentation available via mdBook:

```bash
cd docs
mdbook serve  # http://localhost:3000
```

## Requirements

- Zig 0.16 or later

## License

MIT

## Contributing

See [CONTRIBUTING.md](docs/src/appendix/contributing.md) for guidelines.
