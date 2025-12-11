# CPU Backend Architecture

The CPU backend is tenzor's execution engine, providing optimized kernels for tensor operations.

## Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Expression Graph                          │
│   (Type-level representation of computation)                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Fusion Engine                             │
│   - Pattern detection                                        │
│   - Kernel selection                                         │
│   - Code generation                                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Executor                                  │
│   - Expression dispatch                                      │
│   - Memory management                                        │
│   - Kernel invocation                                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Kernels                                   │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│   │Elementwise│  │  Matmul  │  │  Reduce  │                  │
│   └──────────┘  └──────────┘  └──────────┘                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    SIMD Layer                                │
│   - Vector operations                                        │
│   - Hardware abstraction                                     │
└─────────────────────────────────────────────────────────────┘
```

## Components

### Executor (`executor.zig`)

The executor is the entry point for evaluation:

```zig
const tz = @import("tenzor");

// Evaluate an expression
const result = try tz.eval(expression, allocator);

// Evaluate into existing buffer
tz.evalInto(expression, buffer);
```

The executor:
1. Analyzes expression type at compile time
2. Dispatches to appropriate kernel
3. Manages intermediate buffers
4. Returns final result

### Kernels

Specialized implementations for each operation type:

| Kernel | File | Purpose |
|--------|------|---------|
| Elementwise | `kernels/elementwise.zig` | Unary/binary operations |
| Matmul | `kernels/matmul.zig` | Matrix multiplication |
| Reduce | `kernels/reduce.zig` | Aggregations |

### SIMD Layer (`simd.zig`)

Hardware abstraction for vectorization:

```zig
const simd = @import("tenzor").backend.cpu.simd;

// Vector type for element type
const Vec = simd.Vec(f32);  // e.g., @Vector(8, f32) on AVX

// Operations
const sum = simd.add(f32, vec_a, vec_b);
const result = simd.reduceAdd(f32, vec);
```

### Threading (`threading.zig`)

Parallel execution support:

```zig
const threading = @import("tenzor").backend.cpu.threading;

const pool = try threading.ThreadPool.create(allocator, .{});
defer pool.destroy();

pool.parallelFor(0, n, chunk_size, context, workFn);
```

## Data Flow

### 1. Expression Analysis

At compile time, the expression type is analyzed:

```zig
const expr = input.matmul(weights).add(bias).relu();

// Compile-time: determine fusion pattern
const pattern = fusion.analyzer.analyze(@TypeOf(expr));
```

### 2. Kernel Selection

Based on pattern, select kernel:

```zig
switch (pattern) {
    .single => executeSingle(expr),
    .elementwise_chain => executeFusedChain(expr),
    .matmul_epilogue => executeMatmulFused(expr),
    .reduce_epilogue => executeReduceFused(expr),
}
```

### 3. Buffer Allocation

Allocate intermediate buffers as needed:

```zig
const intermediate = try allocator.alloc(f32, size);
defer allocator.free(intermediate);
```

### 4. Kernel Execution

Execute the selected kernel:

```zig
// SIMD-optimized kernel
elementwise.unaryOp(.relu, input, output);
matmul.multiply(a, b, c);
reduce.sum(input, output, axis);
```

### 5. Result Return

Return or write to output buffer:

```zig
// Copy to user buffer or return allocated slice
return result;
```

## Memory Management

The backend uses several memory strategies:

### Stack Allocation

For small, fixed-size tensors:

```zig
var buffer: [256]f32 = undefined;
```

### Heap Allocation

For dynamic sizes:

```zig
const buffer = try allocator.alloc(f32, size);
defer allocator.free(buffer);
```

### Buffer Pooling

Reuse recently freed buffers:

```zig
const pool = BufferPool.init(allocator);
const buf = pool.get(size) orelse try allocator.alloc(f32, size);
defer pool.put(buf);
```

## Performance Characteristics

### Vectorization

- AVX: 8 f32 or 4 f64 per instruction
- AVX-512: 16 f32 or 8 f64 per instruction
- Automatic fallback for remainders

### Cache Optimization

- Tiled algorithms for large matrices
- Sequential access patterns
- Prefetching where beneficial

### Threading

- Parallel for large workloads
- Work stealing for load balance
- Configurable thread count

## Next Steps

- [SIMD Optimization](./simd.md) - Vector operations
- [Execution](./execution.md) - Expression evaluation
- [Memory Management](../memory/allocator.md) - Buffer strategies
