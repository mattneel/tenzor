# Introduction

**Tenzor** is a comptime tensor library for Zig that provides zero-cost abstractions for numerical computing. It leverages Zig's powerful compile-time capabilities to deliver:

- **Compile-time shape validation** - Shape mismatches are caught at compile time, not runtime
- **Zero-cost abstractions** - Expression graphs are fully resolved at compile time
- **Automatic operation fusion** - Chains of operations are fused into single kernels
- **SIMD optimization** - Vectorized kernels for maximum CPU performance

## Philosophy

Tenzor follows the principle that **if it compiles, it works**. By encoding tensor shapes and operations in the type system, entire categories of bugs are eliminated before your program ever runs.

```zig
const std = @import("std");
const tz = @import("tenzor");

// Shapes are part of the type
const Matrix = tz.Tensor(f32, .{ 3, 4 });  // 3x4 matrix
const Vector = tz.Tensor(f32, .{4});       // 4-element vector

// This compiles - shapes are compatible
const result = Matrix.add(Vector);  // Broadcasting works

// This would NOT compile - shape mismatch
// const bad = Matrix.matmul(Vector);  // Error: inner dimensions don't match
```

## Key Features

### Lazy Evaluation

Operations build expression graphs that are only evaluated when you call `eval()`:

```zig
const a = tensor_a.add(tensor_b);     // No computation yet
const b = a.mul(tensor_c);            // Still no computation
const c = b.relu();                   // Building the graph

const result = tz.eval(c, allocator); // Now it computes
```

### Automatic Fusion

The fusion engine detects patterns and generates optimized kernels:

```zig
// These three operations...
const expr = x.matmul(weights).add(bias).relu();

// ...become a single fused kernel at compile time
```

### Type-Safe Broadcasting

Broadcasting rules are checked at compile time:

```zig
const A = tz.Tensor(f32, .{ 3, 4 });    // [3, 4]
const B = tz.Tensor(f32, .{4});          // [4]
const C = tz.Tensor(f32, .{ 3, 1 });     // [3, 1]

// Result shapes are computed at compile time
const AB = A.add(B);  // [3, 4] + [4] = [3, 4]
const AC = A.mul(C);  // [3, 4] * [3, 1] = [3, 4]
```

## When to Use Tenzor

Tenzor is ideal for:

- **ML inference** - Compile-time optimization for fixed model architectures
- **Scientific computing** - Type-safe numerical algorithms
- **Embedded systems** - Predictable memory usage, no runtime allocations in hot paths
- **Performance-critical code** - SIMD-optimized kernels with zero abstraction overhead

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      User Code                               │
│   tensor.add(other).relu().matmul(weights)                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Expression Graph                           │
│   Types encode: shapes, operations, data flow               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Fusion Engine                             │
│   Detects patterns, generates fused kernels                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    CPU Backend                               │
│   SIMD kernels, threading, memory management                │
└─────────────────────────────────────────────────────────────┘
```

## What's Next?

- [Installation](./getting-started/installation.md) - Add tenzor to your project
- [Quick Start](./getting-started/quick-start.md) - Build something in 5 minutes
- [Core Concepts](./core/tensors.md) - Understand the fundamentals
