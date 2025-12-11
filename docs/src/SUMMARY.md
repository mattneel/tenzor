# Summary

[Introduction](./introduction.md)

# Getting Started

- [Installation](./getting-started/installation.md)
- [Quick Start](./getting-started/quick-start.md)
- [Your First Tensor](./getting-started/first-tensor.md)

# Core Concepts

- [Tensors](./core/tensors.md)
  - [Creating Tensors](./core/creating-tensors.md)
  - [Shape and Dimensions](./core/shapes.md)
  - [Data Types](./core/dtypes.md)
  - [Memory Layout](./core/memory-layout.md)
- [Expression Graphs](./core/expression-graphs.md)
  - [Lazy Evaluation](./core/lazy-evaluation.md)
  - [Type-Level Encoding](./core/type-level-encoding.md)

# Operations

- [Overview](./operations/overview.md)
- [Unary Operations](./operations/unary.md)
  - [Math Functions](./operations/math-functions.md)
  - [Activation Functions](./operations/activations.md)
- [Binary Operations](./operations/binary.md)
  - [Arithmetic](./operations/arithmetic.md)
  - [Comparisons](./operations/comparisons.md)
- [Matrix Operations](./operations/matrix.md)
  - [Matrix Multiplication](./operations/matmul.md)
  - [Transpose](./operations/transpose.md)
- [Reductions](./operations/reductions.md)

# CPU Backend

- [Architecture](./backend/architecture.md)
- [SIMD Optimization](./backend/simd.md)
  - [Vector Types](./backend/vector-types.md)
  - [Vectorized Kernels](./backend/vectorized-kernels.md)
- [Execution](./backend/execution.md)
  - [eval and evalInto](./backend/eval.md)
  - [Expression Dispatch](./backend/dispatch.md)

# Fusion Engine

- [Overview](./fusion/overview.md)
- [Pattern Detection](./fusion/patterns.md)
  - [Elementwise Chains](./fusion/elementwise-chains.md)
  - [Matmul Epilogues](./fusion/matmul-epilogues.md)
  - [Reduce Epilogues](./fusion/reduce-epilogues.md)
- [Code Generation](./fusion/codegen.md)

# Memory Management

- [Allocator Design](./memory/allocator.md)
- [Buffer Pooling](./memory/pooling.md)
- [Compute Arenas](./memory/arenas.md)

# Threading

- [Thread Pool](./threading/thread-pool.md)
- [Parallel Execution](./threading/parallel-execution.md)
- [Work Partitioning](./threading/partitioning.md)

# Advanced Topics

- [Comptime Magic](./advanced/comptime.md)
- [Shape Algebra](./advanced/shape-algebra.md)
- [Broadcasting](./advanced/broadcasting.md)
- [Custom Operations](./advanced/custom-ops.md)

# API Reference

- [Core Types](./api/core.md)
- [Operations](./api/operations.md)
- [Backend](./api/backend.md)
- [Memory](./api/memory.md)

# Appendices

- [Performance Tips](./appendix/performance.md)
- [Comparison with Other Libraries](./appendix/comparison.md)
- [Contributing](./appendix/contributing.md)
