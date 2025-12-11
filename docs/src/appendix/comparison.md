# Comparison with Other Libraries

## Overview

| Feature | Tenzor | NumPy | PyTorch | Eigen |
|---------|--------|-------|---------|-------|
| Language | Zig | Python/C | Python/C++ | C++ |
| Compile-time shapes | Yes | No | No | Optional |
| Zero-cost abstractions | Yes | No | No | Yes |
| Lazy evaluation | Yes | No | Yes | Yes |
| SIMD | Auto | Via BLAS | Via BLAS | Auto |
| GPU support | No | Via CuPy | Yes | No |

---

## NumPy

### Similarities

- Broadcasting semantics
- Familiar operations (add, mul, matmul, etc.)
- N-dimensional arrays

### Differences

| Aspect | Tenzor | NumPy |
|--------|--------|-------|
| Shape checking | Compile-time | Runtime |
| Memory model | Explicit | GC-managed |
| Error handling | Compile errors | Runtime exceptions |
| Fusion | Automatic | Manual (einsum) |

### Code Comparison

```python
# NumPy
import numpy as np

a = np.zeros((3, 4))
b = np.ones((4, 5))
c = a @ b + np.array([1, 2, 3, 4, 5])
```

```zig
// Tenzor
const A = Tensor(f32, .{ 3, 4 });
const B = Tensor(f32, .{ 4, 5 });
const Bias = Tensor(f32, .{5});

var a = A.fill(0);
var b = B.fill(1);
var bias = Bias.init(.{ 1, 2, 3, 4, 5 });

const c = a.matmul(b).add(bias).eval(allocator);
```

### When to Choose

**NumPy:**
- Rapid prototyping
- Interactive exploration
- Existing Python ecosystem

**Tenzor:**
- Performance-critical production
- Embedded systems
- Type safety requirements

---

## PyTorch

### Similarities

- Lazy evaluation (autograd)
- Expression graphs
- Modern API design

### Differences

| Aspect | Tenzor | PyTorch |
|--------|--------|---------|
| Autograd | No | Yes |
| Dynamic shapes | No | Yes |
| GPU | No | Yes |
| Compilation | AOT | JIT (TorchScript) |
| Runtime overhead | Zero | Python interop |

### Code Comparison

```python
# PyTorch
import torch

x = torch.randn(32, 784)
w = torch.randn(784, 128)
b = torch.randn(128)
y = torch.relu(x @ w + b)
```

```zig
// Tenzor
const X = Tensor(f32, .{ 32, 784 });
const W = Tensor(f32, .{ 784, 128 });
const B = Tensor(f32, .{128});

const y = x.matmul(w).add(b).relu().eval(allocator);
```

### When to Choose

**PyTorch:**
- Training neural networks
- GPU acceleration needed
- Dynamic computation graphs
- Research/experimentation

**Tenzor:**
- Inference only
- CPU deployment
- Static shapes known
- No Python dependency

---

## Eigen

### Similarities

- C++/Zig (systems language)
- Expression templates / comptime
- Lazy evaluation
- SIMD optimization

### Differences

| Aspect | Tenzor | Eigen |
|--------|--------|-------|
| Template complexity | Low (Zig) | High (C++) |
| Compile times | Fast | Slow |
| Error messages | Clear | Cryptic |
| Fixed-size support | Primary | Optional |

### Code Comparison

```cpp
// Eigen
#include <Eigen/Dense>

Eigen::Matrix<float, 3, 4> A;
Eigen::Matrix<float, 4, 5> B;
auto C = A * B;  // Lazy
Eigen::Matrix<float, 3, 5> result = C;  // Evaluated
```

```zig
// Tenzor
const A = Tensor(f32, .{ 3, 4 });
const B = Tensor(f32, .{ 4, 5 });

var a: A = ...;
var b: B = ...;
const c = a.matmul(b);  // Lazy
const result = c.eval(allocator);  // Evaluated
```

### When to Choose

**Eigen:**
- Existing C++ codebase
- Dense linear algebra focus
- Mature ecosystem

**Tenzor:**
- New Zig projects
- Simpler compile-time model
- Better error messages

---

## BLAS/LAPACK

### Relationship

Tenzor provides higher-level abstractions vs raw BLAS:

```c
// BLAS
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            M, N, K, 1.0, A, K, B, N, 0.0, C, N);
```

```zig
// Tenzor
const result = a.matmul(b).eval(allocator);
```

### Performance

For large matrices, BLAS may be faster due to:
- Decades of optimization
- Platform-specific tuning
- Assembly kernels

Tenzor advantages:
- No external dependency
- Fusion with surrounding ops
- Smaller binaries

---

## Feature Matrix

| Feature | Tenzor | NumPy | PyTorch | Eigen |
|---------|--------|-------|---------|-------|
| **Shapes** |
| Compile-time | ✓ | ✗ | ✗ | ◐ |
| Dynamic | ✗ | ✓ | ✓ | ✓ |
| **Operations** |
| Element-wise | ✓ | ✓ | ✓ | ✓ |
| Matrix multiply | ✓ | ✓ | ✓ | ✓ |
| Convolution | ✗ | ◐ | ✓ | ✗ |
| **Optimization** |
| SIMD | ✓ | Via BLAS | Via BLAS | ✓ |
| Fusion | ✓ | ✗ | ◐ | ✓ |
| Threading | ✓ | Via BLAS | ✓ | ✓ |
| **Deployment** |
| No runtime | ✓ | ✗ | ✗ | ✓ |
| Embedded | ✓ | ✗ | ✗ | ◐ |
| Static linking | ✓ | ✗ | ✗ | ✓ |

Legend: ✓ = Yes, ✗ = No, ◐ = Partial

---

## Migration Guide

### From NumPy

| NumPy | Tenzor |
|-------|--------|
| `np.array([...])` | `Tensor(...).init(.{...})` |
| `np.zeros(shape)` | `Tensor(...).fill(0)` |
| `a + b` | `a.add(b)` |
| `a @ b` | `a.matmul(b)` |
| `np.sum(a, axis=0)` | `a.sum(.{0}, false)` |
| `a.reshape(...)` | `a.reshape(...)` |

### From PyTorch

| PyTorch | Tenzor |
|---------|--------|
| `torch.tensor([...])` | `Tensor(...).init(.{...})` |
| `torch.zeros(shape)` | `Tensor(...).fill(0)` |
| `a + b` | `a.add(b)` |
| `torch.matmul(a, b)` | `a.matmul(b)` |
| `torch.relu(x)` | `x.relu()` |
| `x.view(...)` | `x.reshape(...)` |
