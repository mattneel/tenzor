# tenzor

**A comptime tensor library for Zig with zero-cost abstractions, automatic operation fusion, and multi-backend execution.**

Version: 0.1.0-draft  
Status: Specification  
Last Updated: December 2024

---

## Table of Contents

1. [Vision](#vision)
2. [Core Insight](#core-insight)
3. [Architecture Overview](#architecture-overview)
4. [Expression Graph System](#expression-graph-system)
5. [Type System Design](#type-system-design)
6. [Backend Abstraction Layer](#backend-abstraction-layer)
7. [CPU Backend (Reference)](#cpu-backend-reference)
8. [CUDA Backend](#cuda-backend)
9. [Vulkan/SPIR-V Backend](#vulkanspir-v-backend)
10. [Metal Backend](#metal-backend)
11. [WebGPU Backend](#webgpu-backend)
12. [Fusion Engine](#fusion-engine)
13. [Autograd System](#autograd-system)
14. [Memory Management](#memory-management)
15. [C FFI Layer](#c-ffi-layer)
16. [Python Bindings](#python-bindings)
17. [Testing Strategy](#testing-strategy)
18. [Benchmarking](#benchmarking)
19. [Build System](#build-system)
20. [Roadmap](#roadmap)

---

## Vision

tenzor aims to be a serious alternative to PyTorch, JAX, and TensorFlow for performance-critical tensor computation. The key differentiators:

| Feature | PyTorch | JAX/XLA | tenzor |
|---------|---------|---------|--------|
| Shape errors | Runtime | Runtime | **Compile time** |
| Op fusion | None (eager) | JIT trace | **Comptime** |
| Fusion overhead | N/A | Trace + compile | **Zero (at compile)** |
| GPU backends | CUDA only | CUDA/TPU | **CUDA/Vulkan/Metal/WebGPU** |
| WASM support | No | No | **Yes** |
| Binary size | 100s MB | 100s MB | **KBs-MBs** |
| Dependencies | Python + libs | Python + libs | **None** |
| Language | Python (C++ core) | Python (C++ core) | **Zig (with C FFI)** |

**Philosophy**: Compile time is free. Runtime is not.

---

## Core Insight

Tensor operations don't execute—they build a comptime expression graph. Fusion happens at compile time by analyzing the graph and generating optimal kernels.

```
// Conceptually:
const expr = a.matmul(b).relu().add(bias);

// At comptime, expr's type encodes:
// Add(Relu(Matmul(Tensor("a"), Tensor("b"))), Tensor("bias"))

// Only .eval() executes—comptime generates a fused kernel
const result = expr.eval(allocator);
```

The trick: operations return comptime types encoding the computation, not runtime results. The type system IS the expression graph.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         User API                                 │
│  Tensor(f32, .{64, 128}) / .matmul() / .relu() / .eval()        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Expression Graph (Comptime)                   │
│  Type-encoded DAG: Matmul(T, T) → Relu(_) → Add(_, T)           │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Fusion Engine (Comptime)                    │
│  Pattern matching, fusion decisions, kernel planning            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Backend Abstraction Layer                      │
│  Unified interface for code generation and execution            │
└─────────────────────────────────────────────────────────────────┘
          │           │           │           │           │
          ▼           ▼           ▼           ▼           ▼
┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐
│    CPU     │ │   CUDA     │ │  Vulkan    │ │   Metal    │ │  WebGPU    │
│   SIMD     │ │   PTX      │ │  SPIR-V    │ │   MSL      │ │   WGSL     │
└────────────┘ └────────────┘ └────────────┘ └────────────┘ └────────────┘
```

---

## Expression Graph System

### Node Types

The expression graph consists of typed nodes representing operations:

```
ExprNode ::= 
    | Tensor(T, Shape)              -- Leaf: input tensor
    | Constant(T, Shape, value)     -- Leaf: compile-time constant
    | UnaryOp(op, Input)            -- Unary: relu, tanh, exp, log, neg, sqrt, rsqrt, sigmoid, gelu, silu
    | BinaryOp(op, Lhs, Rhs)        -- Binary: add, sub, mul, div, pow, max, min
    | Matmul(Lhs, Rhs)              -- Matrix multiply
    | Conv2D(Input, Kernel, opts)   -- 2D convolution
    | Reduce(op, Input, axes)       -- Reduction: sum, mean, max, min, prod
    | Transpose(Input, perm)        -- Axis permutation
    | Reshape(Input, new_shape)     -- View with same elements
    | Broadcast(Input, new_shape)   -- Implicit expansion
    | Slice(Input, ranges)          -- Subview
    | Concat(Inputs, axis)          -- Concatenation
    | Gather(Input, Indices, axis)  -- Index selection
    | Scatter(Input, Indices, Src)  -- Index assignment
```

### Shape Algebra

All shape computation happens at comptime:

```
Shape ::= .{ dim0, dim1, ..., dimN }  -- Tuple of comptime usize

Rules:
- Matmul: [M, K] × [K, N] → [M, N]
- Broadcast: follows NumPy broadcasting semantics
- Reduce(axis=i): removes dimension i (or keeps with size 1 if keepdims)
- Transpose(perm): reorders dimensions according to permutation
- Reshape: product of dimensions must match
```

Shape mismatches are compile errors, not runtime panics.

### Expression Properties (Comptime)

Each expression node exposes comptime properties:

```
Node.T        : type           -- Element type (f16, f32, f64, bf16)
Node.shape    : Shape          -- Output shape tuple
Node.op       : OpTag          -- Operation discriminant
Node.is_leaf  : bool           -- True for Tensor/Constant
Node.rank     : usize          -- Number of dimensions
Node.numel    : usize          -- Total element count (product of shape)
```

---

## Type System Design

### Tensor Type Constructor

```
Tensor(comptime T: type, comptime shape: anytype) type
```

- `T`: Element type. Supported: `f16`, `bf16`, `f32`, `f64`, `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`
- `shape`: Comptime tuple of dimensions, e.g., `.{64, 128, 256}`

### Operation Return Types

Each operation returns a new comptime type encoding the operation:

```
Tensor(f32, .{M, K}).matmul(Tensor(f32, .{K, N}))  →  Matmul(Tensor(f32, .{M,K}), Tensor(f32, .{K,N}))

// The return type encodes:
// - Operation: matmul
// - Input types (preserved for traversal)
// - Output shape: .{M, N}
// - Output dtype: f32
```

### Method Chaining

All expression types expose the same interface, enabling chaining:

```
.matmul(other)    → Matmul type
.add(other)       → Add type
.relu()           → Relu type
.eval(allocator)  → concrete Tensor (terminal)
.evalGpu(device)  → GpuTensor (terminal)
```

### Comptime Validation

Invalid operations fail at compile time:

```
// Compile error: matmul shape mismatch: [64, 128] × [256, 64]
const bad = Tensor(f32, .{64, 128}).matmul(Tensor(f32, .{256, 64}));

// Compile error: cannot broadcast [64, 128] with [64, 64]
const bad2 = Tensor(f32, .{64, 128}).add(Tensor(f32, .{64, 64}));

// Compile error: type mismatch: f32 vs f64
const bad3 = Tensor(f32, .{64}).add(Tensor(f64, .{64}));
```

---

## Backend Abstraction Layer

### Backend Interface

Each backend implements a comptime interface:

```
Backend :: struct {
    // Comptime: generate kernel for expression
    fn genKernel(comptime Expr: type) Kernel

    // Runtime: execute kernel with concrete data
    fn execute(kernel: Kernel, inputs: []const *anyopaque, output: *anyopaque, stream: ?Stream) void

    // Runtime: memory management
    fn alloc(size: usize) *anyopaque
    fn free(ptr: *anyopaque) void
    fn copyToDevice(host: []const u8, device: *anyopaque) void
    fn copyToHost(device: *anyopaque, host: []u8) void

    // Synchronization
    fn synchronize(stream: ?Stream) void
}
```

### Backend Selection

```
// Explicit backend selection
const result = expr.eval(.cpu, allocator);
const result = expr.eval(.cuda, allocator);
const result = expr.eval(.vulkan, allocator);
const result = expr.eval(.metal, allocator);
const result = expr.eval(.webgpu, allocator);

// Auto-selection (prefers GPU if available)
const result = expr.eval(.auto, allocator);

// Default (CPU)
const result = expr.eval(allocator);
```

### Device Abstraction

```
Device :: struct {
    backend: BackendType,
    index: u32,           // For multi-GPU
    properties: DeviceProperties,
}

DeviceProperties :: struct {
    name: []const u8,
    compute_units: u32,
    memory_bytes: u64,
    max_workgroup_size: [3]u32,
    supports_f16: bool,
    supports_bf16: bool,
    // ...
}
```

---

## CPU Backend (Reference)

The CPU backend serves as:
1. Reference implementation for correctness testing
2. Fallback when no GPU available
3. Development/debugging target
4. Baseline for benchmarks

### SIMD Strategy

Use Zig's `@Vector` for portable SIMD:

```
// Comptime-selected vector width
const Vec = @Vector(std.simd.suggestVectorLength(f32) orelse 8, f32);

// Operations map to SIMD instructions:
// vec + vec  →  addps/vaddps (x86) or fadd (ARM)
// vec * vec  →  mulps/vmulps (x86) or fmul (ARM)
// @max(vec, splat(0))  →  maxps/vmaxps (relu)
```

### Kernel Structure

CPU kernels follow this pattern:

```
1. Outer loop: tiles for cache efficiency
2. Middle loop: SIMD vectors
3. Inner: scalar remainder

For matmul: 6-level loop nest with register blocking
```

### Cache Optimization

Comptime tile size selection based on:
- L1 cache size (typically 32KB)
- L2 cache size (typically 256KB-1MB)
- Element size
- Register count

Target: tiles fit in L1, working set fits in L2.

### Threading

Use Zig's `std.Thread.Pool` for parallel execution:
- Partition work across cores
- Each thread processes independent tiles
- No synchronization needed for elementwise ops
- Reduction ops use thread-local accumulators + final merge

---

## CUDA Backend

### Architecture

```
┌────────────────────────────────────────┐
│           Comptime Layer               │
│  Expression analysis, fusion planning  │
└────────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────┐
│         PTX Code Generation            │
│  Comptime string building → PTX ASM    │
└────────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────┐
│          CUDA Driver API               │
│  cuModuleLoadData, cuLaunchKernel      │
└────────────────────────────────────────┘
```

### PTX Generation

Generate PTX assembly as comptime strings:

```
// Advantages of PTX over NVCC:
// - No CUDA toolkit dependency at compile time
// - Comptime string generation (no external compiler invocation)
// - JIT compilation by driver (optimized for target GPU)
// - Smaller binary size

// PTX is forward-compatible: write once, runs on future GPUs
```

PTX generation strategy:
1. Elementwise ops: one thread per element, fused into single kernel
2. Matmul: tiled algorithm with shared memory
3. Reductions: tree reduction in shared memory

### Memory Management

```
- cuMemAlloc / cuMemFree for device memory
- cuMemcpyHtoD / cuMemcpyDtoH for transfers
- Async variants with streams for overlap
- Memory pools for allocation reuse
```

### Kernel Launch

```
cuLaunchKernel(
    function,
    gridDimX, gridDimY, gridDimZ,
    blockDimX, blockDimY, blockDimZ,
    sharedMemBytes,
    stream,
    kernelParams,
    extra
)
```

Grid/block dimensions computed at comptime based on tensor shape.

### Tensor Cores (Future)

For Volta+ GPUs, generate WMMA (Warp Matrix Multiply Accumulate) instructions:
- 16×16×16 matrix tiles
- Mixed precision: f16 inputs, f32 accumulation
- Significant speedup for transformer workloads

---

## Vulkan/SPIR-V Backend

### Why Vulkan?

- Cross-platform: Windows, Linux, Android, (MoltenVK on macOS/iOS)
- Vendor-agnostic: NVIDIA, AMD, Intel, Qualcomm, ARM
- Explicit control: memory, synchronization, scheduling
- Compute shaders: general-purpose GPU compute

### Architecture

```
┌────────────────────────────────────────┐
│           Comptime Layer               │
│  Expression analysis, fusion planning  │
└────────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────┐
│        SPIR-V Code Generation          │
│  Comptime binary building → SPIR-V     │
└────────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────┐
│            Vulkan API                  │
│  vkCreateShaderModule, vkCmdDispatch   │
└────────────────────────────────────────┘
```

### SPIR-V Generation

SPIR-V is a binary format. Generate at comptime:

```
// SPIR-V structure:
// 1. Header (magic, version, generator, bound, schema)
// 2. Capabilities (Shader, ...)
// 3. Extensions
// 4. Imports (GLSL.std.450 for math functions)
// 5. Memory model
// 6. Entry points
// 7. Decorations (bindings, offsets, ...)
// 8. Type declarations
// 9. Constants
// 10. Function definitions
```

Strategy: Build SPIR-V binary as comptime byte array. No external tools needed.

### Descriptor Sets

```
Set 0: Input tensors (storage buffers, read-only)
Set 1: Output tensors (storage buffers, read-write)
Set 2: Uniforms (dimensions, strides, scalars)
```

Layout generated at comptime based on expression leaf count.

### Synchronization

```
- Pipeline barriers between kernels
- Memory barriers for RAW hazards
- Semaphores for multi-queue
- Fences for CPU-GPU sync
```

### Subgroups

Use subgroup operations for efficient reductions:
- `subgroupAdd`, `subgroupMax`, etc.
- Warp-level primitives (32 threads on NVIDIA, 64 on AMD)

---

## Metal Backend

### Why Metal?

- Native on Apple Silicon (M1/M2/M3/M4)
- Unified memory architecture (no explicit copies)
- Metal Performance Shaders (MPS) for optimized primitives
- Required for iOS deployment

### Architecture

```
┌────────────────────────────────────────┐
│           Comptime Layer               │
│  Expression analysis, fusion planning  │
└────────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────┐
│         MSL Code Generation            │
│  Comptime string building → MSL        │
└────────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────┐
│            Metal API                   │
│  MTLDevice, MTLCommandQueue, dispatch  │
└────────────────────────────────────────┘
```

### MSL Generation

Metal Shading Language (MSL) is C++-like. Generate as comptime strings:

```
kernel void fused_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    output[gid] = max(0.0f, input[gid] * weight[gid] + bias[gid]);
}
```

### Unified Memory

On Apple Silicon, CPU and GPU share memory:
- `MTLResourceStorageModeShared`: zero-copy access
- No explicit transfers needed
- Significant performance advantage

### MPS Integration (Optional)

For operations where MPS is faster than generated code:
- `MPSMatrixMultiplication` for large matmuls
- `MPSNNConvolution` for convolutions
- Fallback to generated code for fused patterns MPS doesn't support

### Threadgroup Memory

Metal equivalent of shared memory:
- `threadgroup float tile[16][16]`
- Used for tiled matmul, reductions
- Size limits vary by GPU

---

## WebGPU Backend

### Why WebGPU?

- Browser deployment (WASM + WebGPU)
- Cross-platform abstraction (Dawn on native)
- Future-proof (replacing WebGL)
- Zig → WASM is first-class

### Architecture

```
┌────────────────────────────────────────┐
│           Comptime Layer               │
│  Expression analysis, fusion planning  │
└────────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────┐
│         WGSL Code Generation           │
│  Comptime string building → WGSL       │
└────────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────┐
│           WebGPU API                   │
│  wgpu::Device, createComputePipeline   │
└────────────────────────────────────────┘
```

### WGSL Generation

WebGPU Shading Language. Generate as comptime strings:

```
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&output)) { return; }
    output[i] = max(0.0, input[i] * weight[i] + bias[i]);
}
```

### Browser Deployment

```
1. Compile Zig to WASM (zig build -Dtarget=wasm32-freestanding)
2. WASM module exports: init, createTensor, eval, ...
3. JavaScript glue: WebGPU device management, buffer creation
4. Generated WGSL kernels embedded in WASM or passed to JS
```

### Native via Dawn/wgpu-native

For native applications wanting WebGPU abstraction:
- Dawn (Google's WebGPU implementation)
- wgpu-native (Rust wgpu with C API)

Both provide the same API, backend translates to Vulkan/Metal/D3D12.

### Workgroup Considerations

- Max workgroup size: typically 256
- Workgroup memory: limited (~16KB typically)
- No subgroup operations in base WebGPU (extension pending)

---

## Fusion Engine

### Fusion Categories

```
1. Elementwise Fusion
   - All pointwise ops (relu, add, mul, tanh, ...)
   - Fuse into single loop/kernel
   - Zero intermediate allocations

2. Matmul Epilogue Fusion
   - Matmul + elementwise ops on output
   - Fuse into matmul's write loop
   - Example: matmul → add(bias) → relu

3. Reduction Fusion
   - Elementwise before reduction
   - Example: square → sum (for L2 norm)

4. Broadcast Fusion
   - Implicit broadcast in binary ops
   - No materialization of broadcasted tensor
```

### Fusion Decision Algorithm

```
function plan_fusion(expr: ExprNode) -> ExecutionPlan:
    ops = collect_ops(expr)  // comptime traversal
    
    if all_elementwise(ops):
        return FusedElementwise(expr)
    
    if starts_with_matmul(ops) and rest_elementwise(ops[1:]):
        return MatmulWithEpilogue(expr)
    
    if ends_with_reduction(ops) and rest_elementwise(ops[:-1]):
        return ElementwiseIntoReduction(expr)
    
    // Can't fuse everything—find breakpoints
    segments = find_fusable_segments(ops)
    return MultiKernel(segments)
```

### Fusion Barriers

Operations that break fusion:
- Matmul (unless at start of chain)
- Convolution
- Reduction (unless at end)
- Reshape with non-contiguous result
- Operations requiring multiple passes over data

### Code Generation Patterns

**Fused Elementwise:**
```
for each output element i (vectorized):
    output[i] = f(g(h(input[i])))  // nested ops inlined
```

**Matmul + Epilogue:**
```
for each output tile:
    acc = matmul_tile(A_tile, B_tile)
    output_tile = epilogue(acc)  // fused: bias + relu etc
```

**Elementwise + Reduction:**
```
acc = identity
for each input element i (vectorized):
    acc = reduce_op(acc, transform(input[i]))
output = finalize(acc)
```

---

## Autograd System

### Design Philosophy

Autograd is built on the same comptime expression graph:

```
// Forward pass builds expression
const y = x.matmul(w).add(b).relu();
const loss = y.sub(target).square().mean();

// Backward pass is also comptime-generated
const grads = loss.backward();

// Access gradients
const dw = grads.wrt(w);
const db = grads.wrt(b);
```

### Gradient Expression Generation

At comptime, `backward()` constructs the gradient expression graph:

```
For each op, define VJP (vector-Jacobian product):

vjp(Add, grad_out) = (grad_out, grad_out)
vjp(Mul, grad_out) = (grad_out * rhs, grad_out * lhs)
vjp(Matmul, grad_out) = (grad_out @ rhs.T, lhs.T @ grad_out)
vjp(Relu, grad_out) = grad_out * (input > 0)
vjp(Sum, grad_out) = broadcast(grad_out, input.shape)
...
```

Comptime walks the forward graph in reverse, applying VJPs.

### Gradient Checkpointing

For memory efficiency, support selective recomputation:

```
const y = checkpoint(expensive_subgraph);
// Forward values not stored; recomputed during backward
```

### Higher-Order Gradients

Since gradients are expressions, can differentiate again:

```
const grads = loss.backward();
const hessian_vec = grads.wrt(w).backward();  // Hessian-vector product
```

### No-Grad Regions

```
const result = noGrad(fn() {
    return model.forward(input);
});
// No gradient tracking in this scope
```

---

## Memory Management

### Allocation Strategy

```
1. Eager allocation: allocate on .eval()
2. Memory pool: reuse buffers of same size
3. Arena allocation: bulk free after computation
```

### Comptime Memory Planning

Analyze tensor lifetimes at comptime:

```
const a = input.relu();      // alloc a
const b = a.matmul(w);       // alloc b; a still live
const c = b.add(bias);       // alloc c; can free a
const d = c.relu();          // can reuse a's memory (same size)
```

Generate allocation plan that minimizes peak memory.

### Buffer Aliasing

When safe, reuse input buffer for output (in-place operations):

```
// relu can be in-place if input has refcount 1
output = relu_inplace(input);  // no new allocation
```

Aliasing analysis at comptime.

### GPU Memory

- Dedicated allocator per device
- Memory pools for common sizes
- Async prefetching for streaming workloads
- Unified memory support where available (Apple Silicon, some CUDA)

---

## C FFI Layer

### Design Goals

- Minimal, stable ABI
- No Zig runtime dependency for users
- Opaque handle-based API
- Thread-safe

### API Surface

```c
// Opaque handles
typedef struct tz_tensor* tz_tensor_t;
typedef struct tz_device* tz_device_t;
typedef struct tz_stream* tz_stream_t;

// Device management
tz_device_t tz_device_create(tz_backend backend, int device_index);
void tz_device_destroy(tz_device_t device);

// Tensor creation
tz_tensor_t tz_tensor_create(tz_device_t device, tz_dtype dtype, 
                              const size_t* shape, size_t ndim);
tz_tensor_t tz_tensor_from_data(tz_device_t device, tz_dtype dtype,
                                 const size_t* shape, size_t ndim,
                                 const void* data);
void tz_tensor_destroy(tz_tensor_t tensor);

// Data access
void tz_tensor_copy_to_host(tz_tensor_t tensor, void* dst);
void tz_tensor_copy_from_host(tz_tensor_t tensor, const void* src);

// Operations (eager API for FFI)
tz_tensor_t tz_matmul(tz_tensor_t a, tz_tensor_t b);
tz_tensor_t tz_add(tz_tensor_t a, tz_tensor_t b);
tz_tensor_t tz_relu(tz_tensor_t x);
// ... etc

// Fused operations (pre-compiled common patterns)
tz_tensor_t tz_linear_relu(tz_tensor_t x, tz_tensor_t w, tz_tensor_t b);
tz_tensor_t tz_attention(tz_tensor_t q, tz_tensor_t k, tz_tensor_t v, tz_tensor_t mask);

// Error handling
const char* tz_get_last_error(void);
```

### Compilation

```bash
# Build shared library
zig build -Doptimize=ReleaseFast -Dtarget=x86_64-linux-gnu
# Output: libtenzor.so (Linux), libtenzor.dylib (macOS), tenzor.dll (Windows)
```

### Pre-compiled Kernels

Since FFI can't use comptime fusion, provide pre-fused common patterns:
- `linear_relu`: matmul + bias + relu
- `attention`: scaled dot-product attention
- `layer_norm`: normalization + scale + shift
- `gelu`: GELU activation
- `softmax`: numerically stable softmax

---

## Python Bindings

### Design Goals

- Pythonic API (NumPy-like)
- Zero-copy where possible
- Support for Python buffer protocol
- PyTorch-like autograd interface

### Implementation Options

1. **ctypes/cffi**: Pure Python, uses C FFI
2. **pybind11**: C++ bindings (requires C++ toolchain)
3. **PyO3**: Rust bindings to C FFI (if using Rust wrapper)
4. **Cython**: C extensions (another compilation step)

Recommendation: **cffi** for initial implementation (simplest), potentially pybind11 later for performance.

### API Design

```python
import tenzor as tz

# Device selection
device = tz.device("cuda:0")  # or "cpu", "vulkan", "metal"

# Tensor creation (from NumPy)
x = tz.tensor(np_array, device=device)
w = tz.randn(128, 256, device=device)

# Operations (lazy by default)
y = x @ w
y = y.relu()
y = y + bias

# Evaluation
result = y.eval()  # triggers fusion and execution
result_np = result.numpy()  # copy to NumPy

# Autograd
x = tz.tensor(data, requires_grad=True)
y = model(x)
loss = criterion(y, target)
loss.backward()
print(x.grad)

# Context manager for device
with tz.device("cuda:0"):
    x = tz.randn(64, 128)  # on GPU
```

### NumPy Interop

```python
# Zero-copy from NumPy (if contiguous and CPU)
x = tz.as_tensor(np_array)  # shares memory

# Copy from NumPy
x = tz.tensor(np_array)  # owns memory

# To NumPy
np_array = x.numpy()  # copy to CPU NumPy array
```

### PyTorch Migration

Provide compatibility layer for easy migration:

```python
import tenzor.torch_compat as torch

# Same API as PyTorch
x = torch.randn(64, 128)
y = torch.relu(x @ w + b)
```

---

## Testing Strategy

### Unit Tests

Per-operation correctness:

```
For each op:
    1. Generate random inputs
    2. Compute with tenzor
    3. Compute reference (NumPy or known-good impl)
    4. Assert allclose(result, reference, rtol, atol)
```

### Gradient Tests

Verify autograd correctness:

```
For each differentiable op:
    1. Compute gradient with backward()
    2. Compute numerical gradient (finite differences)
    3. Assert allclose(analytical, numerical)
```

### Fusion Tests

Verify fusion produces correct results:

```
For each fusion pattern:
    1. Compute with fusion
    2. Compute without fusion (separate ops)
    3. Assert allclose(fused, unfused)
```

### Backend Tests

Same test suite runs on all backends:

```
for backend in [cpu, cuda, vulkan, metal, webgpu]:
    run_test_suite(backend)
    compare_results_cross_backend()
```

### Property-Based Tests

Use hypothesis-style testing:

```
For random shapes, dtypes, ops:
    Verify: associativity, commutativity, broadcasting rules
    Verify: gradient correctness
    Verify: backend consistency
```

### Stress Tests

- Large tensors (GB scale)
- Long fusion chains
- Concurrent execution
- Memory pressure

---

## Benchmarking

### Microbenchmarks

Per-operation performance:

```
Operations:
- matmul: varying M, K, N
- elementwise: varying size
- reduction: varying size and axes
- conv2d: varying spatial size, channels, kernel

Metrics:
- FLOPS (for compute-bound)
- GB/s (for memory-bound)
- Latency (for small tensors)
```

### Fusion Benchmarks

Measure fusion benefit:

```
Patterns:
- matmul + bias + relu
- layernorm
- attention (Q @ K.T @ V with masking)
- MLP block

Compare:
- tenzor (fused)
- tenzor (unfused, for reference)
- PyTorch (eager)
- JAX (with jit)
```

### End-to-End Benchmarks

Full model performance:

```
Models:
- ResNet-50
- BERT-base
- GPT-2
- Llama-7B (inference)

Metrics:
- Throughput (samples/sec or tokens/sec)
- Memory usage
- Time to first token (for LLMs)
```

### Comparison Targets

```
CPU:
- NumPy + MKL
- PyTorch CPU
- ONNX Runtime

CUDA:
- PyTorch CUDA
- JAX/XLA
- TensorRT

Apple Silicon:
- PyTorch MPS
- MLX
- CoreML
```

### Benchmark Harness

```
- Warmup runs (exclude from timing)
- Multiple iterations (statistical significance)
- Report: mean, std, min, max, p50, p99
- Memory tracking
- Power measurement (where available)
```

---

## Build System

### build.zig Structure

```
tenzor/
├── build.zig
├── build.zig.zon         # dependencies
├── src/
│   ├── main.zig          # library root
│   ├── core/
│   │   ├── tensor.zig
│   │   ├── shape.zig
│   │   ├── dtype.zig
│   │   └── expr.zig
│   ├── ops/
│   │   ├── elementwise.zig
│   │   ├── matmul.zig
│   │   ├── reduce.zig
│   │   └── ...
│   ├── backends/
│   │   ├── cpu/
│   │   ├── cuda/
│   │   ├── vulkan/
│   │   ├── metal/
│   │   └── webgpu/
│   ├── autograd/
│   ├── memory/
│   └── ffi/
├── tests/
├── benchmarks/
├── examples/
└── bindings/
    ├── c/
    └── python/
```

### Build Options

```bash
# Debug build
zig build

# Release build
zig build -Doptimize=ReleaseFast

# Specific backends
zig build -Dbackends=cpu,cuda

# Cross-compilation
zig build -Dtarget=aarch64-macos
zig build -Dtarget=wasm32-freestanding

# Build shared library for FFI
zig build lib -Doptimize=ReleaseFast

# Run tests
zig build test

# Run benchmarks
zig build bench
```

### Dependencies

External dependencies (all optional):

```
CUDA:     CUDA Driver API headers (cuda.h)
Vulkan:   Vulkan headers (vulkan/vulkan.h)
Metal:    Metal framework (macOS/iOS only)
WebGPU:   wgpu-native or Dawn headers

All loaded dynamically at runtime—no link-time dependency.
```

### CI/CD

```yaml
# Test matrix
os: [ubuntu, macos, windows]
zig: [master, 0.12.0]
backends: [cpu, cuda, vulkan, metal, webgpu]

# Jobs
- build
- test
- benchmark (nightly)
- publish (on tag)
```

---

## Roadmap

### Phase 1: Foundation (MVP)

- [ ] Core tensor type and expression graph
- [ ] Shape algebra with comptime validation
- [ ] Basic ops: matmul, add, mul, relu, tanh, sigmoid
- [ ] CPU backend with SIMD
- [ ] Elementwise fusion
- [ ] Basic test suite

### Phase 2: GPU Backends

- [ ] CUDA backend with PTX generation
- [ ] Vulkan backend with SPIR-V generation
- [ ] Metal backend with MSL generation
- [ ] WebGPU backend with WGSL generation
- [ ] Matmul epilogue fusion
- [ ] Multi-device support

### Phase 3: Autograd

- [ ] Backward pass generation
- [ ] Gradient accumulation
- [ ] Gradient checkpointing
- [ ] Optimizer primitives (SGD, Adam)

### Phase 4: Completeness

- [ ] Convolution ops
- [ ] Attention primitives
- [ ] Normalization ops (LayerNorm, BatchNorm)
- [ ] Reduction ops (sum, mean, max, min)
- [ ] Index ops (gather, scatter)
- [ ] Memory planning

### Phase 5: Ecosystem

- [ ] C FFI
- [ ] Python bindings
- [ ] Pre-trained model loading (ONNX, SafeTensors)
- [ ] Serialization

### Phase 6: Performance

- [ ] Tensor Core support (CUDA)
- [ ] Quantization (int8, int4)
- [ ] Kernel autotuning
- [ ] Graph-level optimization
- [ ] Distributed (multi-GPU, multi-node)

### Phase 7: Production

- [ ] Comprehensive documentation
- [ ] Performance parity with PyTorch
- [ ] Stability and backward compatibility
- [ ] Enterprise features (profiling, debugging)

---

## Appendix A: Op Coverage

### Elementwise Unary

| Op | CPU | CUDA | Vulkan | Metal | WebGPU |
|----|-----|------|--------|-------|--------|
| neg | ✓ | ✓ | ✓ | ✓ | ✓ |
| abs | ✓ | ✓ | ✓ | ✓ | ✓ |
| exp | ✓ | ✓ | ✓ | ✓ | ✓ |
| log | ✓ | ✓ | ✓ | ✓ | ✓ |
| sqrt | ✓ | ✓ | ✓ | ✓ | ✓ |
| rsqrt | ✓ | ✓ | ✓ | ✓ | ✓ |
| sin | ✓ | ✓ | ✓ | ✓ | ✓ |
| cos | ✓ | ✓ | ✓ | ✓ | ✓ |
| tanh | ✓ | ✓ | ✓ | ✓ | ✓ |
| sigmoid | ✓ | ✓ | ✓ | ✓ | ✓ |
| relu | ✓ | ✓ | ✓ | ✓ | ✓ |
| gelu | ✓ | ✓ | ✓ | ✓ | ✓ |
| silu | ✓ | ✓ | ✓ | ✓ | ✓ |

### Elementwise Binary

| Op | CPU | CUDA | Vulkan | Metal | WebGPU |
|----|-----|------|--------|-------|--------|
| add | ✓ | ✓ | ✓ | ✓ | ✓ |
| sub | ✓ | ✓ | ✓ | ✓ | ✓ |
| mul | ✓ | ✓ | ✓ | ✓ | ✓ |
| div | ✓ | ✓ | ✓ | ✓ | ✓ |
| pow | ✓ | ✓ | ✓ | ✓ | ✓ |
| max | ✓ | ✓ | ✓ | ✓ | ✓ |
| min | ✓ | ✓ | ✓ | ✓ | ✓ |

### Matrix/Tensor Ops

| Op | CPU | CUDA | Vulkan | Metal | WebGPU |
|----|-----|------|--------|-------|--------|
| matmul | ✓ | ✓ | ✓ | ✓ | ✓ |
| conv2d | ✓ | ✓ | ✓ | ✓ | ✓ |
| transpose | ✓ | ✓ | ✓ | ✓ | ✓ |

### Reductions

| Op | CPU | CUDA | Vulkan | Metal | WebGPU |
|----|-----|------|--------|-------|--------|
| sum | ✓ | ✓ | ✓ | ✓ | ✓ |
| mean | ✓ | ✓ | ✓ | ✓ | ✓ |
| max | ✓ | ✓ | ✓ | ✓ | ✓ |
| min | ✓ | ✓ | ✓ | ✓ | ✓ |
| prod | ✓ | ✓ | ✓ | ✓ | ✓ |

---

## Appendix B: Error Messages

Shape errors are compile-time with clear diagnostics:

```
error: matmul shape mismatch
  left:  Tensor(f32, .{64, 128})
  right: Tensor(f32, .{256, 64})
  expected right shape: .{128, _}
  
note: matmul requires left.shape[1] == right.shape[0]
```

```
error: broadcast incompatible
  left:  .{64, 128}
  right: .{64, 64}
  
note: shapes must be equal or one must be 1 in each dimension
```

```
error: type mismatch in binary operation
  left:  f32
  right: f64
  
note: operands must have the same dtype
hint: use .cast(f64) to convert
```

---

## Appendix C: Performance Model

### Roofline Analysis

For each kernel, compute:

```
Operational Intensity = FLOPs / Bytes

If OI < machine_balance:
    memory_bound → optimize for bandwidth
Else:
    compute_bound → optimize for FLOPS
```

Machine balance varies by hardware:
- CPU: ~10 FLOP/byte
- GPU: ~50-200 FLOP/byte

### Matmul Performance

Theoretical peak:
```
FLOPS = 2 * M * N * K
Bandwidth = (M*K + K*N + M*N) * sizeof(T)

For large matmul: compute-bound
For skinny matmul: memory-bound
```

### Fusion Benefit

```
Without fusion (N ops, size S):
    Memory traffic = N * 2 * S * sizeof(T)  // read + write each intermediate
    
With fusion:
    Memory traffic = 2 * S * sizeof(T)  // read input + write output
    
Speedup potential = N× for memory-bound ops
```

---

*End of specification.*
