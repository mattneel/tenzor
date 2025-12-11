# Performance Benchmarks

This page shows performance benchmarks for Tenzor's kernels and models.

> **Interactive Report**: View the [full interactive benchmark report](benchmarks-report.html) with detailed statistics, distributions, and sortable tables.

---

## Summary

Benchmarks run on the current system with `zig build bench-report`. Results show median time per operation.

### Matrix Multiplication

| Benchmark | Size | Time | Throughput |
|-----------|------|------|------------|
| matmul_naive | 64x64x64 | ~7 μs | 71 G/s |
| matmul_tiled | 64x64x64 | ~7 μs | 71 G/s |
| matmul_tiled | 256x256x256 | ~460 μs | 73 G/s |
| matmul_parallel | 256x256x256 | ~1.0 ms | 33 G/s |
| matmul_tiled | 512x512x512 | ~6.2 ms | 43 G/s |
| matmul_parallel | 512x512x512 | ~3.6 ms | 75 G/s |

**Observations**:
- Tiled matmul is 1.5x faster than naive at 256³
- Parallel matmul provides 1.7x speedup at 512³

### Convolution 2D

| Benchmark | Configuration | Time |
|-----------|---------------|------|
| LeNet conv1 | 1x1x28x28, 6 filters | ~6 μs |
| LeNet conv1 | 64x1x28x28, 6 filters | ~450 μs |
| LeNet conv2 | 1x6x14x14, 16 filters | ~38 μs |
| LeNet conv2 | 64x6x14x14, 16 filters | ~2.4 ms |
| 3x3 conv | 64x64x56x56 | ~36 ms |

### Elementwise Operations

| Operation | 64K elements | 1M elements |
|-----------|--------------|-------------|
| ReLU | ~13 μs | ~330 μs |
| exp | ~235 μs | ~3.7 ms |
| tanh | ~286 μs | ~4.5 ms |
| add (binary) | ~21 μs | ~490 μs |
| mul (binary) | ~21 μs | ~490 μs |

### Reductions

| Operation | 64K elements | 1M elements |
|-----------|--------------|-------------|
| sum | ~8 μs | ~150 μs |
| max | ~19 μs | ~307 μs |
| mean | ~8 μs | ~150 μs |
| argmax | ~53 μs | ~840 μs |

### Softmax

| Size | Time | Throughput |
|------|------|------------|
| 64x64 | ~20 μs | 984 M/s |
| 128x128 | ~81 μs | 991 M/s |
| 256x256 | ~321 μs | 989 M/s |
| 512x512 | ~1.3 ms | 987 M/s |

### Layer Normalization

| Configuration | Time | Throughput |
|---------------|------|------------|
| 128x384 (Arctic) | ~13 μs | 24 G/s |
| 512x384 | ~55 μs | 22 G/s |
| 128x768 (BERT) | ~26 μs | 23 G/s |
| 512x768 | ~106 μs | 23 G/s |

### LeNet-5 Model

| Benchmark | Batch Size | Time | Images/sec |
|-----------|------------|------|------------|
| Forward | 1 | ~443 μs | 2.3K |
| Forward | 16 | ~1.4 ms | 11.4K |
| Forward | 64 | ~4.5 ms | 14.2K |
| Training | 16 | ~3.2 ms | 5.0K |
| Training | 64 | ~12 ms | 5.3K |

### Thread Scaling (512x512x512 matmul)

| Threads | Time | Speedup |
|---------|------|---------|
| 1 | ~14.8 ms | 1.0x |
| 2 | ~9.3 ms | 1.6x |
| 4 | ~8.1 ms | 1.8x |
| 8 | ~4.9 ms | 3.0x |

---

## Running Benchmarks

### Quick Benchmark Run

```bash
zig build bench
```

### Generate HTML Report

```bash
zig build bench-report
```

This generates `docs/src/appendix/benchmarks.html` with interactive tables and detailed statistics.

---

## PyTorch CPU Reference Comparison

Comparison with PyTorch 2.8.0 CPU (with MKL/BLAS) on the same hardware.

### Matrix Multiplication (512x512x512)

| Implementation | Threads | Time | Throughput |
|----------------|---------|------|------------|
| PyTorch | 1 | 2.44 ms | 110 G/s |
| PyTorch | 24 | 300 μs | 894 G/s |
| Tenzor tiled | 1 | 6.2 ms | 43 G/s |
| Tenzor parallel | 8 | 3.6 ms | 75 G/s |

**Notes**: PyTorch uses Intel MKL which has highly optimized BLAS routines. Tenzor's pure-Zig implementation is ~2.5x slower single-threaded.

### LeNet-5 Forward Pass

| Implementation | Batch | Time | Images/sec |
|----------------|-------|------|------------|
| PyTorch | 1 | 81 μs | 12.3K |
| PyTorch | 64 | 1.46 ms | 44K |
| PyTorch (24 threads) | 1 | 1.0 ms | 1.0K |
| PyTorch (24 threads) | 64 | 436 μs | 147K |
| Tenzor | 1 | 443 μs | 2.3K |
| Tenzor | 64 | 4.5 ms | 14K |

**Notes**: Tenzor is 5.5x slower than single-threaded PyTorch for batch=1, but 2.8x faster than 24-thread PyTorch at batch=1 (threading overhead). For larger batches, PyTorch's parallelism wins.

### Elementwise (1M elements)

| Operation | PyTorch (1 thread) | PyTorch (24 threads) | Tenzor |
|-----------|-------------------|----------------------|--------|
| ReLU | 118 μs | 25 μs | 330 μs |
| exp | 173 μs | 37 μs | 3.7 ms |
| tanh | 1.07 ms | 146 μs | 4.5 ms |
| add | 159 μs | 26 μs | 490 μs |

**Notes**: PyTorch uses vectorized intrinsics from MKL-VML for transcendentals. Tenzor uses pure SIMD without external dependencies.

### Softmax (256x256)

| Implementation | Time | Throughput |
|----------------|------|------------|
| PyTorch (1 thread) | 35 μs | 9.5 G/s |
| PyTorch (24 threads) | 13 μs | 25 G/s |
| Tenzor | 321 μs | 989 M/s |

### Key Takeaways

1. **PyTorch wins on raw compute**: MKL/BLAS gives 2-10x speedup for matrix ops
2. **Tenzor wins on simplicity**: No external dependencies, single binary, ~150KB
3. **Threading overhead**: PyTorch's 24-thread overhead hurts small batch sizes
4. **Compilation**: Tenzor compiles in <1s; PyTorch has Python startup overhead

### When to Use Tenzor

- Embedded/edge deployment (no Python, no MKL dependency)
- WebAssembly targets
- Real-time applications where startup latency matters
- Learning/educational purposes (readable pure-Zig implementation)
- Scenarios where single-digit millisecond latency is acceptable

### When to Use PyTorch

- Maximum throughput is critical
- Large batch training/inference
- GPU acceleration needed
- Ecosystem (pretrained models, tooling)

---

## Methodology

- **Warmup**: 50ms warmup phase to stabilize CPU frequency
- **Measurement**: 500ms measurement window
- **Samples**: 30-500 samples per benchmark
- **Statistics**: Median reported (robust to outliers)
- **Build**: ReleaseFast optimization

All benchmarks use the [Ziterion](https://github.com/mattneel/ziterion) benchmarking library.
