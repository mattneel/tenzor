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

## Methodology

- **Warmup**: 50ms warmup phase to stabilize CPU frequency
- **Measurement**: 500ms measurement window
- **Samples**: 30-500 samples per benchmark
- **Statistics**: Median reported (robust to outliers)
- **Build**: ReleaseFast optimization

All benchmarks use the [Ziterion](https://github.com/mattneel/ziterion) benchmarking library.
