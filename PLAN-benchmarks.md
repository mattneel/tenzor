# Tenzor Benchmark Suite Plan

## Overview

Add a comprehensive Ziterion-based benchmark suite to Tenzor for:
1. **Performance regression detection** in CI
2. **Backend comparison** (CPU vs future GPU/WASM backends)
3. **Optimization guidance** (identify bottlenecks, validate improvements)
4. **Documentation** (concrete performance numbers for users)

## Design Principles

- **Comptime parameterization**: Generate benchmarks for multiple sizes/configs at compile time
- **Statistical rigor**: Ziterion provides warmup, outlier detection, confidence intervals
- **Throughput-focused**: Report FLOPS/bandwidth, not just time
- **Reproducible**: Deterministic data initialization, baseline comparison
- **Backend-agnostic**: Abstract interface for future GPU/WASM backends

---

## Phase 1: Infrastructure Setup

### 1.1 Directory Structure

```
tenzor/
├── bench/
│   ├── build.zig              # Benchmark build configuration
│   ├── src/
│   │   ├── main.zig           # Entry point, CLI filtering
│   │   ├── utils.zig          # Common utilities (data gen, validation)
│   │   │
│   │   ├── kernels/           # Individual kernel benchmarks
│   │   │   ├── matmul.zig
│   │   │   ├── conv2d.zig
│   │   │   ├── softmax.zig
│   │   │   ├── layernorm.zig
│   │   │   ├── elementwise.zig
│   │   │   ├── reduce.zig
│   │   │   ├── maxpool.zig
│   │   │   └── gather.zig
│   │   │
│   │   ├── models/            # End-to-end model benchmarks
│   │   │   ├── lenet.zig
│   │   │   └── arctic.zig
│   │   │
│   │   └── scaling/           # Threading/parallelism benchmarks
│   │       ├── thread_scaling.zig
│   │       └── batch_scaling.zig
│   │
│   └── baselines/
│       ├── cpu_x86_64.json    # x86_64 baseline
│       ├── cpu_aarch64.json   # ARM baseline
│       └── wasm.json          # WASM baseline (future)
│
├── build.zig                  # Add bench step
└── build.zig.zon              # Add ziterion dependency
```

### 1.2 Dependency Setup

**build.zig.zon:**
```zig
.dependencies = .{
    .ziterion = .{
        .path = "../ziterion",  // Local path for development
        // .url = "..." for release
    },
    // ... existing deps
},
```

**build.zig additions:**
```zig
// Benchmark executable
const bench_exe = b.addExecutable(.{
    .name = "tenzor-bench",
    .root_source_file = b.path("bench/src/main.zig"),
    .target = target,
    .optimize = .ReleaseFast,  // Always ReleaseFast for benchmarks
});

// Add dependencies
const ziterion_dep = b.dependency("ziterion", .{
    .target = target,
    .optimize = .ReleaseFast,
});
bench_exe.root_module.addImport("ziterion", ziterion_dep.module("ziterion"));
bench_exe.root_module.addImport("tenzor", mod);

// Bench step
const bench_step = b.step("bench", "Run performance benchmarks");
const bench_run = b.addRunArtifact(bench_exe);
if (b.args) |args| bench_run.addArgs(args);
bench_step.dependOn(&bench_run.step);

// Quick bench (subset for CI)
const quick_bench_step = b.step("bench-quick", "Run quick benchmark subset");
const quick_bench_run = b.addRunArtifact(bench_exe);
quick_bench_run.addArg("--quick");
quick_bench_step.dependOn(&quick_bench_run.step);
```

### 1.3 Main Entry Point

**bench/src/main.zig:**
```zig
const std = @import("std");
const ziterion = @import("ziterion").ziterion;

// Import all benchmark modules
const matmul_bench = @import("kernels/matmul.zig");
const conv2d_bench = @import("kernels/conv2d.zig");
const softmax_bench = @import("kernels/softmax.zig");
const layernorm_bench = @import("kernels/layernorm.zig");
const elementwise_bench = @import("kernels/elementwise.zig");
const reduce_bench = @import("kernels/reduce.zig");
const lenet_bench = @import("models/lenet.zig");
const arctic_bench = @import("models/arctic.zig");
const scaling_bench = @import("scaling/thread_scaling.zig");

// Aggregate all benchmarks
pub const benchmarks = struct {
    // Matmul benchmarks
    pub usingnamespace matmul_bench.benchmarks;
    // Conv2d benchmarks
    pub usingnamespace conv2d_bench.benchmarks;
    // Softmax benchmarks
    pub usingnamespace softmax_bench.benchmarks;
    // ... etc
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    try ziterion.run(benchmarks, .{
        .allocator = gpa.allocator(),
        .config = .{
            .warmup_time_ns = 100 * std.time.ns_per_ms,
            .measurement_time_ns = 3 * std.time.ns_per_s,
            .min_samples = 50,
            .max_samples = 1000,
            .target_cv = 0.02,  // Stop at 2% CV
            .output_format = .terminal,
            .verbosity = .normal,
        },
    });
}
```

---

## Phase 2: Kernel Benchmarks

### 2.1 Matrix Multiplication (Highest Priority)

**bench/src/kernels/matmul.zig:**

```zig
const std = @import("std");
const ziterion = @import("ziterion").ziterion;
const tenzor = @import("tenzor");
const matmul = tenzor.backend.cpu.kernels.matmul;
const threading = tenzor.backend.cpu.threading;

// Parameterized benchmark generator
fn makeMatmulBench(
    comptime M: usize,
    comptime K: usize,
    comptime N: usize,
    comptime variant: enum { naive, tiled, parallel },
) fn (*ziterion.State) void {
    return struct {
        fn benchmark(state: *ziterion.State) void {
            // Setup outside measurement loop
            var a: [M * K]f32 = undefined;
            var b: [K * N]f32 = undefined;
            var c: [M * N]f32 = undefined;

            // Initialize with deterministic data
            for (&a, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 17)) * 0.1;
            for (&b, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 13)) * 0.1;

            // Report throughput: 2*M*K*N FLOPs per matmul
            const flops = 2 * M * K * N;
            state.setItemsProcessed(flops);
            state.setBytesProcessed((M * K + K * N + M * N) * @sizeOf(f32));

            // Thread pool for parallel variant
            var pool: ?*threading.ThreadPool = null;
            if (variant == .parallel) {
                pool = threading.ThreadPool.create(state.scratchAllocator(), .{}) catch null;
            }
            defer if (pool) |p| p.destroy();

            while (state.next()) {
                switch (variant) {
                    .naive => matmul.matmulNaive(f32, &a, &b, &c, M, K, N),
                    .tiled => matmul.matmulTiled(f32, &a, &b, &c, M, K, N),
                    .parallel => matmul.matmulTiledParallel(f32, pool.?, &a, &b, &c, M, K, N),
                }
                ziterion.doNotOptimize(c);
            }
        }
    }.benchmark;
}

pub const benchmarks = struct {
    // Small matrices (L1 cache resident)
    pub const matmul_naive_64x64x64 = makeMatmulBench(64, 64, 64, .naive);
    pub const matmul_tiled_64x64x64 = makeMatmulBench(64, 64, 64, .tiled);

    // Medium matrices (L2/L3 cache)
    pub const matmul_naive_256x256x256 = makeMatmulBench(256, 256, 256, .naive);
    pub const matmul_tiled_256x256x256 = makeMatmulBench(256, 256, 256, .tiled);
    pub const matmul_parallel_256x256x256 = makeMatmulBench(256, 256, 256, .parallel);

    // Large matrices (memory bound)
    pub const matmul_tiled_1024x1024x1024 = makeMatmulBench(1024, 1024, 1024, .tiled);
    pub const matmul_parallel_1024x1024x1024 = makeMatmulBench(1024, 1024, 1024, .parallel);

    // Transformer-relevant shapes (embedding dim × seq_len)
    pub const matmul_tiled_384x512x384 = makeMatmulBench(384, 512, 384, .tiled);   // Arctic-like
    pub const matmul_parallel_384x512x384 = makeMatmulBench(384, 512, 384, .parallel);

    // Batched matmul
    pub const batched_matmul_32x128x64x128 = makeBatchedMatmulBench(32, 128, 64, 128);
};
```

**Key sizes to benchmark:**

| Size | Category | Relevance |
|------|----------|-----------|
| 64×64×64 | L1-resident | Micro-kernel efficiency |
| 256×256×256 | L2/L3 | Tiling effectiveness |
| 1024×1024×1024 | Memory-bound | Bandwidth utilization |
| 384×512×384 | Transformer | Arctic embedding dim |
| Batched [32,128,64,128] | Batch inference | Real workload |

### 2.2 Convolution

**bench/src/kernels/conv2d.zig:**

```zig
fn makeConv2dBench(
    comptime batch: usize,
    comptime H: usize,
    comptime W: usize,
    comptime C_in: usize,
    comptime C_out: usize,
    comptime K: usize,
    comptime parallel: bool,
) fn (*ziterion.State) void {
    return struct {
        fn benchmark(state: *ziterion.State) void {
            const out_h = H - K + 1;
            const out_w = W - K + 1;

            var input: [batch * H * W * C_in]f32 = undefined;
            var weight: [C_out * K * K * C_in]f32 = undefined;
            var bias: [C_out]f32 = undefined;
            var output: [batch * out_h * out_w * C_out]f32 = undefined;

            // Initialize...

            // FLOPs: batch * out_h * out_w * C_out * (2 * K * K * C_in)
            const flops = batch * out_h * out_w * C_out * 2 * K * K * C_in;
            state.setItemsProcessed(flops);

            var pool: ?*threading.ThreadPool = null;
            if (parallel) pool = threading.ThreadPool.create(...) catch null;
            defer if (pool) |p| p.destroy();

            while (state.next()) {
                if (parallel) {
                    conv2d.conv2dForwardParallel(f32, pool.?, ...);
                } else {
                    conv2d.conv2dForward(f32, ...);
                }
                ziterion.doNotOptimize(output);
            }
        }
    }.benchmark;
}

pub const benchmarks = struct {
    // LeNet conv1: 28x28x1 -> 24x24x6, 5x5 kernel
    pub const conv2d_lenet1_b1 = makeConv2dBench(1, 28, 28, 1, 6, 5, false);
    pub const conv2d_lenet1_b64 = makeConv2dBench(64, 28, 28, 1, 6, 5, false);
    pub const conv2d_lenet1_b64_par = makeConv2dBench(64, 28, 28, 1, 6, 5, true);

    // LeNet conv2: 12x12x6 -> 8x8x16, 5x5 kernel
    pub const conv2d_lenet2_b64 = makeConv2dBench(64, 12, 12, 6, 16, 5, false);
    pub const conv2d_lenet2_b64_par = makeConv2dBench(64, 12, 12, 6, 16, 5, true);

    // Larger conv (ResNet-like)
    pub const conv2d_resnet_56x56x64_3x3 = makeConv2dBench(1, 56, 56, 64, 64, 3, false);
    pub const conv2d_resnet_56x56x64_3x3_par = makeConv2dBench(1, 56, 56, 64, 64, 3, true);
};
```

### 2.3 Softmax & LayerNorm (Transformer-critical)

```zig
// bench/src/kernels/softmax.zig
fn makeSoftmaxBench(comptime rows: usize, comptime cols: usize, comptime parallel: bool) ...

pub const benchmarks = struct {
    // Attention scores: [batch, heads, seq, seq]
    pub const softmax_512x512 = makeSoftmaxBench(512, 512, false);        // Attention matrix
    pub const softmax_512x512_par = makeSoftmaxBench(512, 512, true);
    pub const softmax_8192x384 = makeSoftmaxBench(8192, 384, false);      // Batch × embedding
    pub const softmax_8192x384_par = makeSoftmaxBench(8192, 384, true);
};

// bench/src/kernels/layernorm.zig
pub const benchmarks = struct {
    pub const layernorm_512x384 = makeLayerNormBench(512, 384, false);    // Arctic embedding
    pub const layernorm_512x384_par = makeLayerNormBench(512, 384, true);
    pub const layernorm_2048x768 = makeLayerNormBench(2048, 768, false);  // BERT-like
};
```

### 2.4 Element-wise Operations

```zig
// bench/src/kernels/elementwise.zig
fn makeElementwiseBench(comptime size: usize, comptime op: ElementwiseOp) ...

pub const benchmarks = struct {
    // Memory bandwidth bound - measure peak throughput
    pub const relu_1M = makeElementwiseBench(1_000_000, .relu);
    pub const exp_1M = makeElementwiseBench(1_000_000, .exp);
    pub const gelu_1M = makeElementwiseBench(1_000_000, .gelu);
    pub const add_1M = makeBinaryBench(1_000_000, .add);
    pub const mul_1M = makeBinaryBench(1_000_000, .mul);

    // Small (cache resident) - measure compute
    pub const relu_64K = makeElementwiseBench(65536, .relu);
    pub const exp_64K = makeElementwiseBench(65536, .exp);
};
```

### 2.5 Reductions

```zig
// bench/src/kernels/reduce.zig
pub const benchmarks = struct {
    pub const reduce_sum_1M = makeReduceBench(1_000_000, .sum);
    pub const reduce_max_1M = makeReduceBench(1_000_000, .max);
    pub const reduce_mean_1M = makeReduceBench(1_000_000, .mean);

    // Axis reductions (for softmax denominators, etc.)
    pub const reduce_axis_1024x1024 = makeAxisReduceBench(1024, 1024, .sum, 1);
};
```

---

## Phase 3: Model Benchmarks

### 3.1 LeNet-5 End-to-End

```zig
// bench/src/models/lenet.zig
pub const benchmarks = struct {
    // Single image inference
    pub fn lenet_forward_b1(state: *ziterion.State) void {
        var model = lenet.LeNet.init(state.scratchAllocator(), .{ .batch_size = 1 }, null) catch return;
        var input: [1 * 28 * 28]f32 = ...;

        state.setItemsProcessed(1);  // 1 image
        while (state.next()) {
            _ = model.forward(&input, 1);
            ziterion.doNotOptimize(model.cache.fc3_out);
        }
    }

    // Batch inference
    pub fn lenet_forward_b64(state: *ziterion.State) void { ... }
    pub fn lenet_forward_b64_parallel(state: *ziterion.State) void { ... }

    // Training step (forward + backward + update)
    pub fn lenet_train_step_b64(state: *ziterion.State) void {
        // Forward
        _ = model.forward(&batch_input, 64);
        // Backward
        model.backward(&batch_input, 64);
        // SGD update (inline, no optimizer object)
        // Report: images/sec
        state.setItemsProcessed(64);
    }
};
```

### 3.2 Arctic-Embed-XS Inference

```zig
// bench/src/models/arctic.zig
pub const benchmarks = struct {
    // Single text embedding
    pub fn arctic_embed_single(state: *ziterion.State) void {
        // Load model weights (pause timer during setup)
        state.pauseTimer();
        var ctx = arctic.InferenceContext.init(allocator, weights) catch return;
        defer ctx.deinit();
        state.resumeTimer();

        const tokens: [128]u32 = ...;  // Pre-tokenized
        state.setItemsProcessed(128);  // Tokens processed

        while (state.next()) {
            const embedding = ctx.forward(&tokens, 128);
            ziterion.doNotOptimize(embedding);
        }
    }

    // Batch embedding (4 texts)
    pub fn arctic_embed_batch4(state: *ziterion.State) void { ... }

    // With parallel attention
    pub fn arctic_embed_single_parallel(state: *ziterion.State) void { ... }
};
```

---

## Phase 4: Scaling Benchmarks

### 4.1 Thread Scaling Analysis

```zig
// bench/src/scaling/thread_scaling.zig

// Generate benchmarks for 1, 2, 4, 8, 16 threads
fn makeScalingBench(comptime thread_count: usize) fn (*ziterion.State) void {
    return struct {
        fn benchmark(state: *ziterion.State) void {
            var pool = threading.ThreadPool.create(allocator, .{
                .num_threads = thread_count,
            }) catch return;
            defer pool.destroy();

            // Run matmul 1024x1024x1024 (representative)
            state.setItemsProcessed(2 * 1024 * 1024 * 1024);  // 2 GFLOP

            while (state.next()) {
                matmul.matmulTiledParallel(f32, pool, ...);
                ziterion.doNotOptimize(c);
            }
        }
    }.benchmark;
}

pub const benchmarks = struct {
    pub const scaling_matmul_t1 = makeScalingBench(1);
    pub const scaling_matmul_t2 = makeScalingBench(2);
    pub const scaling_matmul_t4 = makeScalingBench(4);
    pub const scaling_matmul_t8 = makeScalingBench(8);
    pub const scaling_matmul_t16 = makeScalingBench(16);
};
```

### 4.2 Batch Size Scaling

```zig
// bench/src/scaling/batch_scaling.zig
pub const benchmarks = struct {
    pub const lenet_b1 = makeBatchBench(1);
    pub const lenet_b4 = makeBatchBench(4);
    pub const lenet_b16 = makeBatchBench(16);
    pub const lenet_b64 = makeBatchBench(64);
    pub const lenet_b256 = makeBatchBench(256);
};
```

---

## Phase 5: CI Integration

### 5.1 GitHub Actions Workflow

**.github/workflows/bench.yml:**
```yaml
name: Benchmarks

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Zig
        uses: goto-bus-stop/setup-zig@v2
        with:
          version: master

      - name: Run quick benchmarks
        run: zig build bench-quick
        env:
          ZITERION_BASELINE: bench/baselines/cpu_x86_64.json
          ZITERION_FAIL_ON_REGRESSION: 1
          ZITERION_REGRESSION_THRESHOLD: 0.15  # 15% regression threshold

      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-results
          path: bench-results.json
```

### 5.2 Baseline Management

```bash
# Generate new baseline
ZITERION_SAVE_BASELINE=bench/baselines/cpu_x86_64.json zig build bench

# Compare against baseline
ZITERION_BASELINE=bench/baselines/cpu_x86_64.json zig build bench

# Fail on regression > 10%
ZITERION_FAIL_ON_REGRESSION=1 ZITERION_REGRESSION_THRESHOLD=0.10 zig build bench
```

---

## Phase 6: Backend Abstraction (Future-Proofing)

### 6.1 Backend Interface

```zig
// bench/src/backend.zig
pub const Backend = enum {
    cpu,
    cuda,  // Future
    wasm,  // Future
};

pub fn getBackend() Backend {
    // Detect at runtime or compile-time
    if (builtin.cpu.arch == .wasm32) return .wasm;
    // Check CUDA availability...
    return .cpu;
}

pub const BackendConfig = struct {
    backend: Backend,
    thread_count: ?usize,
    device_id: ?usize,  // For GPU
};
```

### 6.2 Multi-Backend Benchmark Runner

```zig
// Future: run same benchmark on multiple backends
pub fn runMultiBackend(comptime bench_fn: fn (*State) void, backends: []const Backend) void {
    for (backends) |backend| {
        std.debug.print("Backend: {s}\n", .{@tagName(backend)});
        // Run benchmark with backend-specific setup
    }
}
```

---

## Implementation Order

### Week 1: Infrastructure
1. [ ] Add ziterion dependency to build.zig.zon
2. [ ] Create bench/ directory structure
3. [ ] Implement bench/build.zig
4. [ ] Create main.zig with benchmark aggregation
5. [ ] Add `zig build bench` and `zig build bench-quick` steps

### Week 2: Core Kernel Benchmarks
1. [ ] matmul benchmarks (all variants, multiple sizes)
2. [ ] conv2d benchmarks (LeNet shapes)
3. [ ] elementwise benchmarks (relu, exp, gelu)
4. [ ] reduce benchmarks

### Week 3: Transformer Kernel Benchmarks
1. [ ] softmax benchmarks (attention-relevant sizes)
2. [ ] layernorm benchmarks
3. [ ] gather/embedding benchmarks

### Week 4: Model & Scaling Benchmarks
1. [ ] LeNet end-to-end benchmarks
2. [ ] Arctic inference benchmarks
3. [ ] Thread scaling benchmarks
4. [ ] Batch scaling benchmarks

### Week 5: CI & Polish
1. [ ] Generate initial baselines
2. [ ] Set up GitHub Actions workflow
3. [ ] Add benchmark documentation
4. [ ] Performance analysis and optimization targets

---

## Expected Outputs

### Terminal Output Example
```
Running 47 benchmark(s)...

Kernel: matmul
  matmul_naive_64x64x64         125.3 µs   +/- 1.2%   |  2.10 GFLOPS
  matmul_tiled_64x64x64          45.2 µs   +/- 0.8%   |  5.82 GFLOPS
  matmul_tiled_256x256x256      8.45 ms    +/- 1.5%   |  3.97 GFLOPS
  matmul_parallel_256x256x256   1.23 ms    +/- 2.1%   |  27.3 GFLOPS  (8 threads)

Kernel: conv2d
  conv2d_lenet1_b64            12.3 ms    +/- 1.8%   |  142K img/s
  conv2d_lenet1_b64_par         2.1 ms    +/- 2.5%   |  832K img/s

Model: lenet
  lenet_forward_b1              0.82 ms   +/- 1.1%   |  1.22K img/s
  lenet_forward_b64            18.5 ms    +/- 1.4%   |  3.46K img/s
  lenet_train_step_b64         52.3 ms    +/- 2.0%   |  1.22K img/s

Scaling Analysis:
  scaling_matmul_t1            45.2 ms    |  1.00x baseline
  scaling_matmul_t2            23.8 ms    |  1.90x speedup
  scaling_matmul_t4            12.5 ms    |  3.62x speedup
  scaling_matmul_t8             7.2 ms    |  6.28x speedup
```

### JSON Output for CI
```json
{
  "benchmarks": [
    {
      "name": "matmul_tiled_256x256x256",
      "metrics": {
        "median_ns": 8450000,
        "throughput_gflops": 3.97,
        "cv": 0.015
      },
      "comparison": {
        "baseline_ns": 8200000,
        "change": "+3.0%",
        "verdict": "no_change"
      }
    }
  ],
  "summary": {
    "total": 47,
    "improved": 3,
    "regressed": 0,
    "no_change": 44
  }
}
```

---

## Success Criteria

1. **Coverage**: All major kernels have benchmarks at multiple sizes
2. **Reproducibility**: <3% coefficient of variation on repeated runs
3. **CI Integration**: Automated regression detection on PRs
4. **Documentation**: Performance numbers in docs/appendix/performance.md
5. **Baseline Comparison**: Can compare CPU vs future backends

---

## References

- [Ziterion Repository](../ziterion/) - Benchmark library
- [Criterion.rs](https://bheisler.github.io/criterion.rs/book/) - Inspiration for statistical approach
- [Google Benchmark](https://github.com/google/benchmark) - C++ patterns
- [PyTorch Benchmark Utils](https://pytorch.org/docs/stable/benchmark_utils.html) - ML-specific patterns
