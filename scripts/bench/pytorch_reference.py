#!/usr/bin/env python3
"""PyTorch CPU reference benchmarks for comparison with Tenzor.

Measures equivalent operations to Tenzor benchmarks for fair comparison.

Usage:
    python scripts/bench/pytorch_reference.py
    python scripts/bench/pytorch_reference.py --json  # Output as JSON
"""

import argparse
import json
import time
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BenchResult:
    name: str
    category: str
    median_us: float  # microseconds
    min_us: float
    max_us: float
    samples: int
    throughput: str | None = None


def benchmark(
    fn: Callable[[], None],
    warmup_ms: float = 50,
    measure_ms: float = 500,
    min_samples: int = 30,
) -> tuple[float, float, float, int]:
    """Run benchmark and return (median, min, max, samples) in microseconds."""
    # Warmup
    warmup_end = time.perf_counter() + warmup_ms / 1000
    while time.perf_counter() < warmup_end:
        fn()

    # Measure
    times = []
    measure_end = time.perf_counter() + measure_ms / 1000
    while time.perf_counter() < measure_end or len(times) < min_samples:
        start = time.perf_counter()
        fn()
        elapsed = (time.perf_counter() - start) * 1e6  # to microseconds
        times.append(elapsed)
        if len(times) >= 500:  # cap samples
            break

    times.sort()
    median = times[len(times) // 2]
    return median, min(times), max(times), len(times)


def format_time(us: float) -> str:
    """Format time in appropriate units."""
    if us < 1:
        return f"{us * 1000:.0f} ns"
    elif us < 1000:
        return f"{us:.2f} us"
    elif us < 1_000_000:
        return f"{us / 1000:.2f} ms"
    else:
        return f"{us / 1_000_000:.2f} s"


def format_throughput(ops_per_sec: float) -> str:
    """Format throughput."""
    if ops_per_sec < 1000:
        return f"{ops_per_sec:.1f}/s"
    elif ops_per_sec < 1_000_000:
        return f"{ops_per_sec / 1000:.1f}K/s"
    elif ops_per_sec < 1_000_000_000:
        return f"{ops_per_sec / 1_000_000:.1f}M/s"
    else:
        return f"{ops_per_sec / 1_000_000_000:.1f}G/s"


# =============================================================================
# Matrix Multiplication Benchmarks
# =============================================================================


def bench_matmul(M: int, K: int, N: int) -> BenchResult:
    a = torch.randn(M, K)
    b = torch.randn(K, N)
    c = torch.empty(M, N)

    def fn():
        torch.mm(a, b, out=c)

    median, min_t, max_t, samples = benchmark(fn)
    flops = 2 * M * K * N
    throughput = format_throughput(flops / (median / 1e6))

    return BenchResult(
        name=f"matmul_{M}x{K}x{N}",
        category="Matrix Multiplication",
        median_us=median,
        min_us=min_t,
        max_us=max_t,
        samples=samples,
        throughput=throughput,
    )


# =============================================================================
# Convolution Benchmarks
# =============================================================================


def bench_conv2d(batch: int, in_ch: int, h: int, w: int, out_ch: int, k: int) -> BenchResult:
    x = torch.randn(batch, in_ch, h, w)
    weight = torch.randn(out_ch, in_ch, k, k)
    bias = torch.randn(out_ch)

    def fn():
        F.conv2d(x, weight, bias)

    median, min_t, max_t, samples = benchmark(fn)
    # FLOPs for conv2d: 2 * batch * out_ch * out_h * out_w * in_ch * k * k
    out_h, out_w = h - k + 1, w - k + 1
    flops = 2 * batch * out_ch * out_h * out_w * in_ch * k * k
    throughput = format_throughput(flops / (median / 1e6))

    return BenchResult(
        name=f"conv2d_{batch}x{in_ch}x{h}x{w}_{k}x{k}",
        category="Convolution 2D",
        median_us=median,
        min_us=min_t,
        max_us=max_t,
        samples=samples,
        throughput=throughput,
    )


# =============================================================================
# Elementwise Benchmarks
# =============================================================================


def bench_elementwise(name: str, fn_factory: Callable, size: int) -> BenchResult:
    x = torch.randn(size)
    fn = fn_factory(x)

    median, min_t, max_t, samples = benchmark(fn)
    throughput = format_throughput(size / (median / 1e6))

    size_str = f"{size // 1024}K" if size < 1_000_000 else f"{size // 1_000_000}M"
    return BenchResult(
        name=f"{name}_{size_str}",
        category="Elementwise",
        median_us=median,
        min_us=min_t,
        max_us=max_t,
        samples=samples,
        throughput=throughput,
    )


def bench_binary(name: str, fn_factory: Callable, size: int) -> BenchResult:
    a = torch.randn(size)
    b = torch.randn(size)
    out = torch.empty(size)
    fn = fn_factory(a, b, out)

    median, min_t, max_t, samples = benchmark(fn)
    throughput = format_throughput(size / (median / 1e6))

    size_str = f"{size // 1024}K" if size < 1_000_000 else f"{size // 1_000_000}M"
    return BenchResult(
        name=f"{name}_{size_str}",
        category="Elementwise",
        median_us=median,
        min_us=min_t,
        max_us=max_t,
        samples=samples,
        throughput=throughput,
    )


# =============================================================================
# Reduction Benchmarks
# =============================================================================


def bench_reduction(name: str, fn: Callable, size: int) -> BenchResult:
    x = torch.randn(size)

    median, min_t, max_t, samples = benchmark(lambda: fn(x))
    throughput = format_throughput(size / (median / 1e6))

    size_str = f"{size // 1024}K" if size < 1_000_000 else f"{size // 1_000_000}M"
    return BenchResult(
        name=f"{name}_{size_str}",
        category="Reductions",
        median_us=median,
        min_us=min_t,
        max_us=max_t,
        samples=samples,
        throughput=throughput,
    )


# =============================================================================
# Softmax Benchmarks
# =============================================================================


def bench_softmax(rows: int, cols: int) -> BenchResult:
    x = torch.randn(rows, cols)

    def fn():
        F.softmax(x, dim=-1)

    median, min_t, max_t, samples = benchmark(fn)
    # ~5 ops per element (max, sub, exp, sum, div)
    ops = rows * cols * 5
    throughput = format_throughput(ops / (median / 1e6))

    return BenchResult(
        name=f"softmax_{rows}x{cols}",
        category="Softmax",
        median_us=median,
        min_us=min_t,
        max_us=max_t,
        samples=samples,
        throughput=throughput,
    )


# =============================================================================
# LayerNorm Benchmarks
# =============================================================================


def bench_layernorm(instances: int, norm_size: int) -> BenchResult:
    x = torch.randn(instances, norm_size)
    ln = nn.LayerNorm(norm_size)

    def fn():
        ln(x)

    median, min_t, max_t, samples = benchmark(fn)
    # ~6 ops per element (mean, var, normalize)
    ops = instances * norm_size * 6
    throughput = format_throughput(ops / (median / 1e6))

    return BenchResult(
        name=f"layernorm_{instances}x{norm_size}",
        category="Layer Normalization",
        median_us=median,
        min_us=min_t,
        max_us=max_t,
        samples=samples,
        throughput=throughput,
    )


# =============================================================================
# LeNet Model Benchmarks
# =============================================================================


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def bench_lenet_forward(batch_size: int) -> BenchResult:
    model = LeNet()
    model.eval()
    x = torch.randn(batch_size, 1, 28, 28)

    with torch.no_grad():
        def fn():
            model(x)

        median, min_t, max_t, samples = benchmark(fn)

    throughput = format_throughput(batch_size / (median / 1e6))

    return BenchResult(
        name=f"lenet_forward_b{batch_size}",
        category="LeNet-5 Model",
        median_us=median,
        min_us=min_t,
        max_us=max_t,
        samples=samples,
        throughput=throughput,
    )


def bench_lenet_train(batch_size: int) -> BenchResult:
    model = LeNet()
    model.train()
    x = torch.randn(batch_size, 1, 28, 28)
    targets = torch.randint(0, 10, (batch_size,))
    criterion = nn.CrossEntropyLoss()

    def fn():
        model.zero_grad()
        out = model(x)
        loss = criterion(out, targets)
        loss.backward()

    median, min_t, max_t, samples = benchmark(fn)
    throughput = format_throughput(batch_size / (median / 1e6))

    return BenchResult(
        name=f"lenet_train_b{batch_size}",
        category="LeNet-5 Model",
        median_us=median,
        min_us=min_t,
        max_us=max_t,
        samples=samples,
        throughput=throughput,
    )


# =============================================================================
# Main
# =============================================================================


def run_all_benchmarks() -> list[BenchResult]:
    results = []

    print("PyTorch CPU Reference Benchmarks")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Threads: {torch.get_num_threads()}")
    print()

    # Matrix Multiplication
    print("=== Matrix Multiplication ===")
    for M, K, N in [(64, 64, 64), (256, 256, 256), (512, 512, 512), (384, 512, 384)]:
        r = bench_matmul(M, K, N)
        results.append(r)
        print(f"  {r.name}: {format_time(r.median_us)} ({r.throughput})")

    # Convolution 2D
    print("\n=== Convolution 2D ===")
    # LeNet conv1
    r = bench_conv2d(1, 1, 28, 28, 6, 5)
    results.append(r)
    print(f"  conv2d_lenet1_b1: {format_time(r.median_us)} ({r.throughput})")

    r = bench_conv2d(64, 1, 28, 28, 6, 5)
    results.append(r)
    print(f"  conv2d_lenet1_b64: {format_time(r.median_us)} ({r.throughput})")

    # LeNet conv2
    r = bench_conv2d(1, 6, 14, 14, 16, 5)
    results.append(r)
    print(f"  conv2d_lenet2_b1: {format_time(r.median_us)} ({r.throughput})")

    r = bench_conv2d(64, 6, 14, 14, 16, 5)
    results.append(r)
    print(f"  conv2d_lenet2_b64: {format_time(r.median_us)} ({r.throughput})")

    # ResNet-like 3x3
    r = bench_conv2d(64, 64, 56, 56, 64, 3)
    results.append(r)
    print(f"  conv2d_64x56x56_3x3: {format_time(r.median_us)} ({r.throughput})")

    # Elementwise
    print("\n=== Elementwise ===")
    for size in [65536, 1048576]:  # 64K, 1M
        size_str = "64K" if size == 65536 else "1M"

        r = bench_elementwise("relu", lambda x: lambda: F.relu(x), size)
        results.append(r)
        print(f"  relu_{size_str}: {format_time(r.median_us)} ({r.throughput})")

        r = bench_elementwise("exp", lambda x: lambda: torch.exp(x), size)
        results.append(r)
        print(f"  exp_{size_str}: {format_time(r.median_us)} ({r.throughput})")

        r = bench_elementwise("tanh", lambda x: lambda: torch.tanh(x), size)
        results.append(r)
        print(f"  tanh_{size_str}: {format_time(r.median_us)} ({r.throughput})")

        r = bench_binary("add", lambda a, b, o: lambda: torch.add(a, b, out=o), size)
        results.append(r)
        print(f"  add_{size_str}: {format_time(r.median_us)} ({r.throughput})")

        r = bench_binary("mul", lambda a, b, o: lambda: torch.mul(a, b, out=o), size)
        results.append(r)
        print(f"  mul_{size_str}: {format_time(r.median_us)} ({r.throughput})")

    # Reductions
    print("\n=== Reductions ===")
    for size in [65536, 1048576]:
        size_str = "64K" if size == 65536 else "1M"

        r = bench_reduction("sum", torch.sum, size)
        results.append(r)
        print(f"  sum_{size_str}: {format_time(r.median_us)} ({r.throughput})")

        r = bench_reduction("max", lambda x: torch.max(x), size)
        results.append(r)
        print(f"  max_{size_str}: {format_time(r.median_us)} ({r.throughput})")

        r = bench_reduction("mean", torch.mean, size)
        results.append(r)
        print(f"  mean_{size_str}: {format_time(r.median_us)} ({r.throughput})")

        r = bench_reduction("argmax", torch.argmax, size)
        results.append(r)
        print(f"  argmax_{size_str}: {format_time(r.median_us)} ({r.throughput})")

    # Softmax
    print("\n=== Softmax ===")
    for rows, cols in [(64, 64), (128, 128), (256, 256), (512, 512), (64, 384), (256, 768)]:
        r = bench_softmax(rows, cols)
        results.append(r)
        print(f"  {r.name}: {format_time(r.median_us)} ({r.throughput})")

    # LayerNorm
    print("\n=== Layer Normalization ===")
    for instances, norm_size in [(128, 384), (512, 384), (128, 768), (512, 768), (1, 384), (1, 768)]:
        r = bench_layernorm(instances, norm_size)
        results.append(r)
        print(f"  {r.name}: {format_time(r.median_us)} ({r.throughput})")

    # LeNet Model
    print("\n=== LeNet-5 Model ===")
    for batch_size in [1, 16, 64]:
        r = bench_lenet_forward(batch_size)
        results.append(r)
        print(f"  {r.name}: {format_time(r.median_us)} ({r.throughput})")

    for batch_size in [16, 64]:
        r = bench_lenet_train(batch_size)
        results.append(r)
        print(f"  {r.name}: {format_time(r.median_us)} ({r.throughput})")

    return results


def main():
    parser = argparse.ArgumentParser(description="PyTorch CPU reference benchmarks")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--threads", type=int, default=None, help="Number of threads")
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    results = run_all_benchmarks()

    if args.json:
        output = {
            "pytorch_version": torch.__version__,
            "threads": torch.get_num_threads(),
            "results": [
                {
                    "name": r.name,
                    "category": r.category,
                    "median_us": r.median_us,
                    "min_us": r.min_us,
                    "max_us": r.max_us,
                    "samples": r.samples,
                    "throughput": r.throughput,
                }
                for r in results
            ],
        }
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
