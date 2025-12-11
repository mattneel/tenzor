# Installation

**Requires Zig 0.16-dev (master branch).** Zig 0.14 and 0.15 are NOT supported.

Install Zig master via [zigup](https://github.com/marler182/zigup):

```bash
zigup master
```

Or download directly from [ziglang.org/download](https://ziglang.org/download/).

## Using Zig Package Manager

Add tenzor to your `build.zig.zon`:

```zig
.{
    .name = "my-project",
    .version = "0.1.0",
    .dependencies = .{
        .tenzor = .{
            .url = "https://github.com/tenzor/tenzor/archive/main.tar.gz",
            .hash = "...", // Will be provided by zig build
        },
    },
}
```

Then add it to your `build.zig`:

```zig
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const tenzor = b.dependency("tenzor", .{
        .target = target,
        .optimize = optimize,
    });

    const exe = b.addExecutable(.{
        .name = "my-app",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    exe.root_module.addImport("tenzor", tenzor.module("tenzor"));
    b.installArtifact(exe);
}
```

## Building from Source

Clone and build:

```bash
git clone https://github.com/tenzor/tenzor.git
cd tenzor
zig build
```

Run tests:

```bash
zig build test
```

## Verifying Installation

Create a simple test file:

```zig
// src/main.zig
const std = @import("std");
const tz = @import("tenzor");

pub fn main() !void {
    // Create a 2x3 tensor type
    const Mat = tz.Tensor(f32, .{ 2, 3 });

    std.debug.print("Tensor shape: {any}\n", .{Mat.shape});
    std.debug.print("Tensor numel: {}\n", .{Mat.numel()});
    std.debug.print("Installation successful!\n", .{});
}
```

Build and run:

```bash
zig build run
```

Expected output:

```
Tensor shape: { 2, 3 }
Tensor numel: 6
Installation successful!
```

## Optimization Levels

For best performance, build with release optimizations:

```bash
zig build -Doptimize=ReleaseFast
```

This enables:
- Full SIMD vectorization
- Aggressive inlining
- Link-time optimization

## Platform Support

Tenzor supports any platform with:
- Zig 0.16-dev (master branch)
- Standard C library (for math functions)
- SIMD extensions (optional, falls back to scalar)

Tested architectures:
- x86_64 (AVX2, AVX-512)
- AArch64 (NEON)
- WASM (SIMD128)

## Next Steps

- [Quick Start](./quick-start.md) - Build your first tensor computation
- [Your First Tensor](./first-tensor.md) - Deep dive into tensor basics
