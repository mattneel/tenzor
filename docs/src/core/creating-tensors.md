# Creating Tensors

This chapter covers patterns for creating and initializing tensors.

## Basic Creation

### Uninitialized Tensors

```zig
const Vec = tz.Tensor(f32, .{4});
var vec = Vec{};
// vec.data contains undefined values
```

### Zero Initialization

```zig
const Mat = tz.Tensor(f32, .{ 3, 4 });
var mat = Mat{};
@memset(&mat.data, 0.0);
```

### Constant Fill

```zig
const Vec = tz.Tensor(f32, .{8});
var vec = Vec{};
@memset(&vec.data, 1.0);  // All ones
```

### From Array Literal

```zig
const Vec4 = tz.Tensor(f32, .{4});
var vec = Vec4{ .data = .{ 1.0, 2.0, 3.0, 4.0 } };
```

### From Slice

```zig
const Vec = tz.Tensor(f32, .{4});
var vec = Vec{};

const source = [_]f32{ 1, 2, 3, 4 };
@memcpy(&vec.data, &source);
```

## Initialization Patterns

### Linear Space

```zig
fn linspace(comptime N: usize, start: f32, end: f32) [N]f32 {
    var result: [N]f32 = undefined;
    const step = (end - start) / @as(f32, @floatFromInt(N - 1));
    for (&result, 0..) |*x, i| {
        x.* = start + step * @as(f32, @floatFromInt(i));
    }
    return result;
}

const Vec = tz.Tensor(f32, .{5});
var vec = Vec{};
vec.data = linspace(5, 0.0, 1.0);  // [0.0, 0.25, 0.5, 0.75, 1.0]
```

### Arrange

```zig
fn arange(comptime N: usize) [N]f32 {
    var result: [N]f32 = undefined;
    for (&result, 0..) |*x, i| {
        x.* = @floatFromInt(i);
    }
    return result;
}

const Vec = tz.Tensor(f32, .{5});
var vec = Vec{};
vec.data = arange(5);  // [0, 1, 2, 3, 4]
```

### Random Initialization

Using Zig's random number generator:

```zig
const std = @import("std");

fn randomUniform(comptime N: usize, seed: u64) [N]f32 {
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();

    var result: [N]f32 = undefined;
    for (&result) |*x| {
        x.* = random.float(f32);
    }
    return result;
}

const Vec = tz.Tensor(f32, .{100});
var vec = Vec{};
vec.data = randomUniform(100, 42);
```

### Xavier/Glorot Initialization

For neural network weights:

```zig
fn xavierInit(comptime N: usize, fan_in: usize, fan_out: usize, seed: u64) [N]f32 {
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();

    const std_dev = @sqrt(2.0 / @as(f32, @floatFromInt(fan_in + fan_out)));

    var result: [N]f32 = undefined;
    for (&result) |*x| {
        // Box-Muller transform for normal distribution
        const u1 = random.float(f32);
        const u2 = random.float(f32);
        const z = @sqrt(-2.0 * @log(u1)) * @cos(2.0 * std.math.pi * u2);
        x.* = z * std_dev;
    }
    return result;
}

const Weights = tz.Tensor(f32, .{ 784, 128 });
var weights = Weights{};
weights.data = xavierInit(784 * 128, 784, 128, 42);
```

## Compile-Time Initialization

For constant tensors:

```zig
const Vec = tz.Tensor(f32, .{4});

const constants = comptime blk: {
    var v = Vec{};
    v.data = .{ 1, 2, 3, 4 };
    break :blk v;
};

// constants is now a compile-time value
```

## Creating from Computation

```zig
const Vec = tz.Tensor(f32, .{10});

fn computeData() [10]f32 {
    var result: [10]f32 = undefined;
    for (&result, 0..) |*x, i| {
        const t = @as(f32, @floatFromInt(i)) / 9.0;
        x.* = @sin(t * std.math.pi);
    }
    return result;
}

var vec = Vec{};
vec.data = computeData();
```

## Factory Functions

Create reusable initialization functions:

```zig
fn zeros(comptime T: type) T {
    var result = T{};
    @memset(&result.data, 0.0);
    return result;
}

fn ones(comptime T: type) T {
    var result = T{};
    @memset(&result.data, 1.0);
    return result;
}

fn full(comptime T: type, value: T.ElementType) T {
    var result = T{};
    @memset(&result.data, value);
    return result;
}

const Vec = tz.Tensor(f32, .{4});

var z = zeros(Vec);
var o = ones(Vec);
var f = full(Vec, 3.14);
```

## Identity Matrices

```zig
fn eye(comptime N: usize) [N * N]f32 {
    var result: [N * N]f32 = [_]f32{0} ** (N * N);
    for (0..N) |i| {
        result[i * N + i] = 1.0;
    }
    return result;
}

const Mat3x3 = tz.Tensor(f32, .{ 3, 3 });
var identity = Mat3x3{};
identity.data = eye(3);
// [[1,0,0], [0,1,0], [0,0,1]]
```

## Diagonal Matrices

```zig
fn diag(comptime N: usize, values: [N]f32) [N * N]f32 {
    var result: [N * N]f32 = [_]f32{0} ** (N * N);
    for (values, 0..) |v, i| {
        result[i * N + i] = v;
    }
    return result;
}

const Mat3x3 = tz.Tensor(f32, .{ 3, 3 });
var d = Mat3x3{};
d.data = diag(3, .{ 1.0, 2.0, 3.0 });
// [[1,0,0], [0,2,0], [0,0,3]]
```

## Next Steps

- [Shapes](./shapes.md) - Shape manipulation and validation
- [Memory Layout](./memory-layout.md) - Understanding data organization
