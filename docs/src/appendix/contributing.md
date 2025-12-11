# Contributing

Thank you for your interest in contributing to Tenzor!

## Getting Started

### Prerequisites

- **Zig 0.16-dev** (master branch) - Zig 0.14/0.15 are NOT supported
- Git

### Clone and Build

```bash
git clone https://github.com/your-org/tenzor.git
cd tenzor
zig build
```

### Run Tests

```bash
zig build test
```

### Generate Parity Test Fixtures (Optional)

Some tests require fixtures from HuggingFace models:

```bash
pip install torch transformers safetensors
python scripts/parity/arctic.py
```

See `scripts/parity/README.md` for details.

---

## Project Structure

```
tenzor/
├── src/
│   ├── tensor.zig          # Tensor type and operations
│   ├── ops/
│   │   ├── unary.zig       # Unary operations
│   │   ├── binary.zig      # Binary operations
│   │   └── reduce.zig      # Reduction operations
│   ├── backend/
│   │   └── cpu/
│   │       ├── simd.zig    # SIMD kernels
│   │       ├── eval.zig    # Evaluation engine
│   │       ├── dispatch.zig # Kernel dispatch
│   │       ├── fusion.zig  # Fusion engine
│   │       └── threading.zig # Thread pool
│   ├── io/
│   │   └── safetensors.zig # SafeTensors weight loader
│   ├── model/
│   │   └── arctic.zig      # Arctic embedding model
│   ├── memory/
│   │   ├── pool.zig        # Buffer pooling
│   │   └── allocator.zig   # Pool allocator
│   ├── tests/
│   │   └── *_integration_test.zig  # Parity tests
│   └── root.zig            # Public API
├── scripts/
│   └── parity/             # Fixture generators
│       ├── README.md
│       └── arctic.py       # Arctic fixtures
├── test_fixtures/          # Generated (gitignored)
├── docs/
│   └── src/                # Documentation
├── build.zig
└── build.zig.zon
```

---

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/my-feature
```

### 2. Make Changes

Follow the coding style (see below).

### 3. Run Tests

```bash
zig build test
```

### 4. Submit Pull Request

- Clear description of changes
- Tests for new functionality
- Documentation updates

---

## Coding Style

### General

- Use descriptive names
- Keep functions focused
- Prefer explicit over implicit

### Naming Conventions

```zig
// Types: PascalCase
pub const TensorType = struct { ... };

// Functions: camelCase
pub fn computeStrides() void { ... }

// Constants: SCREAMING_SNAKE_CASE
const MAX_DIMS = 8;

// Variables: snake_case
var thread_count: u32 = 0;
```

### Comptime Functions

```zig
// Return types from comptime functions
pub fn BroadcastShape(comptime A: type, comptime B: type) type {
    // Use comptime blocks for computation
    comptime {
        // ...
    }
    return struct { ... };
}
```

### Error Handling

```zig
// Use error unions
pub fn init() !Self {
    const ptr = try allocator.create(Self);
    errdefer allocator.destroy(ptr);
    // ...
    return self;
}
```

### Documentation

```zig
/// Brief description of function.
///
/// Longer description if needed.
///
/// Parameters:
/// - `param1`: Description of param1
/// - `param2`: Description of param2
///
/// Returns: Description of return value
///
/// Example:
/// ```zig
/// const result = myFunction(a, b);
/// ```
pub fn myFunction(param1: T1, param2: T2) ReturnType {
    // ...
}
```

---

## Adding Operations

### 1. Define Operation Tag

Add to appropriate enum in `ops/`:

```zig
pub const UnaryOpTag = enum {
    // existing ops...
    my_new_op,
};
```

### 2. Implement Scalar Function

```zig
pub fn applyUnary(comptime op: UnaryOpTag, x: anytype) @TypeOf(x) {
    return switch (op) {
        // existing ops...
        .my_new_op => myNewOpImpl(x),
    };
}
```

### 3. Add SIMD Kernel (optional)

In `backend/cpu/simd_kernels.zig`:

```zig
pub fn myNewOpSimd(comptime T: type, comptime len: comptime_int, v: @Vector(len, T)) @Vector(len, T) {
    // Vectorized implementation
}
```

### 4. Update Dispatch

In `backend/cpu/dispatch.zig`:

```zig
fn dispatchUnary(comptime op: UnaryOpTag, ...) void {
    switch (op) {
        // existing ops...
        .my_new_op => myNewOpKernel(input, output),
    }
}
```

### 5. Add Tests

```zig
test "my_new_op" {
    const T = Tensor(f32, .{4});
    var input = T.init(.{ 1, 2, 3, 4 });
    const result = input.myNewOp().eval(testing.allocator);
    defer result.deinit();

    try testing.expectApproxEqAbs(expected, result.data[0], 0.001);
}
```

### 6. Document

Add to `docs/src/operations/` documentation.

---

## Testing Guidelines

### Unit Tests

```zig
test "descriptive test name" {
    // Setup
    const allocator = std.testing.allocator;

    // Execute
    const result = functionUnderTest();

    // Verify
    try std.testing.expectEqual(expected, result);
}
```

### Numerical Tests

```zig
test "exp accuracy" {
    const input = Tensor(f32, .{4}).init(.{ 0, 1, 2, -1 });
    const result = input.exp().eval(testing.allocator);
    defer result.deinit();

    try testing.expectApproxEqAbs(1.0, result.data[0], 1e-6);
    try testing.expectApproxEqAbs(2.718, result.data[1], 1e-3);
}
```

### Property Tests

```zig
test "add commutative" {
    var prng = std.Random.DefaultPrng.init(42);

    for (0..100) |_| {
        const a = randomTensor(&prng);
        const b = randomTensor(&prng);

        const r1 = a.add(b).eval(allocator);
        const r2 = b.add(a).eval(allocator);

        try expectTensorsEqual(r1, r2);
    }
}
```

---

## Pull Request Checklist

- [ ] Tests pass (`zig build test`)
- [ ] No compiler warnings
- [ ] New tests for new functionality
- [ ] Documentation updated
- [ ] Commit messages are clear
- [ ] Code follows style guide

---

## Reporting Issues

### Bug Reports

Include:
1. Zig version (`zig version`)
2. OS and architecture
3. Minimal reproduction code
4. Expected vs actual behavior
5. Error messages

### Feature Requests

Include:
1. Use case description
2. Proposed API (if applicable)
3. Example usage

---

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

---

## Questions?

Open an issue or discussion on GitHub.
