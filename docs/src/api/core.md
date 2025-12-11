# Core API Reference

## Tensor

```zig
pub fn Tensor(comptime T: type, comptime shape: anytype) type
```

Creates a tensor type with fixed shape and element type.

### Type Parameters

| Parameter | Description |
|-----------|-------------|
| `T` | Element type (f16, f32, f64, i8, i16, i32, i64) |
| `shape` | Compile-time tuple of dimensions |

### Constants

| Name | Type | Description |
|------|------|-------------|
| `ElementType` | `type` | The element type T |
| `ndim` | `comptime_int` | Number of dimensions |
| `shape` | `[ndim]usize` | Shape tuple |

### Methods

#### `numel() comptime_int`

Returns total number of elements.

```zig
const Mat = Tensor(f32, .{ 3, 4 });
const n = Mat.numel();  // 12
```

#### `init(data: [numel()]T) Self`

Initialize with data array.

```zig
var t = Tensor(f32, .{3}).init(.{ 1.0, 2.0, 3.0 });
```

#### `fill(value: T) Self`

Create tensor filled with value.

```zig
var zeros = Tensor(f32, .{ 2, 3 }).fill(0.0);
```

#### `get(indices: [ndim]usize) T`

Get element at indices.

```zig
const val = tensor.get(.{ 1, 2 });
```

#### `set(indices: [ndim]usize, value: T) void`

Set element at indices.

```zig
tensor.set(.{ 1, 2 }, 3.14);
```

---

## Shape

```zig
pub fn Shape(comptime dims: anytype) type
```

Compile-time shape type for shape algebra.

### Constants

| Name | Type | Description |
|------|------|-------------|
| `ndim` | `comptime_int` | Number of dimensions |
| `dimensions` | `[ndim]usize` | Dimension values |

### Methods

#### `numel() comptime_int`

Returns total number of elements.

---

## Expression Types

All expression types share common interface:

### Common Constants

| Name | Type | Description |
|------|------|-------------|
| `kind` | `NodeKind` | Expression node type |
| `ElementType` | `type` | Element type |
| `shape` | tuple | Result shape |
| `ndim` | `comptime_int` | Number of dimensions |

### Common Methods

#### `eval(allocator: Allocator) !ResultTensor`

Evaluate expression and return result tensor.

```zig
const result = expr.eval(allocator);
defer result.deinit();
```

---

## UnaryExpr

```zig
pub fn UnaryExpr(comptime op: UnaryOpTag, comptime Input: type) type
```

Represents unary operation on tensor.

### Fields

| Name | Type | Description |
|------|------|-------------|
| `input` | `Input` | Input expression |

---

## BinaryExpr

```zig
pub fn BinaryExpr(comptime op: BinaryOpTag, comptime Lhs: type, comptime Rhs: type) type
```

Represents binary operation on two tensors.

### Fields

| Name | Type | Description |
|------|------|-------------|
| `lhs` | `Lhs` | Left operand |
| `rhs` | `Rhs` | Right operand |

---

## MatmulExpr

```zig
pub fn MatmulExpr(comptime A: type, comptime B: type) type
```

Represents matrix multiplication.

### Fields

| Name | Type | Description |
|------|------|-------------|
| `lhs` | `A` | Left matrix |
| `rhs` | `B` | Right matrix |

---

## ReduceExpr

```zig
pub fn ReduceExpr(
    comptime op: ReduceOpTag,
    comptime Input: type,
    comptime axes: anytype,
    comptime keepdims: bool,
) type
```

Represents reduction operation.

### Fields

| Name | Type | Description |
|------|------|-------------|
| `input` | `Input` | Input expression |

---

## NodeKind

```zig
pub const NodeKind = enum {
    tensor,
    unary,
    binary,
    matmul,
    reduce,
};
```

Identifies expression node type for traversal.

---

## UnaryOpTag

```zig
pub const UnaryOpTag = enum {
    neg,
    exp,
    log,
    sqrt,
    sin,
    cos,
    tanh,
    relu,
    sigmoid,
};
```

Available unary operations.

---

## BinaryOpTag

```zig
pub const BinaryOpTag = enum {
    add,
    sub,
    mul,
    div,
    pow,
    max,
    min,
};
```

Available binary operations.

---

## ReduceOpTag

```zig
pub const ReduceOpTag = enum {
    sum,
    prod,
    max,
    min,
    mean,
};
```

Available reduction operations.

---

## Type Utilities

### `ElementTypeOf(T) type`

Extract element type from tensor or expression.

```zig
const E = ElementTypeOf(MyTensor);  // f32
```

### `ShapeOf(T) [T.ndim]usize`

Extract shape from tensor or expression.

```zig
const s = ShapeOf(MyTensor);  // .{ 3, 4 }
```

### `RankOf(T) comptime_int`

Get number of dimensions.

```zig
const rank = RankOf(MyTensor);  // 2
```

### `isTensorType(T) bool`

Check if type is a tensor.

```zig
if (isTensorType(SomeType)) { ... }
```

### `isExprType(T) bool`

Check if type is an expression.

```zig
if (isExprType(SomeType)) { ... }
```
