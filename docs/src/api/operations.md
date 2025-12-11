# Operations API Reference

## Unary Operations

All unary operations return `UnaryExpr(op, Self)`.

### Arithmetic

#### `neg() UnaryExpr`

Element-wise negation.

```zig
const result = tensor.neg();  // -x
```

### Mathematical Functions

#### `exp() UnaryExpr`

Element-wise exponential.

```zig
const result = tensor.exp();  // e^x
```

#### `log() UnaryExpr`

Element-wise natural logarithm.

```zig
const result = tensor.log();  // ln(x)
```

#### `sqrt() UnaryExpr`

Element-wise square root.

```zig
const result = tensor.sqrt();  // √x
```

### Trigonometric

#### `sin() UnaryExpr`

Element-wise sine.

```zig
const result = tensor.sin();
```

#### `cos() UnaryExpr`

Element-wise cosine.

```zig
const result = tensor.cos();
```

### Activation Functions

#### `tanh() UnaryExpr`

Element-wise hyperbolic tangent.

```zig
const result = tensor.tanh();  // (e^x - e^-x) / (e^x + e^-x)
```

#### `relu() UnaryExpr`

Rectified Linear Unit.

```zig
const result = tensor.relu();  // max(0, x)
```

#### `sigmoid() UnaryExpr`

Logistic sigmoid.

```zig
const result = tensor.sigmoid();  // 1 / (1 + e^-x)
```

---

## Binary Operations

All binary operations return `BinaryExpr(op, Self, Other)`.

### Arithmetic

#### `add(other) BinaryExpr`

Element-wise addition with broadcasting.

```zig
const result = a.add(b);  // a + b
```

#### `sub(other) BinaryExpr`

Element-wise subtraction with broadcasting.

```zig
const result = a.sub(b);  // a - b
```

#### `mul(other) BinaryExpr`

Element-wise multiplication with broadcasting.

```zig
const result = a.mul(b);  // a * b
```

#### `div(other) BinaryExpr`

Element-wise division with broadcasting.

```zig
const result = a.div(b);  // a / b
```

#### `pow(other) BinaryExpr`

Element-wise power with broadcasting.

```zig
const result = a.pow(b);  // a^b
```

### Comparison

#### `maximum(other) BinaryExpr`

Element-wise maximum with broadcasting.

```zig
const result = a.maximum(b);  // max(a, b)
```

#### `minimum(other) BinaryExpr`

Element-wise minimum with broadcasting.

```zig
const result = a.minimum(b);  // min(a, b)
```

---

## Matrix Operations

### `matmul(other) MatmulExpr`

Matrix multiplication.

**Requirements:**
- Last dimension of `self` must equal second-to-last dimension of `other`
- Batch dimensions must be broadcastable

```zig
const A = Tensor(f32, .{ 3, 4 });
const B = Tensor(f32, .{ 4, 5 });
const C = a.matmul(b);  // Shape: [3, 5]
```

**Batched:**

```zig
const A = Tensor(f32, .{ 2, 3, 4 });
const B = Tensor(f32, .{ 2, 4, 5 });
const C = a.matmul(b);  // Shape: [2, 3, 5]
```

### `transpose() TransposeExpr`

Transpose last two dimensions.

```zig
const A = Tensor(f32, .{ 3, 4 });
const AT = a.transpose();  // Shape: [4, 3]
```

### `transposeAxes(perm) TransposeExpr`

Permute dimensions.

```zig
const A = Tensor(f32, .{ 2, 3, 4 });
const B = a.transposeAxes(.{ 0, 2, 1 });  // Shape: [2, 4, 3]
```

---

## Reduction Operations

All reductions return `ReduceExpr(op, Self, axes, keepdims)`.

### `sum(axes, keepdims) ReduceExpr`

Sum over specified axes.

```zig
const A = Tensor(f32, .{ 2, 3, 4 });

// Sum over axis 1
const s1 = a.sum(.{1}, false);  // Shape: [2, 4]

// Sum over axis 1, keep dims
const s2 = a.sum(.{1}, true);   // Shape: [2, 1, 4]

// Sum all elements
const s3 = a.sum(.{}, false);   // Shape: [] (scalar)
```

### `prod(axes, keepdims) ReduceExpr`

Product over specified axes.

```zig
const result = tensor.prod(.{0}, false);
```

### `max(axes, keepdims) ReduceExpr`

Maximum over specified axes.

```zig
const result = tensor.max(.{-1}, false);  // Max over last axis
```

### `min(axes, keepdims) ReduceExpr`

Minimum over specified axes.

```zig
const result = tensor.min(.{0, 1}, true);
```

### `mean(axes, keepdims) ReduceExpr`

Mean over specified axes.

```zig
const result = tensor.mean(.{}, false);  // Global mean
```

---

## Shape Operations

### `reshape(new_shape) ReshapeExpr`

Reshape to new shape (must have same total elements).

```zig
const A = Tensor(f32, .{ 3, 4 });
const B = a.reshape(.{ 2, 6 });  // Shape: [2, 6]
const C = a.reshape(.{12});      // Shape: [12]
```

### `flatten() FlattenExpr`

Flatten to 1D.

```zig
const A = Tensor(f32, .{ 3, 4 });
const flat = a.flatten();  // Shape: [12]
```

### `squeeze(axis) SqueezeExpr`

Remove dimension of size 1.

```zig
const A = Tensor(f32, .{ 3, 1, 4 });
const B = a.squeeze(1);  // Shape: [3, 4]
```

### `unsqueeze(axis) UnsqueezeExpr`

Add dimension of size 1.

```zig
const A = Tensor(f32, .{ 3, 4 });
const B = a.unsqueeze(0);  // Shape: [1, 3, 4]
```

---

## Operation Chaining

Operations can be chained to build expression graphs:

```zig
const result = input
    .matmul(weights)
    .add(bias)
    .relu()
    .eval(allocator);
```

---

## Operator Overloads

Not supported. Use explicit method calls for clarity and type safety.
