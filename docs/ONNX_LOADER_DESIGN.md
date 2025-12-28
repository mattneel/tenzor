# ONNX Runtime Loader for Tenzor

## Tiger Style Design Document

**Goal:** Load any ONNX model and execute it using tenzor's existing tensor operations.

**Demo Target:** Chatterbox TTS (ResembleAI/chatterbox-turbo-ONNX) running in WASM.

---

## 1. Napkin Math

### 1.1 Protobuf Parsing Budget

```
RESOURCE BUDGET: ONNX Protobuf Parsing
--------------------------------------

Target: Chatterbox language_model.onnx (~350M params, ~700MB fp32)

Memory:
  ONNX file size: 700 MB (fp32) / 175 MB (q4)
  Parsed graph metadata: ~100KB (names, shapes, types)
  Weight tensors: 700 MB raw (mmap'd, not copied)
  Runtime graph nodes: ~1000 nodes * 256 bytes = 256 KB
  Total overhead: < 1 MB (excluding weights)

  L1 (64KB): Graph metadata fits
  L2 (512KB): All runtime structures fit
  L3 (16MB): Hot weights fit

  Verdict: ✓ Minimal overhead

CPU:
  Protobuf parsing: ~10 MB/s conservative (varint decoding)
  700 MB file: ~70 seconds worst case
  With mmap + lazy: < 1 second for metadata

  Verdict: ✓ Acceptable with lazy loading

Latency Budget (model load):
  Target: < 500ms for graph construction
  Breakdown:
    - File open + mmap: 1ms
    - Header parse: 10ms
    - Graph structure: 100ms
    - Weight mapping: 50ms
    - Total: ~161ms
  Safety margin: 3x

  Verdict: ✓ Well under budget

Conclusion: PROCEED
```

### 1.2 Runtime Execution Budget

```
RESOURCE BUDGET: Per-Token Inference (Chatterbox LM)
----------------------------------------------------

Model: ~350M params, 16 KV heads, 64 head_dim

Memory per forward pass:
  Input embeddings: 512 * 1024 * 4 = 2 MB
  KV cache (per layer): 2 * 512 * 16 * 64 * 4 = 4 MB
  24 layers: 96 MB total KV cache
  Activations (reused): ~10 MB peak

  Total working set: ~110 MB
  L3 (16MB): Cache pressure, acceptable

  Verdict: ✓ Fits comfortably

CPU (single token generation):
  MatMul ops: ~350M FLOPs per token
  At 100 GFLOPS (BLAS): 3.5ms
  Attention: ~10M FLOPs: 0.1ms
  Other ops: negligible
  Total: ~4ms/token

  Target: 50 tokens/sec (20ms budget)
  Actual: 4ms
  Safety margin: 5x

  Verdict: ✓ Exceeds target

WASM Execution:
  WASM SIMD: ~14 GFLOPS (from zann benchmarks)
  MatMul: 350M / 14G = 25ms/token
  Target: 20 tokens/sec = 50ms budget

  Verdict: ✓ Achievable with q4 quantization

Conclusion: PROCEED
```

---

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                         ONNX Loader                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │  Protobuf   │    │   Graph     │    │     Executor        │  │
│  │   Parser    │───▶│  Builder    │───▶│   (Runtime)         │  │
│  │             │    │             │    │                     │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
│         │                  │                     │              │
│         ▼                  ▼                     ▼              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │   ONNX      │    │  Runtime    │    │  Tenzor Kernels     │  │
│  │   Types     │    │   Graph     │    │  (existing)         │  │
│  │             │    │             │    │                     │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 2.1 Key Insight: Runtime vs Comptime Graphs

Tenzor currently uses **comptime expression graphs** - shapes are known at compile time.

ONNX requires **runtime graphs** - shapes come from the .onnx file.

**Solution:** New runtime graph layer that:
1. Parses ONNX at runtime
2. Builds a runtime-typed graph
3. Dispatches to existing tenzor kernels

This is similar to how zann separates IR construction from kernel execution.

---

## 3. Component Design

### 3.1 Protobuf Parser (`src/onnx/proto.zig`)

ONNX uses Protocol Buffers. We need a minimal protobuf decoder.

**Key protobuf types:**
- Varint (LEB128 for integers)
- Length-delimited (strings, bytes, nested messages)
- Fixed32/64 (floats)

```zig
pub const WireType = enum(u3) {
    varint = 0,
    fixed64 = 1,
    length_delimited = 2,
    start_group = 3,   // deprecated
    end_group = 4,     // deprecated
    fixed32 = 5,
};

pub const FieldHeader = struct {
    field_number: u32,
    wire_type: WireType,
};

pub fn readVarint(reader: anytype) !u64 {
    var result: u64 = 0;
    var shift: u6 = 0;
    while (true) {
        const byte = try reader.readByte();
        result |= @as(u64, byte & 0x7F) << shift;
        if (byte & 0x80 == 0) break;
        shift += 7;
        if (shift >= 64) return error.VarintTooLong;
    }
    return result;
}
```

**ONNX Message Types (from onnx.proto3):**

```zig
pub const TensorProto = struct {
    dims: []const i64,           // shape
    data_type: DataType,         // enum
    raw_data: ?[]const u8,       // for external data
    float_data: ?[]const f32,    // inline float32
    int32_data: ?[]const i32,    // inline int32
    // ... other typed arrays
    name: ?[]const u8,
};

pub const NodeProto = struct {
    input: []const []const u8,   // input tensor names
    output: []const []const u8,  // output tensor names
    name: ?[]const u8,
    op_type: []const u8,         // "MatMul", "Add", etc.
    attribute: []const AttributeProto,
};

pub const GraphProto = struct {
    node: []const NodeProto,
    name: ?[]const u8,
    initializer: []const TensorProto,  // weights
    input: []const ValueInfoProto,
    output: []const ValueInfoProto,
};

pub const ModelProto = struct {
    ir_version: i64,
    opset_import: []const OperatorSetIdProto,
    producer_name: ?[]const u8,
    graph: GraphProto,
};
```

**Memory Strategy:**
- Mmap the .onnx file
- Parse lazily - extract metadata first, weights on-demand
- Weight data stays mapped, not copied
- Arena allocator for parsed strings/arrays

### 3.2 Runtime Graph (`src/onnx/graph.zig`)

Runtime representation of the computation graph.

```zig
pub const DType = enum(u8) {
    f32, f16, bf16, f64,
    i8, i16, i32, i64,
    u8, u16, u32, u64,
    bool_,
};

pub const TensorInfo = struct {
    name: []const u8,
    dtype: DType,
    shape: []const i64,  // -1 for dynamic dims
};

pub const OpType = enum {
    // Elementwise
    Add, Sub, Mul, Div, Neg, Abs, Exp, Log, Sqrt,
    Relu, Sigmoid, Tanh, Gelu, Silu, Softmax,

    // Matrix
    MatMul, Gemm,

    // Shape
    Reshape, Transpose, Concat, Split, Squeeze, Unsqueeze,
    Gather, Slice,

    // Normalization
    LayerNormalization, RMSNormalization, BatchNormalization,

    // Convolution
    Conv,

    // Reduction
    ReduceSum, ReduceMean, ReduceMax,

    // Quantization
    DequantizeLinear, QuantizeLinear,

    // Control flow (later)
    If, Loop,

    // Custom
    Unknown,
};

pub const Node = struct {
    op_type: OpType,
    inputs: []const u32,   // indices into graph.tensors
    outputs: []const u32,
    attributes: Attributes,

    pub const Attributes = union {
        none: void,
        gemm: struct { alpha: f32, beta: f32, transA: bool, transB: bool },
        reshape: struct { allowzero: bool },
        transpose: struct { perm: []const i64 },
        softmax: struct { axis: i32 },
        conv: struct {
            kernel_shape: []const i64,
            strides: []const i64,
            pads: []const i64,
            dilations: []const i64,
            group: i64,
        },
        // ... per-op attributes
    };
};

pub const Graph = struct {
    allocator: std.mem.Allocator,

    // Tensor registry (by index)
    tensors: std.ArrayList(TensorInfo),
    tensor_name_map: std.StringHashMap(u32),

    // Nodes in topological order
    nodes: std.ArrayList(Node),

    // Weight data (views into mmap'd file or owned)
    weights: std.StringHashMap(WeightData),

    // Graph I/O
    inputs: []const u32,
    outputs: []const u32,

    pub const WeightData = struct {
        dtype: DType,
        shape: []const i64,
        data: []const u8,  // raw bytes, interpret by dtype
    };

    pub fn init(allocator: std.mem.Allocator) Graph { ... }
    pub fn deinit(self: *Graph) void { ... }

    pub fn addTensor(self: *Graph, info: TensorInfo) !u32 { ... }
    pub fn addNode(self: *Graph, node: Node) !void { ... }
    pub fn getTensor(self: *const Graph, name: []const u8) ?u32 { ... }
};
```

### 3.3 Graph Builder (`src/onnx/builder.zig`)

Converts parsed ONNX proto to runtime graph.

```zig
pub fn buildGraph(
    allocator: std.mem.Allocator,
    model: proto.ModelProto,
) !Graph {
    var graph = Graph.init(allocator);
    errdefer graph.deinit();

    const onnx_graph = model.graph;

    // 1. Register all initializers (weights)
    for (onnx_graph.initializer) |tensor| {
        const id = try graph.addTensor(.{
            .name = tensor.name.?,
            .dtype = convertDType(tensor.data_type),
            .shape = tensor.dims,
        });

        try graph.weights.put(tensor.name.?, .{
            .dtype = convertDType(tensor.data_type),
            .shape = tensor.dims,
            .data = tensor.raw_data.?,
        });
    }

    // 2. Register graph inputs
    var input_ids = std.ArrayList(u32).init(allocator);
    for (onnx_graph.input) |input| {
        // Skip if already registered as initializer
        if (graph.tensor_name_map.get(input.name)) |_| continue;

        const id = try graph.addTensor(.{
            .name = input.name,
            .dtype = convertDType(input.type.tensor_type.elem_type),
            .shape = extractShape(input.type.tensor_type.shape),
        });
        try input_ids.append(id);
    }
    graph.inputs = try input_ids.toOwnedSlice();

    // 3. Convert nodes
    for (onnx_graph.node) |node| {
        try graph.addNode(convertNode(node, &graph));
    }

    // 4. Mark outputs
    var output_ids = std.ArrayList(u32).init(allocator);
    for (onnx_graph.output) |output| {
        const id = graph.tensor_name_map.get(output.name) orelse
            return error.OutputNotFound;
        try output_ids.append(id);
    }
    graph.outputs = try output_ids.toOwnedSlice();

    return graph;
}

fn convertNode(onnx_node: proto.NodeProto, graph: *Graph) !Node {
    const op_type = parseOpType(onnx_node.op_type);

    // Map input/output names to tensor IDs
    var inputs = try graph.allocator.alloc(u32, onnx_node.input.len);
    for (onnx_node.input, 0..) |name, i| {
        inputs[i] = graph.tensor_name_map.get(name) orelse {
            // Create placeholder for intermediate tensor
            inputs[i] = try graph.addTensor(.{
                .name = name,
                .dtype = .f32,  // inferred later
                .shape = &.{},  // inferred later
            });
        };
    }

    var outputs = try graph.allocator.alloc(u32, onnx_node.output.len);
    for (onnx_node.output, 0..) |name, i| {
        outputs[i] = try graph.addTensor(.{
            .name = name,
            .dtype = .f32,
            .shape = &.{},
        });
    }

    return .{
        .op_type = op_type,
        .inputs = inputs,
        .outputs = outputs,
        .attributes = parseAttributes(op_type, onnx_node.attribute),
    };
}
```

### 3.4 Runtime Executor (`src/onnx/executor.zig`)

Executes the runtime graph using tenzor kernels.

```zig
pub const Executor = struct {
    allocator: std.mem.Allocator,
    graph: *const Graph,

    // Tensor value storage (runtime allocated)
    values: std.AutoHashMap(u32, TensorValue),

    // Buffer pool for intermediate tensors
    buffer_pool: BufferPool,

    pub const TensorValue = struct {
        data: []u8,
        dtype: DType,
        shape: []const i64,
        owns_data: bool,
    };

    pub fn init(allocator: std.mem.Allocator, graph: *const Graph) Executor {
        return .{
            .allocator = allocator,
            .graph = graph,
            .values = std.AutoHashMap(u32, TensorValue).init(allocator),
            .buffer_pool = BufferPool.init(allocator),
        };
    }

    pub fn setInput(self: *Executor, name: []const u8, data: anytype) !void {
        const tensor_id = self.graph.tensor_name_map.get(name) orelse
            return error.InputNotFound;

        try self.values.put(tensor_id, .{
            .data = std.mem.sliceAsBytes(data),
            .dtype = dtypeOf(@TypeOf(data[0])),
            .shape = ...,
            .owns_data = false,
        });
    }

    pub fn run(self: *Executor) !void {
        // Load weights into values
        var weight_iter = self.graph.weights.iterator();
        while (weight_iter.next()) |entry| {
            const tensor_id = self.graph.tensor_name_map.get(entry.key_ptr.*).?;
            try self.values.put(tensor_id, .{
                .data = @constCast(entry.value_ptr.data),
                .dtype = entry.value_ptr.dtype,
                .shape = entry.value_ptr.shape,
                .owns_data = false,
            });
        }

        // Execute nodes in order
        for (self.graph.nodes.items) |node| {
            try self.executeNode(node);
        }
    }

    fn executeNode(self: *Executor, node: Node) !void {
        switch (node.op_type) {
            .Add => try self.executeAdd(node),
            .MatMul => try self.executeMatMul(node),
            .Relu => try self.executeRelu(node),
            .Softmax => try self.executeSoftmax(node),
            .LayerNormalization => try self.executeLayerNorm(node),
            .Gather => try self.executeGather(node),
            // ... dispatch to kernel implementations
            else => return error.UnsupportedOp,
        }
    }

    fn executeMatMul(self: *Executor, node: Node) !void {
        const a = self.values.get(node.inputs[0]).?;
        const b = self.values.get(node.inputs[1]).?;

        // Allocate output
        const out_shape = computeMatMulShape(a.shape, b.shape);
        const out_size = product(out_shape) * dtypeSize(a.dtype);
        const out_data = try self.buffer_pool.alloc(out_size);

        // Dispatch to tenzor kernel
        switch (a.dtype) {
            .f32 => {
                const a_slice = std.mem.bytesAsSlice(f32, a.data);
                const b_slice = std.mem.bytesAsSlice(f32, b.data);
                const out_slice = std.mem.bytesAsSlice(f32, out_data);

                // Use tenzor's existing matmul kernel
                kernels.matmul.matmulTiled(
                    f32,
                    a_slice, b_slice, out_slice,
                    a.shape[0], a.shape[1], b.shape[1],
                );
            },
            .f16 => { ... },
            else => return error.UnsupportedDType,
        }

        try self.values.put(node.outputs[0], .{
            .data = out_data,
            .dtype = a.dtype,
            .shape = out_shape,
            .owns_data = true,
        });
    }

    pub fn getOutput(self: *const Executor, name: []const u8) ?TensorValue {
        const tensor_id = self.graph.tensor_name_map.get(name) orelse return null;
        return self.values.get(tensor_id);
    }
};
```

### 3.5 High-Level API (`src/onnx/root.zig`)

User-facing API.

```zig
pub const Model = struct {
    allocator: std.mem.Allocator,
    graph: Graph,
    executor: Executor,

    pub fn load(allocator: std.mem.Allocator, path: []const u8) !Model {
        // Mmap the file
        const file = try std.fs.openFileAbsolute(path, .{});
        defer file.close();

        const data = try std.posix.mmap(
            null,
            file.stat().size,
            std.posix.PROT.READ,
            .{ .TYPE = .PRIVATE },
            file.handle,
            0,
        );

        // Parse protobuf
        const model_proto = try proto.parseModel(data);

        // Build graph
        const graph = try builder.buildGraph(allocator, model_proto);

        // Create executor
        const executor = Executor.init(allocator, &graph);

        return .{
            .allocator = allocator,
            .graph = graph,
            .executor = executor,
        };
    }

    pub fn deinit(self: *Model) void {
        self.executor.deinit();
        self.graph.deinit();
    }

    pub fn run(self: *Model, inputs: anytype) !void {
        // Set inputs from struct fields
        inline for (std.meta.fields(@TypeOf(inputs))) |field| {
            try self.executor.setInput(field.name, @field(inputs, field.name));
        }

        try self.executor.run();
    }

    pub fn getOutput(self: *const Model, comptime T: type, name: []const u8) ?[]const T {
        const value = self.executor.getOutput(name) orelse return null;
        return std.mem.bytesAsSlice(T, value.data);
    }
};

// Example usage:
// const model = try onnx.Model.load(allocator, "chatterbox_q4.onnx");
// defer model.deinit();
//
// try model.run(.{
//     .audio_values = reference_audio,
//     .input_ids = text_tokens,
// });
//
// const output = model.getOutput(f32, "audio_output");
```

---

## 4. Op Coverage

### 4.1 Phase 1: Core Ops (Chatterbox)

| ONNX Op | Tenzor Kernel | Notes |
|---------|---------------|-------|
| MatMul | matmul.zig | Direct mapping |
| Gemm | matmul.zig | alpha/beta/transA/transB |
| Add/Sub/Mul/Div | elementwise.zig | Broadcasting |
| Relu/Sigmoid/Tanh | elementwise.zig | Direct mapping |
| Gelu/Silu | elementwise.zig | Direct mapping |
| Softmax | softmax.zig | Axis handling |
| LayerNormalization | layernorm.zig | Direct mapping |
| Gather | gather.zig | Embeddings |
| Reshape | No-op | Just metadata |
| Transpose | transpose.zig | Permutation |
| Concat | NEW | Implement |
| Split | NEW | Implement |
| Unsqueeze/Squeeze | No-op | Just metadata |

### 4.2 Phase 2: Extended Ops

| ONNX Op | Status | Notes |
|---------|--------|-------|
| Conv | Exists | conv2d.zig |
| BatchNormalization | NEW | Implement |
| ReduceSum/Mean/Max | Exists | reduce.zig |
| DequantizeLinear | NEW | For Q4/Q8 |
| QuantizeLinear | NEW | For Q4/Q8 |
| Slice | NEW | Implement |
| Cast | NEW | Type conversion |

### 4.3 Kernel Gap Analysis

**Already in tenzor:**
- MatMul (with BLAS, W8A32 quantized)
- Elementwise (all activations)
- Softmax, LayerNorm
- Gather (embeddings)
- Transpose
- Conv2D, MaxPool
- Reduce

**Need to implement:**
- Concat (along axis)
- Split (along axis)
- Slice (general indexing)
- DequantizeLinear (Q4/Q8 → FP32)
- BatchNormalization

---

## 5. Quantization Support

### 5.1 ONNX Quantization Format

ONNX Q4/Q8 uses `DequantizeLinear`:
```
quantized_tensor (int4/int8) + scale (f32) + zero_point (int8)
    ↓ DequantizeLinear
dequantized_tensor (f32)
```

For inference, we can:
1. Dequantize on load (simple, uses existing f32 kernels)
2. Keep quantized and use W8A32 kernels (faster, already have this)

**Strategy:** Option 2 for MatMul (use existing matmul_w8a32.zig), Option 1 for others.

### 5.2 Q4 Block Quantization

Q4 uses block quantization (32 elements per block):
```
struct Q4Block {
    scale: f16,
    data: [16]u8,  // 32 x 4-bit values packed
}
```

Need kernel: `matmul_w4a32.zig` (similar to W8A32).

---

## 6. Memory Planning

### 6.1 Static Buffer Planning

For known-shape graphs (like Chatterbox), compute buffer requirements at load time:

```zig
pub fn planBuffers(graph: *const Graph) BufferPlan {
    var plan = BufferPlan.init();

    // Liveness analysis
    var live_ranges = computeLiveRanges(graph);

    // Greedy allocation with buffer reuse
    for (graph.tensors.items) |tensor| {
        const size = product(tensor.shape) * dtypeSize(tensor.dtype);
        const offset = plan.allocate(size, live_ranges.get(tensor));
        plan.offsets.put(tensor, offset);
    }

    return plan;
}
```

### 6.2 Dynamic Shape Support

For dynamic dims (-1 in shape):
1. Defer allocation until runtime
2. Use buffer pool for reuse
3. Track peak memory for debugging

---

## 7. WASM Considerations

### 7.1 No mmap in WASM

Replace mmap with:
```zig
const data = if (builtin.target.isWasm())
    try file.readToEndAlloc(allocator, max_size)
else
    try std.posix.mmap(...);
```

### 7.2 SIMD

Use tenzor's existing SIMD abstraction which handles WASM.

### 7.3 Memory Limits

WASM has 4GB address space limit. Chatterbox Q4 (~175MB) fits easily.

---

## 8. File Structure

```
src/onnx/
├── root.zig           # Public API (Model, load, run)
├── proto.zig          # Protobuf parser
├── types.zig          # ONNX type definitions
├── builder.zig        # Proto → Graph conversion
├── graph.zig          # Runtime graph structure
├── executor.zig       # Graph execution
├── planner.zig        # Buffer planning
└── ops/
    ├── dispatch.zig   # Op routing
    ├── concat.zig     # NEW kernel
    ├── split.zig      # NEW kernel
    ├── slice.zig      # NEW kernel
    └── dequantize.zig # Quantization ops
```

---

## 9. Testing Strategy

### 9.1 Unit Tests

- Protobuf parser: Known-good .onnx snippets
- Graph builder: Simple graphs (Add, MatMul)
- Each op: Golden reference from PyTorch/ONNX Runtime

### 9.2 Integration Tests

- LeNet ONNX export → Compare with native tenzor LeNet
- MobileNet (common benchmark)
- Chatterbox components

### 9.3 Conformance

Use ONNX test suite (`onnx/backend/test/data/node/`).

---

## 10. Implementation Phases

### Phase 1: Foundation (Week 1-2)
- [ ] Protobuf parser (varint, length-delimited, fixed)
- [ ] ONNX types (ModelProto, GraphProto, NodeProto, TensorProto)
- [ ] Basic graph builder
- [ ] Add op implementation
- [ ] Test with simple Add graph

### Phase 2: Core Ops (Week 3-4)
- [ ] MatMul, Gemm
- [ ] Elementwise ops (Relu, Gelu, etc.)
- [ ] Softmax, LayerNorm
- [ ] Gather, Reshape, Transpose
- [ ] Test with small transformer block

### Phase 3: Chatterbox (Week 5-6)
- [ ] Concat, Split, Slice
- [ ] DequantizeLinear (Q8)
- [ ] Q4 block dequantization
- [ ] Full Chatterbox embed_tokens + language_model
- [ ] Audio generation demo

### Phase 4: Polish (Week 7-8)
- [ ] WASM build
- [ ] Buffer planning optimization
- [ ] Performance profiling
- [ ] Interactive demo page
- [ ] Documentation

---

## 11. Open Questions

1. **Opset versioning:** How strictly to handle opset differences?
   - Proposal: Target opset 17+ (modern), warn on older

2. **External data:** ONNX can reference external weight files.
   - Proposal: Support via path resolution

3. **Subgraphs:** Ops like If/Loop have nested graphs.
   - Proposal: Phase 2, not needed for Chatterbox

4. **Custom ops:** Domain-specific ops.
   - Proposal: Error with helpful message, suggest export options

---

## 12. References

- [ONNX Spec](https://onnx.ai/onnx/intro/)
- [ONNX Operators](https://onnx.ai/onnx/operators/)
- [Protobuf Encoding](https://protobuf.dev/programming-guides/encoding/)
- [Chatterbox ONNX](https://huggingface.co/ResembleAI/chatterbox-turbo-ONNX)
- [zann IR Design](../../../zann/book/src/architecture/ir.md) (similar patterns)
