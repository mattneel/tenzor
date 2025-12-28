# ONNX Loader Implementation Plan

**Goal:** Load any ONNX model and execute using tenzor kernels.
**Demo:** Chatterbox TTS in WASM.

---

## Phase 1: Foundation

### 1.1 Protobuf Parser (`src/onnx/proto.zig`)

```
[ ] Varint decoding (LEB128)
[ ] Wire type parsing (varint, fixed32/64, length-delimited)
[ ] Field header parsing
[ ] Repeated field handling
[ ] Nested message parsing
```

Test: Parse a minimal .onnx file header.

### 1.2 ONNX Types (`src/onnx/types.zig`)

```
[ ] DataType enum (FLOAT, FLOAT16, INT8, etc.)
[ ] TensorProto struct
[ ] ValueInfoProto struct
[ ] AttributeProto struct
[ ] NodeProto struct
[ ] GraphProto struct
[ ] ModelProto struct
```

Test: Round-trip parse a simple Add graph.

### 1.3 Graph Builder (`src/onnx/builder.zig`)

```
[ ] Parse ModelProto from bytes
[ ] Extract initializers (weights)
[ ] Build tensor registry
[ ] Convert NodeProto → Node
[ ] Topological ordering (already in ONNX order)
```

Test: Build graph from Add model, verify structure.

### 1.4 Runtime Graph (`src/onnx/graph.zig`)

```
[ ] Graph struct (nodes, tensors, weights)
[ ] TensorInfo (name, dtype, shape)
[ ] Node struct (op_type, inputs, outputs, attributes)
[ ] OpType enum (start with Add only)
[ ] Tensor name → ID mapping
```

Test: Create graph programmatically, verify lookups.

---

## Phase 2: Core Ops

### 2.1 Executor Framework (`src/onnx/executor.zig`)

```
[ ] Executor struct
[ ] setInput() - bind input tensors
[ ] run() - execute all nodes
[ ] getOutput() - retrieve results
[ ] Buffer pool for intermediates
[ ] Op dispatch switch
```

Test: Execute Add graph, verify output.

### 2.2 Elementwise Ops

```
[ ] Add, Sub, Mul, Div (broadcasting)
[ ] Neg, Abs, Exp, Log, Sqrt
[ ] Relu, Sigmoid, Tanh
[ ] Gelu, Silu
```

Kernels: Use existing `elementwise.zig`.
Test: Each op against PyTorch reference.

### 2.3 Matrix Ops

```
[ ] MatMul (2D, 3D batched)
[ ] Gemm (alpha, beta, transA, transB)
```

Kernels: Use existing `matmul.zig`.
Test: Various shapes, transpose combinations.

### 2.4 Normalization

```
[ ] Softmax (axis parameter)
[ ] LayerNormalization
```

Kernels: Use existing `softmax.zig`, `layernorm.zig`.
Test: Transformer attention pattern.

### 2.5 Shape Ops

```
[ ] Reshape (no-op, just metadata)
[ ] Transpose (permutation)
[ ] Squeeze/Unsqueeze (no-op)
[ ] Gather (embeddings)
```

Kernels: Use existing `transpose.zig`, `gather.zig`.
Test: Attention reshape/transpose dance.

---

## Phase 3: Extended Ops

### 3.1 New Kernels Needed

```
[ ] Concat - join tensors along axis
[ ] Split - split tensor along axis
[ ] Slice - general indexing
```

Implement in `src/backend/cpu/kernels/`.

### 3.2 Quantization

```
[ ] DequantizeLinear (int8 → f32)
[ ] Q4 block dequantization
[ ] Integration with existing W8A32 matmul
```

Test: Load Q8 model, compare with FP32.

### 3.3 Convolution (if needed)

```
[ ] Conv with groups, dilations
[ ] BatchNormalization
```

Kernels: Extend existing `conv2d.zig`.

---

## Phase 4: Chatterbox Integration

### 4.1 Model Loading

```
[ ] Load embed_tokens.onnx
[ ] Load language_model.onnx
[ ] Load speech_encoder.onnx
[ ] Load conditional_decoder.onnx
```

### 4.2 Inference Pipeline

```
[ ] Text tokenization
[ ] Reference audio encoding
[ ] Autoregressive token generation
[ ] KV-cache management
[ ] Audio decoding
```

### 4.3 Demo

```
[ ] WASM build
[ ] Simple web UI (text input → audio)
[ ] Reference audio upload
```

---

## Phase 5: Polish

```
[ ] Buffer planning optimization
[ ] Memory profiling
[ ] Performance benchmarks
[ ] Error messages with op names
[ ] Documentation
[ ] Examples
```

---

## File Structure

```
src/onnx/
├── root.zig           # pub const Model, load(), etc.
├── proto.zig          # Protobuf decoder
├── types.zig          # ONNX message types
├── builder.zig        # Proto → Graph
├── graph.zig          # Runtime graph
├── executor.zig       # Execution engine
├── planner.zig        # Buffer planning
└── ops/
    ├── dispatch.zig   # Op routing
    ├── concat.zig
    ├── split.zig
    ├── slice.zig
    └── dequantize.zig

src/backend/cpu/kernels/
├── concat.zig         # NEW
├── split.zig          # NEW
└── slice.zig          # NEW
```

---

## Test Models

| Model | Purpose | Ops Tested |
|-------|---------|------------|
| add.onnx | Smoke test | Add |
| matmul.onnx | Matrix ops | MatMul |
| transformer_block.onnx | Attention | MatMul, Softmax, LayerNorm, Reshape, Transpose |
| mobilenet.onnx | Conv network | Conv, BatchNorm, Relu, GlobalAvgPool |
| chatterbox_embed.onnx | Target | Gather, LayerNorm, MatMul |
| chatterbox_lm.onnx | Target | Full transformer |

---

## Milestones

- [ ] **M1:** Parse and execute Add graph
- [ ] **M2:** Execute transformer attention block
- [ ] **M3:** Load and run chatterbox embed_tokens
- [ ] **M4:** Full chatterbox inference (CPU)
- [ ] **M5:** WASM demo with audio playback

---

## Resources

- Design doc: `docs/ONNX_LOADER_DESIGN.md`
- ONNX spec: https://onnx.ai/onnx/
- Chatterbox: https://huggingface.co/ResembleAI/chatterbox-turbo-ONNX
- Protobuf encoding: https://protobuf.dev/programming-guides/encoding/
