# Interactive Demo

Draw a digit below and watch Tenzor recognize it in real-time. This runs entirely in your browser using WebAssembly — no server, no network requests, just ~3KB of WASM and ~174KB of model weights.

<div id="demo-container" style="text-align: center; margin: 2em 0;">
  <canvas id="canvas" width="280" height="280" style="border: 2px solid #333; border-radius: 8px; cursor: crosshair; touch-action: none;"></canvas>
  <div style="margin-top: 1em;">
    <button id="predict-btn" style="padding: 0.5em 1.5em; font-size: 1.1em; cursor: pointer; background: #4a4; color: white; border: none; border-radius: 4px; margin-right: 0.5em;">Predict</button>
    <button id="clear-btn" style="padding: 0.5em 1.5em; font-size: 1em; cursor: pointer;">Clear</button>
  </div>
  <div id="result" style="margin-top: 1em; font-size: 2em; font-weight: bold;">
    Draw a digit (0-9)
  </div>
  <div id="confidence" style="font-size: 1em; color: #666;"></div>
  <div id="status" style="margin-top: 0.5em; font-size: 0.9em; color: #888;"></div>
</div>

<script>
(async function() {
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  const result = document.getElementById('result');
  const confidence = document.getElementById('confidence');
  const status = document.getElementById('status');
  const clearBtn = document.getElementById('clear-btn');
  const predictBtn = document.getElementById('predict-btn');

  let wasm = null;
  let inputPtr = null;
  let isDrawing = false;
  let lastX = 0;
  let lastY = 0;

  // Initialize canvas
  ctx.fillStyle = 'black';
  ctx.fillRect(0, 0, 280, 280);
  ctx.strokeStyle = 'white';
  ctx.lineWidth = 20;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  status.textContent = 'Loading model...';

  // Load WASM and weights
  try {
    // Load WASM module
    const wasmResponse = await fetch('tenzor.wasm');
    const wasmBytes = await wasmResponse.arrayBuffer();
    const wasmModule = await WebAssembly.instantiate(wasmBytes, {
      env: {
        memory: new WebAssembly.Memory({ initial: 32 }) // 2MB
      }
    });
    wasm = wasmModule.instance.exports;

    // Load weights
    const weightsResponse = await fetch('weights.bin');
    const weightsData = await weightsResponse.arrayBuffer();
    const weightsArray = new Uint8Array(weightsData);

    // Copy weights to WASM memory and initialize
    const weightsPtr = wasm.__heap_base || 0;
    const wasmMemory = new Uint8Array(wasm.memory.buffer);
    wasmMemory.set(weightsArray, weightsPtr);

    wasm.init(weightsPtr, weightsArray.length);
    inputPtr = wasm.getInputPtr();

    status.textContent = 'Ready! Draw a digit.';
    status.style.color = '#4a4';
  } catch (err) {
    status.textContent = 'Failed to load: ' + err.message;
    status.style.color = '#a44';
    console.error(err);
    return;
  }

  // Drawing handlers
  function getPos(e) {
    const rect = canvas.getBoundingClientRect();
    const clientX = e.clientX || (e.touches && e.touches[0].clientX);
    const clientY = e.clientY || (e.touches && e.touches[0].clientY);
    return {
      x: (clientX - rect.left) * (280 / rect.width),
      y: (clientY - rect.top) * (280 / rect.height)
    };
  }

  function startDraw(e) {
    e.preventDefault();
    isDrawing = true;
    const pos = getPos(e);
    lastX = pos.x;
    lastY = pos.y;
  }

  function draw(e) {
    if (!isDrawing) return;
    e.preventDefault();
    const pos = getPos(e);

    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();

    lastX = pos.x;
    lastY = pos.y;
  }

  function endDraw(e) {
    if (!isDrawing) return;
    isDrawing = false;
  }

  // Mouse events
  canvas.addEventListener('mousedown', startDraw);
  canvas.addEventListener('mousemove', draw);
  canvas.addEventListener('mouseup', endDraw);
  canvas.addEventListener('mouseleave', endDraw);

  // Touch events
  canvas.addEventListener('touchstart', startDraw);
  canvas.addEventListener('touchmove', draw);
  canvas.addEventListener('touchend', endDraw);

  // Clear button
  clearBtn.addEventListener('click', () => {
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, 280, 280);
    result.textContent = 'Draw a digit (0-9)';
    confidence.textContent = '';
  });

  // Predict button
  predictBtn.addEventListener('click', predict);

  // Run prediction
  function predict() {
    if (!wasm) return;

    // Downsample 280x280 to 28x28
    const imageData = ctx.getImageData(0, 0, 280, 280);
    const input = new Float32Array(28 * 28);
    let totalSum = 0;

    for (let y = 0; y < 28; y++) {
      for (let x = 0; x < 28; x++) {
        // Average 10x10 block
        let sum = 0;
        for (let dy = 0; dy < 10; dy++) {
          for (let dx = 0; dx < 10; dx++) {
            const px = x * 10 + dx;
            const py = y * 10 + dy;
            const idx = (py * 280 + px) * 4;
            // Use red channel (grayscale)
            sum += imageData.data[idx];
          }
        }
        // Normalize to [0, 1]
        input[y * 28 + x] = sum / (10 * 10 * 255);
        totalSum += input[y * 28 + x];
      }
    }

    // Check if canvas is blank
    if (totalSum < 5) {
      result.textContent = '?';
      confidence.textContent = 'Draw something first';
      return;
    }

    // Copy to WASM memory
    const wasmInput = new Float32Array(wasm.memory.buffer, inputPtr, 28 * 28);
    wasmInput.set(input);

    // Run inference
    const pred = wasm.predict();
    const conf = wasm.getConfidence();

    if (pred < 10) {
      result.textContent = pred;
      confidence.textContent = conf.toFixed(1) + '% confidence';
    } else {
      result.textContent = '?';
      confidence.textContent = 'Model not initialized';
    }
  }
})();
</script>

## How It Works

This demo showcases Tenzor's portability. The exact same LeNet-5 architecture used for training compiles to:

| Target | Size | Use Case |
|--------|------|----------|
| Native (x86_64) | ~100KB | CLI training & inference |
| WebAssembly | ~3KB | Browser inference |

The WASM module exports three functions:

```zig
export fn init(weight_data: [*]const u8, len: usize) void
export fn getInputPtr() [*]f32
export fn predict() u8
export fn getConfidence() f32
```

### Build It Yourself

```bash
# Build the WASM module
zig build wasm

# Output: docs/src/appendix/tenzor.wasm (~3KB)
```

### The Full Pipeline

1. **Canvas** captures your drawing at 280×280 pixels
2. **JavaScript** downsamples to 28×28 grayscale, normalizes to [0,1]
3. **WASM** runs the forward pass: Conv → Pool → Conv → Pool → FC → FC → FC
4. **Result** displayed with softmax confidence

Total latency: <1ms on modern hardware.

## Model Details

The model was trained using Tenzor's CLI:

```bash
tenzor train -e 20 --lr 0.1 --scheduler cosine --checkpoint ckpt/
```

- **Architecture**: LeNet-5 (44,426 parameters)
- **Dataset**: MNIST (60k train, 10k test)
- **Accuracy**: 99.1% on test set
- **Training time**: ~2 minutes on CPU

See [LeNet-5](../models/lenet.md) for architecture details and [Checkpointing](../training/checkpointing.md) for how the weights were saved.
