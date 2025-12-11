# Interactive Demo

Draw a digit below and watch Tenzor recognize it in real-time. This runs entirely in your browser using WebAssembly — no server, no network requests, just ~3KB of WASM and ~174KB of model weights.

<div id="demo-container" style="text-align: center; margin: 2em 0;">
  <div style="display: inline-flex; align-items: flex-start; gap: 1em;">
    <div>
      <canvas id="canvas" width="280" height="280" style="border: 2px solid #333; border-radius: 8px; cursor: crosshair; touch-action: none;"></canvas>
      <div style="font-size: 0.8em; color: #666; margin-top: 0.3em;">Draw here</div>
    </div>
    <div>
      <canvas id="preview" width="140" height="140" style="border: 2px solid #666; border-radius: 4px; image-rendering: pixelated;"></canvas>
      <div style="font-size: 0.8em; color: #666; margin-top: 0.3em;">Model sees (28×28)</div>
    </div>
  </div>
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
  const preview = document.getElementById('preview');
  const previewCtx = preview.getContext('2d');
  const result = document.getElementById('result');
  const confidence = document.getElementById('confidence');
  const status = document.getElementById('status');
  const clearBtn = document.getElementById('clear-btn');
  const predictBtn = document.getElementById('predict-btn');

  // Initialize preview
  previewCtx.fillStyle = 'black';
  previewCtx.fillRect(0, 0, 140, 140);

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
    previewCtx.fillStyle = 'black';
    previewCtx.fillRect(0, 0, 140, 140);
    result.textContent = 'Draw a digit (0-9)';
    confidence.textContent = '';
  });

  // Predict button
  predictBtn.addEventListener('click', predict);

  // Run prediction
  function predict() {
    if (!wasm) return;

    const imageData = ctx.getImageData(0, 0, 280, 280);

    // Find bounding box of drawn content
    let minX = 280, minY = 280, maxX = 0, maxY = 0;
    for (let y = 0; y < 280; y++) {
      for (let x = 0; x < 280; x++) {
        const idx = (y * 280 + x) * 4;
        if (imageData.data[idx] > 10) {
          minX = Math.min(minX, x);
          minY = Math.min(minY, y);
          maxX = Math.max(maxX, x);
          maxY = Math.max(maxY, y);
        }
      }
    }

    // Check if canvas is blank
    if (maxX <= minX || maxY <= minY) {
      result.textContent = '?';
      confidence.textContent = 'Draw something first';
      return;
    }

    // Add padding around bounding box
    const pad = 20;
    minX = Math.max(0, minX - pad);
    minY = Math.max(0, minY - pad);
    maxX = Math.min(279, maxX + pad);
    maxY = Math.min(279, maxY + pad);

    // Make it square (use larger dimension)
    const w = maxX - minX;
    const h = maxY - minY;
    const size = Math.max(w, h);
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;

    // New square bounding box centered on digit
    const left = Math.max(0, Math.floor(cx - size / 2));
    const top = Math.max(0, Math.floor(cy - size / 2));

    // Sample from square region into 28x28
    const input = new Float32Array(28 * 28);
    const step = size / 28;

    for (let y = 0; y < 28; y++) {
      for (let x = 0; x < 28; x++) {
        let sum = 0;
        let count = 0;

        // Average pixels in this cell
        const startY = Math.floor(top + y * step);
        const endY = Math.floor(top + (y + 1) * step);
        const startX = Math.floor(left + x * step);
        const endX = Math.floor(left + (x + 1) * step);

        for (let py = startY; py < endY && py < 280; py++) {
          for (let px = startX; px < endX && px < 280; px++) {
            const idx = (py * 280 + px) * 4;
            sum += imageData.data[idx];
            count++;
          }
        }
        input[y * 28 + x] = count > 0 ? sum / (count * 255) : 0;
      }
    }

    // Draw preview (scaled 5x from 28x28 to 140x140)
    const previewData = previewCtx.createImageData(140, 140);
    for (let y = 0; y < 140; y++) {
      for (let x = 0; x < 140; x++) {
        const srcIdx = Math.floor(y / 5) * 28 + Math.floor(x / 5);
        const val = Math.floor(input[srcIdx] * 255);
        const dstIdx = (y * 140 + x) * 4;
        previewData.data[dstIdx] = val;
        previewData.data[dstIdx + 1] = val;
        previewData.data[dstIdx + 2] = val;
        previewData.data[dstIdx + 3] = 255;
      }
    }
    previewCtx.putImageData(previewData, 0, 0);

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
