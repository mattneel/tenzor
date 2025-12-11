# LeNet-5

LeNet-5 is a classic convolutional neural network for handwritten digit recognition.

## Architecture

```
Input: [1, 28, 28] (MNIST image)
  │
  ├─ Conv2D(1→6, 5x5, pad=2) → [6, 28, 28]
  ├─ ReLU
  ├─ MaxPool(2x2) → [6, 14, 14]
  │
  ├─ Conv2D(6→16, 5x5) → [16, 10, 10]
  ├─ ReLU
  ├─ MaxPool(2x2) → [16, 5, 5]
  │
  ├─ Flatten → [400]
  │
  ├─ Linear(400→120)
  ├─ ReLU
  │
  ├─ Linear(120→84)
  ├─ ReLU
  │
  ├─ Linear(84→10)
  └─ Softmax → [10] (class probabilities)
```

## Usage

### Creating the Model

```zig
const lenet = @import("tenzor").model.lenet;

const config = lenet.LeNetConfig{
    .batch_size = 64,
};

var model = try lenet.LeNet.init(allocator, config);
defer model.deinit();

// Initialize weights with Kaiming initialization
var prng = std.Random.DefaultPrng.init(42);
model.weights.initKaiming(prng.random());
```

### Forward Pass

```zig
const output = model.forward(batch_images, actual_batch_size);
// output: [batch_size, 10] logits
```

### Computing Loss

```zig
const metrics = model.computeLoss(batch_labels, actual_batch_size);
// metrics.loss: cross-entropy loss
// metrics.accuracy: classification accuracy
```

### Backward Pass

```zig
// Zero gradients
model.grads.zero();

// Compute loss gradient
model.computeLossGradient(batch_labels, actual_batch_size);

// Backpropagate
model.backward(batch_images, actual_batch_size);
```

### Parameter Update

```zig
// SGD update
fn sgdUpdate(params: []f32, grads: []const f32, lr: f32) void {
    for (params, grads) |*p, g| {
        p.* -= lr * g;
    }
}

sgdUpdate(model.weights.conv1_weight, model.grads.conv1_weight, learning_rate);
// ... repeat for all parameters
```

## Model Parameters

| Layer | Shape | Parameters |
|-------|-------|------------|
| conv1_weight | [6, 1, 5, 5] | 150 |
| conv1_bias | [6] | 6 |
| conv2_weight | [16, 6, 5, 5] | 2,400 |
| conv2_bias | [16] | 16 |
| fc1_weight | [400, 120] | 48,000 |
| fc1_bias | [120] | 120 |
| fc2_weight | [120, 84] | 10,080 |
| fc2_bias | [84] | 84 |
| fc3_weight | [84, 10] | 840 |
| fc3_bias | [10] | 10 |
| **Total** | | **61,706** |

## Training

### CLI

```bash
tenzor train -e 10 -b 64 --lr 0.01
```

### Programmatic

```zig
const Trainer = @import("tenzor").training.Trainer;

var trainer = try Trainer.init(allocator, .{
    .epochs = 10,
    .batch_size = 64,
    .learning_rate = 0.01,
    .scheduler = .cosine,
});
defer trainer.deinit();

// Training loop with trainer abstraction
```

## Expected Performance

With default hyperparameters:

| Metric | Value |
|--------|-------|
| Test accuracy | ~98-99% |
| Training time | ~2 min (10 epochs) |
| Convergence | ~5 epochs |

## Implementation Details

- **Activations**: ReLU (not original tanh/sigmoid)
- **Pooling**: Max pooling (not original average pooling)
- **Loss**: Cross-entropy with softmax
- **Initialization**: Kaiming/He initialization
