# MNIST Dataset

Load the MNIST handwritten digit dataset.

## Overview

MNIST is a classic dataset of 28x28 grayscale handwritten digit images (0-9).

- Training set: 60,000 images
- Test set: 10,000 images

## Loading MNIST

```zig
const mnist = @import("tenzor").io.mnist;

var train_data = try mnist.MNISTDataset.load(
    allocator,
    "data/mnist/train-images-idx3-ubyte",
    "data/mnist/train-labels-idx1-ubyte",
);
defer train_data.deinit();

std.debug.print("Loaded {} samples\n", .{train_data.num_samples});
```

## Dataset API

### MNISTDataset

```zig
const MNISTDataset = struct {
    images: []f32,      // Normalized [0, 1] pixel values
    labels: []u8,       // Labels 0-9
    num_samples: usize,

    pub fn load(
        allocator: std.mem.Allocator,
        images_path: []const u8,
        labels_path: []const u8,
    ) !MNISTDataset;

    pub fn deinit(self: *MNISTDataset) void;

    pub fn numBatches(self: *const MNISTDataset, batch_size: u32) usize;

    pub fn getBatch(
        self: *const MNISTDataset,
        batch_idx: usize,
        batch_size: u32,
    ) Batch;

    pub fn shuffle(self: *MNISTDataset, rng: std.rand.Random) void;
};
```

### Batch

```zig
const Batch = struct {
    images: []const f32,  // [batch_size * 784] flattened images
    labels: []const u8,   // [batch_size] labels
};
```

## Training Loop Example

```zig
const batch_size = 64;
const num_batches = train_data.numBatches(batch_size);

for (0..num_epochs) |epoch| {
    train_data.shuffle(prng.random());

    for (0..num_batches) |batch_idx| {
        const batch = train_data.getBatch(batch_idx, batch_size);

        // batch.images: [batch_size * 784] f32
        // batch.labels: [batch_size] u8

        // ... forward, backward, update ...
    }
}
```

## Data Format

### Images (IDX3)

```
┌─────────────────────────────────────┐
│ Magic number: 0x00000803            │
│ Number of images: u32 big-endian    │
│ Rows: 28                            │
│ Cols: 28                            │
├─────────────────────────────────────┤
│ Image data: [N * 28 * 28] u8        │
│ (row-major, 0-255)                  │
└─────────────────────────────────────┘
```

### Labels (IDX1)

```
┌─────────────────────────────────────┐
│ Magic number: 0x00000801            │
│ Number of labels: u32 big-endian    │
├─────────────────────────────────────┤
│ Label data: [N] u8 (0-9)            │
└─────────────────────────────────────┘
```

## Downloading MNIST

Download from http://yann.lecun.com/exdb/mnist/:

```bash
mkdir -p data/mnist
cd data/mnist

# Download and decompress
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

gunzip *.gz
```

Expected files:
```
data/mnist/
  train-images-idx3-ubyte
  train-labels-idx1-ubyte
  t10k-images-idx3-ubyte
  t10k-labels-idx1-ubyte
```

## CLI Usage

```bash
# Train on MNIST (auto-detects data/mnist/)
tenzor train

# Custom data directory
tenzor train -d /path/to/mnist
```

## Image Preprocessing

Images are automatically:
- Normalized from [0, 255] to [0, 1]
- Flattened from 28x28 to 784

For LeNet-5, images are reshaped to [1, 28, 28] (channels-first).
