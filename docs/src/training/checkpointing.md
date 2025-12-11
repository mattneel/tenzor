# Checkpointing

Save and resume training state using the `.tenzor` format for fast mmap-based loading.

## Checkpoint Contents

A checkpoint includes:

- **Model weights** - All trainable parameters
- **Optimizer state** - Momentum, Adam m/v buffers
- **Training metadata** - Epoch, step, best metrics, LR

## Saving Checkpoints

```zig
const checkpoint = @import("tenzor").nn.checkpoint;

try checkpoint.Checkpoint.save(
    "checkpoints/epoch_5.tenzor",
    weights,
    optimizer,
    .{
        .epoch = 5,
        .global_step = 4685,
        .best_val_loss = 0.198,
        .best_val_acc = 0.963,
        .best_epoch = 3,
        .learning_rate = 0.0087,
        .model_name = "lenet",
        .timestamp = std.time.timestamp(),
    },
);
```

## Loading Checkpoints

```zig
var loaded = try checkpoint.Checkpoint.load(
    allocator,
    "checkpoints/epoch_5.tenzor",
);
defer loaded.close();

// Access metadata
const meta = loaded.metadata;
std.debug.print("Resuming from epoch {}\n", .{meta.epoch});

// Load weights into model
try loaded.loadWeights(&model.weights);

// Resume optimizer state
try loaded.resumeOptimizer(&optimizer);
```

## Checkpoint Metadata

```zig
const CheckpointMetadata = struct {
    epoch: u32,
    global_step: u64,
    best_val_loss: f32,
    best_val_acc: f32,
    best_epoch: u32,
    learning_rate: f32,
    model_name: []const u8,
    timestamp: i64,
};
```

## .tenzor Format Layout

```
┌─────────────────────────────────────┐
│ Header (64 bytes)                   │
│   magic: "TENZOR\x00\x00"           │
│   version, tensor_count, offsets    │
├─────────────────────────────────────┤
│ Metadata (JSON)                     │
│   { "epoch": 5, "best_val_acc": ... }│
├─────────────────────────────────────┤
│ Tensor Index                        │
│   name_hash, dtype, shape, offset   │
├─────────────────────────────────────┤
│ Tensor Data (page-aligned)          │
│   conv1_weight, conv1_bias, ...     │
│   optim/conv1_weight/v, ...         │
└─────────────────────────────────────┘
```

## CLI Usage

```bash
# Save checkpoints during training
tenzor train --checkpoint checkpoints/

# Resume from checkpoint
tenzor train --resume checkpoints/epoch_5.tenzor
```

## Automatic Checkpointing

With `TrainerConfig`:

```zig
var trainer = try Trainer.init(allocator, .{
    .checkpoint_dir = "checkpoints",
    .checkpoint_every = 1,  // Save every epoch
});
```

## Best Model Tracking

The trainer automatically tracks the best model:

```zig
if (val_acc > trainer.state.best_val_acc) {
    trainer.state.best_val_acc = val_acc;
    trainer.state.best_epoch = epoch;
    // Save best model checkpoint
}
```
