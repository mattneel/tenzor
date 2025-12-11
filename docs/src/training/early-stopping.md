# Early Stopping

Prevent overfitting by stopping training when validation metrics stop improving.

## Configuration

```zig
const EarlyStopping = struct {
    patience: u32,      // Epochs to wait before stopping
    min_delta: f32,     // Minimum improvement threshold
    monitor: Metric,    // Metric to monitor
};

const Metric = enum {
    val_loss,  // Stop when loss stops decreasing
    val_acc,   // Stop when accuracy stops increasing
};
```

## Basic Usage

```zig
var early_stopping = EarlyStopping.init(
    5,       // patience: wait 5 epochs
    0.001,   // min_delta: 0.1% minimum improvement
    .val_acc // monitor validation accuracy
);

for (epochs) |_| {
    // ... training ...

    if (early_stopping.check(&trainer.state)) {
        std.debug.print("Early stopping triggered at epoch {}\n", .{epoch});
        break;
    }
}
```

## With Trainer

```zig
var trainer = try Trainer.init(allocator, .{
    .early_stopping_patience = 5,
});

// Trainer automatically checks early stopping in endEpoch()
while (trainer.shouldContinue()) {
    trainer.beginEpoch(epoch, num_batches);
    // ... training ...
    try trainer.endEpoch(val_loss, val_acc, epoch_time);
}

const stats = trainer.getStats();
if (stats.stopped_early) {
    std.debug.print("Stopped early at epoch {}\n", .{stats.final_epoch});
}
```

## How It Works

1. After each epoch, compare current metric to best metric
2. If improvement > `min_delta`, reset patience counter
3. If no improvement, increment wait counter
4. If wait counter >= patience, trigger stop

```
Epoch  Val Acc  Best   Wait  Action
  1     85.0%   85.0%   0    New best
  2     87.0%   87.0%   0    New best
  3     86.5%   87.0%   1    No improvement
  4     86.8%   87.0%   2    No improvement
  5     87.1%   87.1%   0    New best (> min_delta)
  6     86.9%   87.1%   1    No improvement
  7     86.7%   87.1%   2    No improvement
  8     86.5%   87.1%   3    No improvement
  9     86.3%   87.1%   4    No improvement
 10     86.0%   87.1%   5    STOP (patience exhausted)
```

## CLI Usage

```bash
# Enable early stopping with patience of 5 epochs
tenzor train --patience 5

# Disable early stopping (default)
tenzor train --patience 0
```

## State Tracking

```zig
const EarlyStopping = struct {
    // Configuration
    patience: u32,
    min_delta: f32,
    monitor: Metric,

    // State
    best_value: f32,
    wait: u32,
    stopped_epoch: ?u32,
};
```

## Tips

- **Patience**: Start with 5-10 epochs for most tasks
- **min_delta**: Use 0.001 (0.1%) to avoid stopping on noise
- **Monitor**: Use `val_acc` for classification, `val_loss` for regression
- **Restore best**: After early stopping, load the best checkpoint
