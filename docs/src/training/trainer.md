# Trainer

The `Trainer` struct provides a unified training loop abstraction that integrates metrics logging, LR scheduling, early stopping, and optional TUI visualization.

## Configuration

```zig
const TrainerConfig = struct {
    epochs: u32 = 10,
    batch_size: u32 = 64,
    learning_rate: f32 = 0.01,
    scheduler: SchedulerType = .cosine,
    warmup_steps: u64 = 0,
    min_lr: f32 = 0.0001,
    checkpoint_dir: ?[]const u8 = null,
    checkpoint_every: u32 = 1,
    log_dir: ?[]const u8 = null,
    use_tui: bool = true,
    early_stopping_patience: u32 = 0,
    seed: u64 = 42,
    model_name: []const u8 = "model",
};
```

## Basic Usage

```zig
var trainer = try Trainer.init(allocator, .{
    .epochs = 10,
    .batch_size = 64,
    .learning_rate = 0.01,
});
defer trainer.deinit();
```

## Training Loop Methods

### `beginEpoch`

Call at the start of each epoch:

```zig
trainer.beginEpoch(epoch_number, total_batches);
```

### `recordBatch`

Call after each batch to record metrics and update TUI:

```zig
try trainer.recordBatch(loss, accuracy, batch_time_ms);
```

### `endEpoch`

Call at the end of each epoch with validation metrics:

```zig
try trainer.endEpoch(val_loss, val_acc, epoch_time_sec);
```

### `shouldContinue`

Check if training should continue (not stopped early, not quit by user):

```zig
while (trainer.shouldContinue()) {
    // ...
}
```

### `getLR`

Get current learning rate from scheduler:

```zig
const lr = trainer.getLR();
```

## State Tracking

The trainer maintains training state accessible via `trainer.state`:

```zig
const TrainerState = struct {
    epoch: u32,
    total_epochs: u32,
    batch: u32,
    total_batches: u32,
    global_step: u64,
    train_loss: f32,
    train_acc: f32,
    val_loss: f32,
    val_acc: f32,
    best_val_loss: f32,
    best_val_acc: f32,
    best_epoch: u32,
    lr: f32,
    should_stop: bool,
};
```

## Final Statistics

Get training statistics after completion:

```zig
const stats = trainer.getStats();

const TrainStats = struct {
    final_epoch: u32,
    final_train_loss: f32,
    final_train_acc: f32,
    final_val_loss: f32,
    final_val_acc: f32,
    best_val_acc: f32,
    best_epoch: u32,
    total_time_sec: f64,
    stopped_early: bool,
};
```
