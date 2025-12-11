# Training Overview

Tenzor provides production-ready training infrastructure with:

- **Trainer abstraction** - Unified training loop with metrics, callbacks
- **LR schedulers** - Constant, step decay, cosine annealing, warmup
- **Checkpointing** - Save/resume training with `.tenzor` format
- **Early stopping** - Patience-based stopping on validation metrics
- **TUI dashboard** - Real-time visualization of training progress

## Quick Start

```zig
const tenzor = @import("tenzor");
const Trainer = tenzor.training.Trainer;

var trainer = try Trainer.init(allocator, .{
    .epochs = 10,
    .batch_size = 64,
    .learning_rate = 0.01,
    .scheduler = .cosine,
    .early_stopping_patience = 5,
    .use_tui = true,
});
defer trainer.deinit();

// Training loop
var epoch: u32 = 1;
while (trainer.shouldContinue()) : (epoch += 1) {
    trainer.beginEpoch(epoch, num_batches);

    for (batches) |batch| {
        // ... forward, backward, update ...
        try trainer.recordBatch(loss, accuracy, batch_time_ms);
    }

    try trainer.endEpoch(val_loss, val_acc, epoch_time_sec);
}

const stats = trainer.getStats();
```

## Components

| Component | Description |
|-----------|-------------|
| `Trainer` | Main training controller |
| `Scheduler` | Learning rate scheduling |
| `MetricsLogger` | CSV/JSON metrics export |
| `EarlyStopping` | Patience-based stopping |
| `Dashboard` | TUI visualization |

## CLI Integration

Training is also available via CLI:

```bash
tenzor train -e 10 --scheduler cosine --patience 5
```

See [CLI Documentation](../cli/train.md) for full options.
