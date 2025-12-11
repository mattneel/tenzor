# TUI Dashboard

The training dashboard provides real-time visualization of training progress using ANSI escape sequences (no external dependencies).

## Features

- Progress bars for epoch and batch
- ASCII sparkline charts for loss and accuracy
- Live metrics display
- Scrolling log view
- ETA calculation
- Keyboard controls (q to quit)

## Layout

```
┌─────────────────────────────────────────────────────────────────┐
│ tenzor train - LeNet-5/MNIST                        [q]uit      │
├─────────────────────────────────────────────────────────────────┤
│ Epoch: 5/10  [████████████████░░░░░░░░░░] 50%   ETA: 2m 30s     │
│ Batch: 234/937 [████████░░░░░░░░░░░░░░░░░] 25%  1234 samples/s  │
├─────────────────────────────────────────────────────────────────┤
│ Loss                              │ Accuracy                    │
│ 2.0 ┤                             │ 100% ┤              ****    │
│     │\                            │      │         ****         │
│ 1.0 ┤ \___                        │  50% ┤    ****              │
│     │     \____                   │      │****                  │
│ 0.0 ┤          ----               │   0% ┤                      │
├─────────────────────────────────────────────────────────────────┤
│ train_loss: 0.234   train_acc: 94.2%   lr: 0.0087              │
│ val_loss:   0.198   val_acc:   96.1%   best: 96.3% (epoch 3)   │
├─────────────────────────────────────────────────────────────────┤
│ [Log]                                                           │
│ Epoch 5: train_loss=0.234, val_acc=96.1%                       │
│ Checkpoint saved: checkpoints/epoch_5.tenzor                   │
│ LR reduced: 0.01 -> 0.0087                                     │
└─────────────────────────────────────────────────────────────────┘
```

## Widgets

### ProgressBar

Shows completion percentage with filled/empty blocks:

```
Epoch: 5/10  [████████████████░░░░░░░░░░] 50%
```

### SparklineChart

ASCII chart using braille characters for data visualization:

```
Loss
2.0 ┤
    │\
1.0 ┤ \___
    │     \____
0.0 ┤          ----
```

### MetricsPanel

Key-value display of current metrics:

```
train_loss: 0.234   train_acc: 94.2%   lr: 0.0087
```

### LogView

Scrolling log of training events:

```
Epoch 5: train_loss=0.234, val_acc=96.1%
Checkpoint saved: checkpoints/epoch_5.tenzor
```

## Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit training |
| `Q` | Quit training |
| `Ctrl+C` | Quit training |
| `Ctrl+Q` | Quit training |

## Programmatic Usage

```zig
var dashboard = try Dashboard.init(
    allocator,
    "Model Name",
    total_epochs,
    total_batches,
);
defer dashboard.deinit();

// Update with training state
try dashboard.update(.{
    .epoch = 5,
    .total_epochs = 10,
    .batch = 234,
    .total_batches = 937,
    .train_loss = 0.234,
    .train_acc = 0.942,
    // ...
});

// Check for quit
if (dashboard.shouldQuit()) {
    break;
}

// Add log message
try dashboard.addLog("Checkpoint saved");
```

## Disabling TUI

For CI/CD or scripted environments:

```bash
tenzor train --no-tui
```

Or programmatically:

```zig
var trainer = try Trainer.init(allocator, .{
    .use_tui = false,
});
```

When TUI is disabled, progress is printed to stderr in plain text.

## Terminal Requirements

- ANSI escape sequence support
- Alternate screen buffer support
- 80+ column width recommended
- Works in most modern terminals (iTerm2, Terminal.app, GNOME Terminal, Windows Terminal)
