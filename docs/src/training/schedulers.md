# LR Scheduling

Tenzor provides several learning rate schedulers for training optimization.

## Scheduler Types

### Constant

Fixed learning rate throughout training:

```zig
const scheduler = Scheduler.initConstant(0.01);
```

### Step Decay

Reduce LR by factor every N steps:

```zig
const scheduler = Scheduler.initStep(
    0.01,   // initial_lr
    0.1,    // decay_factor (multiply by this)
    1000,   // step_size (steps between decays)
);
```

### Cosine Annealing

Smooth cosine decay to minimum LR:

```zig
const scheduler = Scheduler.initCosine(
    0.01,    // initial_lr
    0.0001,  // min_lr
    10000,   // total_steps
);
```

### Warmup

Linear warmup from 0 to target LR:

```zig
const scheduler = Scheduler.initWarmup(
    0.01,  // target_lr
    500,   // warmup_steps
);
```

### Warmup + Cosine

Warmup followed by cosine decay:

```zig
const scheduler = Scheduler.initWarmupCosine(
    0.01,    // initial_lr
    0.0001,  // min_lr
    500,     // warmup_steps
    10000,   // total_steps
);
```

## Usage

### With Trainer

The trainer automatically manages the scheduler:

```zig
var trainer = try Trainer.init(allocator, .{
    .scheduler = .cosine,
    .warmup_steps = 500,
    .learning_rate = 0.01,
    .min_lr = 0.0001,
});

// Get current LR
const lr = trainer.getLR();
```

### Standalone

```zig
var scheduler = Scheduler.initCosine(0.01, 0.0001, 10000);

for (0..10000) |step| {
    const lr = scheduler.getLR(step);
    // Use lr for optimization
}
```

## Visualization

```
LR Schedule: Warmup + Cosine

     ^
0.01 |        ___
     |       /   \
     |      /     \
     |     /       \___
0.001|____/            \____
     +-----------------------------> steps
         warmup    decay
```

## CLI Options

```bash
tenzor train --scheduler cosine --warmup 500 --lr 0.01
```

| Flag | Description |
|------|-------------|
| `--scheduler` | `constant`, `step`, `cosine` |
| `--warmup` | Number of warmup steps |
| `--lr` | Initial learning rate |
