# train Command

Train LeNet-5 on the MNIST dataset with optional TUI dashboard.

## Usage

```bash
tenzor train [OPTIONS]
```

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `-d, --data-dir <PATH>` | MNIST data directory | `data/mnist` |
| `-e, --epochs <NUM>` | Number of training epochs | `10` |
| `-b, --batch-size <NUM>` | Batch size | `64` |
| `-l, --lr <FLOAT>` | Learning rate | `0.01` |
| `--momentum <FLOAT>` | SGD momentum | `0.9` |
| `--scheduler <TYPE>` | LR scheduler | `cosine` |
| `--warmup <NUM>` | Warmup steps | `0` |
| `--patience <NUM>` | Early stopping patience | `0` (disabled) |
| `--checkpoint <PATH>` | Checkpoint directory | none |
| `--no-tui` | Disable TUI dashboard | false |
| `-s, --seed <SEED>` | Random seed | `42` |

## Scheduler Types

- `constant` - Fixed learning rate
- `step` - Reduce LR every N steps
- `cosine` - Cosine annealing to min_lr

## Examples

### Basic Training

```bash
tenzor train
```

### Custom Configuration

```bash
tenzor train \
  -e 20 \
  -b 128 \
  --lr 0.001 \
  --scheduler cosine \
  --warmup 500
```

### With Early Stopping

```bash
tenzor train \
  -e 100 \
  --patience 10 \
  --checkpoint checkpoints/
```

### Headless (CI/Scripts)

```bash
tenzor train --no-tui -e 10
```

## MNIST Data

Download MNIST data from http://yann.lecun.com/exdb/mnist/ and place in `data/mnist/`:

```
data/mnist/
  train-images-idx3-ubyte
  train-labels-idx1-ubyte
  t10k-images-idx3-ubyte
  t10k-labels-idx1-ubyte
```

## Output

With TUI enabled, displays real-time:
- Epoch/batch progress bars
- Loss and accuracy charts
- Current metrics
- Training log

Without TUI:
```
Epoch  1/10: loss=0.4523, train_acc=85.2%, val_acc=87.3%
Epoch  2/10: loss=0.2134, train_acc=93.4%, val_acc=94.1%
...
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error (data not found, etc.) |
