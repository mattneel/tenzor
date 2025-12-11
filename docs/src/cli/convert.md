# convert Command

Convert SafeTensors files to the `.tenzor` mmap-friendly format.

## Usage

```bash
tenzor convert <INPUT> [-o <OUTPUT>]
```

## Arguments

| Argument | Description |
|----------|-------------|
| `<INPUT>` | Input .safetensors file |

## Options

| Option | Description |
|--------|-------------|
| `-o, --output <PATH>` | Output .tenzor file (default: `<input>.tenzor`) |

## Examples

### Basic Conversion

```bash
tenzor convert model.safetensors
# Creates: model.tenzor
```

### Custom Output

```bash
tenzor convert model.safetensors -o optimized/model.tenzor
```

## Output

```
Converting SafeTensors to .tenzor
=================================
Input:  model.safetensors
Output: model.tenzor

Conversion complete!
  Input size:  86.45 MB
  Output size: 86.12 MB
  Time: 234 ms
```

## Benefits of .tenzor

| Feature | SafeTensors | .tenzor |
|---------|-------------|---------|
| Load time | ~50ms (parse JSON header) | <1ms (mmap) |
| Memory | Copy data to RAM | Zero-copy from file |
| Format | JSON header + raw data | Binary header + aligned data |

## Format Details

The `.tenzor` format:
- 64-byte binary header
- JSON metadata section
- Page-aligned tensor index
- Page-aligned tensor data (4KB alignment)

See [.tenzor Format](../io/tenzor-format.md) for full specification.

## Use Cases

- **Faster inference**: Convert once, load instantly
- **Large models**: Mmap avoids loading entire model into RAM
- **Checkpointing**: Training checkpoints use .tenzor format
