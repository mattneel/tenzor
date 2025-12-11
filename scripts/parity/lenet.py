#!/usr/bin/env python3
"""Generate test fixtures for LeNet parity testing.

Generates:
1. Random weights for LeNet
2. Random input and targets
3. Forward pass outputs at each layer
4. Backward pass gradients
5. Loss values

Usage:
    pip install torch
    python scripts/parity/lenet.py
"""

import json
import struct
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def save_f32_binary(path: Path, data: torch.Tensor):
    """Save tensor as raw f32 binary file."""
    data_np = data.detach().cpu().float().numpy().flatten()
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(data_np)}f", *data_np))


def save_u32_binary(path: Path, data: list[int]):
    """Save list of ints as raw u32 binary file."""
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(data)}I", *data))


class LeNetNHWC(nn.Module):
    """LeNet-5 in NHWC format (to match Zig implementation)."""

    def __init__(self):
        super().__init__()
        # Conv1: [6, 1, 5, 5] in NCHW, but we'll store as [6, 5, 5, 1] for NHWC
        self.conv1_weight = nn.Parameter(torch.randn(6, 1, 5, 5))
        self.conv1_bias = nn.Parameter(torch.zeros(6))

        # Conv2: [16, 6, 5, 5]
        self.conv2_weight = nn.Parameter(torch.randn(16, 6, 5, 5))
        self.conv2_bias = nn.Parameter(torch.zeros(16))

        # FC layers
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x, save_intermediates=False):
        """Forward pass. x: [N, 1, 28, 28] NCHW."""
        intermediates = {}

        # Conv1 + ReLU
        x = F.conv2d(x, self.conv1_weight, self.conv1_bias)
        if save_intermediates:
            intermediates["conv1_pre_relu"] = x.clone()
        x = F.relu(x)
        if save_intermediates:
            intermediates["conv1_out"] = x.clone()

        # MaxPool1
        x = F.max_pool2d(x, 2, 2)
        if save_intermediates:
            intermediates["pool1_out"] = x.clone()

        # Conv2 + ReLU
        x = F.conv2d(x, self.conv2_weight, self.conv2_bias)
        if save_intermediates:
            intermediates["conv2_pre_relu"] = x.clone()
        x = F.relu(x)
        if save_intermediates:
            intermediates["conv2_out"] = x.clone()

        # MaxPool2
        x = F.max_pool2d(x, 2, 2)
        if save_intermediates:
            intermediates["pool2_out"] = x.clone()

        # Flatten
        x = x.view(x.size(0), -1)
        if save_intermediates:
            intermediates["flatten"] = x.clone()

        # FC1 + ReLU
        x = self.fc1(x)
        if save_intermediates:
            intermediates["fc1_pre_relu"] = x.clone()
        x = F.relu(x)
        if save_intermediates:
            intermediates["fc1_out"] = x.clone()

        # FC2 + ReLU
        x = self.fc2(x)
        if save_intermediates:
            intermediates["fc2_pre_relu"] = x.clone()
        x = F.relu(x)
        if save_intermediates:
            intermediates["fc2_out"] = x.clone()

        # FC3 (logits)
        x = self.fc3(x)
        if save_intermediates:
            intermediates["fc3_out"] = x.clone()

        if save_intermediates:
            return x, intermediates
        return x


def nchw_to_nhwc(tensor):
    """Convert [N, C, H, W] to [N, H, W, C]."""
    return tensor.permute(0, 2, 3, 1).contiguous()


def nhwc_to_nchw(tensor):
    """Convert [N, H, W, C] to [N, C, H, W]."""
    return tensor.permute(0, 3, 1, 2).contiguous()


def oihw_to_owhc(weight):
    """Convert conv weight [O, I, H, W] to [O, H, W, I] for NHWC."""
    return weight.permute(0, 2, 3, 1).contiguous()


def main():
    output_dir = Path("test_fixtures/lenet")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)

    # Create model
    model = LeNetNHWC()

    # Kaiming initialization
    nn.init.kaiming_uniform_(model.conv1_weight, mode='fan_in', nonlinearity='relu')
    nn.init.kaiming_uniform_(model.conv2_weight, mode='fan_in', nonlinearity='relu')
    nn.init.kaiming_uniform_(model.fc1.weight, mode='fan_in', nonlinearity='relu')
    nn.init.kaiming_uniform_(model.fc2.weight, mode='fan_in', nonlinearity='relu')
    nn.init.kaiming_uniform_(model.fc3.weight, mode='fan_in', nonlinearity='relu')

    # Create test input and targets
    batch_size = 4
    x = torch.randn(batch_size, 1, 28, 28)  # NCHW
    targets = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    print("Generating LeNet fixtures...")

    # Forward pass
    model.eval()
    logits, intermediates = model(x, save_intermediates=True)

    # Compute loss
    loss = F.cross_entropy(logits, targets)
    print(f"  Loss: {loss.item():.4f}")

    # Backward pass
    model.zero_grad()
    loss.backward()

    # Save weights in NHWC format
    print("  Saving weights...")
    save_f32_binary(output_dir / "conv1_weight.bin", oihw_to_owhc(model.conv1_weight.data))
    save_f32_binary(output_dir / "conv1_bias.bin", model.conv1_bias.data)
    save_f32_binary(output_dir / "conv2_weight.bin", oihw_to_owhc(model.conv2_weight.data))
    save_f32_binary(output_dir / "conv2_bias.bin", model.conv2_bias.data)
    save_f32_binary(output_dir / "fc1_weight.bin", model.fc1.weight.data)
    save_f32_binary(output_dir / "fc1_bias.bin", model.fc1.bias.data)
    save_f32_binary(output_dir / "fc2_weight.bin", model.fc2.weight.data)
    save_f32_binary(output_dir / "fc2_bias.bin", model.fc2.bias.data)
    save_f32_binary(output_dir / "fc3_weight.bin", model.fc3.weight.data)
    save_f32_binary(output_dir / "fc3_bias.bin", model.fc3.bias.data)

    # Save input (NHWC)
    print("  Saving input...")
    save_f32_binary(output_dir / "input.bin", nchw_to_nhwc(x))
    save_u32_binary(output_dir / "targets.bin", targets.tolist())

    # Save intermediates (NHWC where applicable)
    print("  Saving intermediates...")
    save_f32_binary(output_dir / "conv1_out.bin", nchw_to_nhwc(intermediates["conv1_out"]))
    save_f32_binary(output_dir / "pool1_out.bin", nchw_to_nhwc(intermediates["pool1_out"]))
    save_f32_binary(output_dir / "conv2_out.bin", nchw_to_nhwc(intermediates["conv2_out"]))
    save_f32_binary(output_dir / "pool2_out.bin", nchw_to_nhwc(intermediates["pool2_out"]))
    save_f32_binary(output_dir / "fc1_out.bin", intermediates["fc1_out"])
    save_f32_binary(output_dir / "fc2_out.bin", intermediates["fc2_out"])
    save_f32_binary(output_dir / "fc3_out.bin", intermediates["fc3_out"])

    # Save gradients
    print("  Saving gradients...")
    save_f32_binary(output_dir / "grad_conv1_weight.bin", oihw_to_owhc(model.conv1_weight.grad))
    save_f32_binary(output_dir / "grad_conv1_bias.bin", model.conv1_bias.grad)
    save_f32_binary(output_dir / "grad_conv2_weight.bin", oihw_to_owhc(model.conv2_weight.grad))
    save_f32_binary(output_dir / "grad_conv2_bias.bin", model.conv2_bias.grad)
    save_f32_binary(output_dir / "grad_fc1_weight.bin", model.fc1.weight.grad)
    save_f32_binary(output_dir / "grad_fc1_bias.bin", model.fc1.bias.grad)
    save_f32_binary(output_dir / "grad_fc2_weight.bin", model.fc2.weight.grad)
    save_f32_binary(output_dir / "grad_fc2_bias.bin", model.fc2.bias.grad)
    save_f32_binary(output_dir / "grad_fc3_weight.bin", model.fc3.weight.grad)
    save_f32_binary(output_dir / "grad_fc3_bias.bin", model.fc3.bias.grad)

    # Save metadata
    metadata = {
        "batch_size": batch_size,
        "loss": loss.item(),
        "shapes": {
            "input": [batch_size, 28, 28, 1],
            "conv1_weight": [6, 5, 5, 1],
            "conv1_out": [batch_size, 24, 24, 6],
            "pool1_out": [batch_size, 12, 12, 6],
            "conv2_weight": [16, 5, 5, 6],
            "conv2_out": [batch_size, 8, 8, 16],
            "pool2_out": [batch_size, 4, 4, 16],
            "fc1_weight": [120, 256],
            "fc1_out": [batch_size, 120],
            "fc2_weight": [84, 120],
            "fc2_out": [batch_size, 84],
            "fc3_weight": [10, 84],
            "fc3_out": [batch_size, 10],
        },
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nFixtures saved to {output_dir}/")
    print("Files:")
    for f in sorted(output_dir.glob("*")):
        size = f.stat().st_size
        print(f"  {f.name}: {size} bytes")


if __name__ == "__main__":
    main()
