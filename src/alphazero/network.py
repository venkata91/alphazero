"""AlphaZero neural network: ResNet backbone with policy + value heads.

Generic over input shape and action space: provided by the Game class.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """One residual block: y = ReLU(x + BN(conv(ReLU(BN(conv(x)))))).

    The skip connection (the "+ x") is what makes deep nets trainable.
    Without it, gradients vanish through deep stacks of layers.
    """

    def __init__(self, n_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.conv2 = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(n_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class AlphaZeroNet(nn.Module):
    """ResNet backbone with two heads: policy (action logits) and value (tanh scalar).

    Architecture:
        input conv (C_in → n_channels)
          → BN → ReLU
          → n_blocks × ResidualBlock(n_channels)
          → policy_head (1x1 conv → flatten → linear → action_size logits)
          → value_head  (1x1 conv → flatten → linear → relu → linear → tanh)
    """

    def __init__(
        self,
        input_shape: tuple[int, ...],
        action_size: int,
        n_blocks: int = 4,
        n_channels: int = 32,
    ):
        super().__init__()
        in_channels, height, width = input_shape

        # Input projection
        self.input_conv = nn.Conv2d(in_channels, n_channels, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(n_channels)

        # Backbone
        self.blocks = nn.Sequential(*[ResidualBlock(n_channels) for _ in range(n_blocks)])

        # Policy head
        self.policy_conv = nn.Conv2d(n_channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * height * width, action_size)

        # Value head
        self.value_conv = nn.Conv2d(n_channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(height * width, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.input_bn(self.input_conv(x)))
        h = self.blocks(h)

        p = F.relu(self.policy_bn(self.policy_conv(h)))
        p = p.flatten(start_dim=1)
        policy_logits = self.policy_fc(p)

        v = F.relu(self.value_bn(self.value_conv(h)))
        v = v.flatten(start_dim=1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        value = v.squeeze(-1)

        return policy_logits, value
