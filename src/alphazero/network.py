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
