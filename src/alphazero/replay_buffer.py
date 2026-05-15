"""Fixed-capacity FIFO replay buffer.

Holds (state, policy_target, value_target) tuples from recent self-play games.
Random uniform sampling — no prioritization (per AlphaZero paper).
"""
from __future__ import annotations

from collections import deque
from typing import Sequence

import numpy as np
import torch


Tuple3 = tuple[np.ndarray, np.ndarray, float]


class ReplayBuffer:
    """FIFO queue of (state, policy, value) tuples.

    Sampling returns PyTorch tensors batched on the first dim, ready to feed
    directly into the network forward pass.
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self._buf: deque[Tuple3] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._buf)

    def add(self, examples: Sequence[Tuple3]) -> None:
        self._buf.extend(examples)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if batch_size > len(self._buf):
            raise ValueError(
                f"Cannot sample {batch_size} from buffer of size {len(self._buf)}"
            )
        indices = np.random.choice(len(self._buf), size=batch_size, replace=False)
        batch = [self._buf[int(i)] for i in indices]
        states = np.stack([b[0] for b in batch])
        policies = np.stack([b[1] for b in batch])
        values = np.array([b[2] for b in batch], dtype=np.float32)
        return (
            torch.from_numpy(states).float(),
            torch.from_numpy(policies).float(),
            torch.from_numpy(values).float(),
        )
