"""Device resolution shared by Trainer and supervised pre-training.

A single source of truth so `"auto"` resolves to the same device for both
self-play training and supervised pre-training. Otherwise a user can
pretrain on MPS, refine on CUDA, and have `--resume-from` silently
change devices mid-pipeline.

Order: CUDA → MPS → CPU.
"""
from __future__ import annotations

import torch


def resolve_device(name: str) -> torch.device:
    """Resolve a device-name string (e.g., 'auto', 'cpu', 'mps', 'cuda') to a torch.device.

    'auto' picks the best available accelerator in this order:
        1. CUDA (if any device is visible)
        2. MPS (Apple Silicon)
        3. CPU
    """
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)
