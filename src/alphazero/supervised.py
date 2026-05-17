"""Supervised pre-training of AlphaZeroNet on a Stockfish-generated corpus.

Pipeline:
    PretrainConfig (loaded from TOML)
    ├─ load_corpus() — discover shards, split into train/val
    ├─ iter_batches() — stream batches from shards
    ├─ lr_schedule() — linear warmup + cosine decay
    ├─ pretrain_supervised() — main loop with early stopping
    └─ save checkpoint in Trainer-compatible format
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class PretrainConfig:
    """Configuration for supervised pre-training.

    Mirrors TrainingConfig style (frozen dataclass, TOML-loadable) but
    holds only the knobs needed for pre-training. AZ refinement uses
    TrainingConfig + chess-refine.toml separately.
    """
    n_blocks: int
    n_channels: int

    num_epochs: int = 3
    batch_size: int = 256
    peak_lr: float = 5e-4
    end_lr: float = 5e-5
    warmup_steps: int = 500
    weight_decay: float = 1e-4

    holdout_fraction: float = 0.05
    early_stopping_patience: int = 2

    corpus_dir: str = "data/chess_corpus"
    output_checkpoint: str = "pretrained.pt"
    log_dir: str = "runs_pretrain"

    seed: int = 42
    device: str = "auto"


def load_pretrain_config(path: Path | str) -> PretrainConfig:
    """Load PretrainConfig from a TOML file. Unknown keys raise ValueError."""
    data = tomllib.loads(Path(path).read_text())
    field_names = {f.name for f in fields(PretrainConfig)}
    unknown = set(data) - field_names
    if unknown:
        raise ValueError(f"Unknown pretrain config keys: {sorted(unknown)}")
    return PretrainConfig(**data)


import numpy as np
import torch
from typing import Iterator


def load_corpus(
    shard_dir: Path | str, holdout_fraction: float
) -> tuple[list[Path], list[Path]]:
    """Discover all shards in shard_dir and split into train/val.

    Returns (train_shards, val_shards) — both sorted by filename. The
    LAST holdout_fraction of shards (by sorted order) becomes val.
    Always reserves at least 1 shard for val.
    """
    shard_dir = Path(shard_dir)
    all_shards = sorted(shard_dir.glob("shard_*.npz"))
    if not all_shards:
        raise ValueError(f"No shards found in {shard_dir}")
    n_val = max(1, int(len(all_shards) * holdout_fraction))
    return all_shards[:-n_val], all_shards[-n_val:]


def iter_batches(
    shards: list[Path],
    batch_size: int,
    device: torch.device,
    shuffle: bool,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Yield (states, move_indices, outcomes) batches from a list of shards.

    Loads one shard at a time, optionally shuffles within the shard,
    then yields batches. The final batch may be smaller than batch_size.

    Returns tensors on `device` with dtypes:
        states: float32 (B, 20, 8, 8)
        move_indices: int64 (B,)
        outcomes: float32 (B,)
    """
    for shard_path in shards:
        data = np.load(shard_path)
        states = data["states"]
        move_indices = data["move_indices"]
        outcomes = data["outcomes"]
        n = states.shape[0]

        order = torch.randperm(n).numpy() if shuffle else np.arange(n)

        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            s = torch.from_numpy(states[idx]).to(torch.float32).to(device)
            m = torch.from_numpy(move_indices[idx]).to(torch.int64).to(device)
            z = torch.from_numpy(outcomes[idx]).to(torch.float32).to(device)
            yield s, m, z


import math


def lr_schedule(
    *,
    step: int,
    warmup_steps: int,
    total_steps: int,
    peak_lr: float,
    end_lr: float,
) -> float:
    """Linear warmup over warmup_steps, then cosine decay to end_lr at total_steps.

    During warmup (step < warmup_steps):
        lr = peak_lr * (step + 1) / warmup_steps

    After warmup (step >= warmup_steps):
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        lr = end_lr + 0.5 * (peak_lr - end_lr) * (1 + cos(pi * progress))
    """
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(progress, 1.0)
    return end_lr + 0.5 * (peak_lr - end_lr) * (1 + math.cos(math.pi * progress))


import torch.nn.functional as F

from .network import AlphaZeroNet


def pretrain_step(
    net: AlphaZeroNet,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    lr: float,
) -> tuple[float, float, float]:
    """One SGD step on (states, move_indices, outcomes). Returns scalar losses.

    Policy loss = cross-entropy(logits, move_indices) — the target is the
    one-hot of Stockfish's chosen move (passed as integer index).
    Value loss = MSE(predicted_value, outcomes) — outcomes are the game
    result from each position's mover POV.
    """
    states, move_indices, outcomes = batch
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    optimizer.zero_grad()
    logits, values = net(states)
    policy_loss = F.cross_entropy(logits, move_indices)
    value_loss = F.mse_loss(values, outcomes)
    total = policy_loss + value_loss
    total.backward()
    optimizer.step()
    return total.item(), policy_loss.item(), value_loss.item()


def compute_val_loss(
    net: AlphaZeroNet,
    val_shards: list[Path],
    batch_size: int,
    device: torch.device,
) -> float:
    """Compute average (policy + value) loss over val_shards without gradients."""
    net.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.inference_mode():
        for states, move_indices, outcomes in iter_batches(
            val_shards, batch_size=batch_size, device=device, shuffle=False
        ):
            logits, values = net(states)
            policy_loss = F.cross_entropy(logits, move_indices, reduction="sum")
            value_loss = F.mse_loss(values, outcomes, reduction="sum")
            total_loss += (policy_loss + value_loss).item()
            total_samples += states.shape[0]
    net.train()
    if total_samples == 0:
        return float("nan")
    return total_loss / total_samples
