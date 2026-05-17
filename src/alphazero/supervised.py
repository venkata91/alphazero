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
