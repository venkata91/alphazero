"""Training configuration: a single frozen dataclass loaded from TOML.

Saving the exact config alongside each checkpoint requires the config be
serializable (TOML round-trip) and stable across runs (frozen dataclass).
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    # Network
    n_blocks: int = 4
    n_channels: int = 32

    # MCTS
    num_simulations: int = 50
    c_puct: float = 1.5
    dirichlet_alpha: float = 1.0
    dirichlet_weight: float = 0.25

    # Self-play
    games_per_iteration: int = 100
    temperature_threshold: int = 6

    # Training
    training_steps_per_iteration: int = 500
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # Replay buffer
    replay_buffer_capacity: int = 50_000
    min_buffer_size: int = 5_000

    # Eval
    eval_interval: int = 5
    eval_games: int = 200

    # Schedule
    num_iterations: int = 50

    # NEW: parallel self-play (scale-invariant knobs)
    num_workers: int = 1                 # 1 = serial fallback (existing behavior)
    inference_batch_size: int = 64       # NN-server batches up to N states per forward
    heartbeat_timeout_s: float = 300.0   # worker considered hung after this silence
    heartbeat_interval_s: float = 5.0    # worker intra-search heartbeat cadence
    progress_log_interval_s: float = 60.0  # parent aggregate progress log cadence

    # Misc
    seed: int = 42
    device: str = "auto"            # "auto" | "cpu" | "mps" | "cuda"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "runs"

    # Checkpoint disk-usage:
    # When True (default) each per-iteration iter_NNNN.pt stores only the
    # eval/fork-required tensors (iteration, config, best_net) — roughly 1/4
    # the size for chess. The full state (candidate_net + optimizer) needed
    # for a true mid-training resume is written separately to latest.pt.
    # Set to False to restore the legacy behavior where every iter_NNNN.pt
    # is a full checkpoint.
    lean_checkpoints: bool = True


def load_config(path: Path | str) -> TrainingConfig:
    """Load TrainingConfig from a TOML file.

    Unknown keys raise ValueError to catch typos early.
    """
    data = tomllib.loads(Path(path).read_text())
    field_names = {f.name for f in fields(TrainingConfig)}
    unknown = set(data) - field_names
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    return TrainingConfig(**data)
