"""Tests for supervised pre-training."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch


def _write_fake_shard(path: Path, n_positions: int, seed: int):
    """Helper: write a fake shard with random data of the right shapes/dtypes."""
    rng = np.random.default_rng(seed)
    np.savez_compressed(
        path,
        states=rng.integers(0, 2, size=(n_positions, 20, 8, 8), dtype=np.int8),
        move_indices=rng.integers(0, 4672, size=(n_positions,), dtype=np.int32),
        outcomes=rng.choice([-1, 0, 1], size=(n_positions,)).astype(np.int8),
    )


def test_pretrain_config_defaults():
    """PretrainConfig has expected default values."""
    from alphazero.supervised import PretrainConfig

    cfg = PretrainConfig(n_blocks=4, n_channels=8)
    assert cfg.n_blocks == 4
    assert cfg.n_channels == 8
    assert cfg.num_epochs == 3
    assert cfg.batch_size == 256
    assert cfg.peak_lr == 5e-4
    assert cfg.end_lr == 5e-5
    assert cfg.warmup_steps == 500
    assert cfg.weight_decay == 1e-4
    assert cfg.holdout_fraction == 0.05
    assert cfg.early_stopping_patience == 2
    assert cfg.corpus_dir == "data/chess_corpus"
    assert cfg.output_checkpoint == "pretrained.pt"
    assert cfg.log_dir == "runs_pretrain"
    assert cfg.seed == 42
    assert cfg.device == "auto"


def test_pretrain_config_loads_from_toml(tmp_path):
    """load_pretrain_config reads a TOML file and constructs a PretrainConfig."""
    from alphazero.supervised import load_pretrain_config

    toml_path = tmp_path / "test.toml"
    toml_path.write_text("""
n_blocks = 10
n_channels = 64
num_epochs = 5
batch_size = 128
peak_lr = 1e-3
""")
    cfg = load_pretrain_config(toml_path)
    assert cfg.n_blocks == 10
    assert cfg.n_channels == 64
    assert cfg.num_epochs == 5
    assert cfg.batch_size == 128
    assert cfg.peak_lr == 1e-3
    assert cfg.warmup_steps == 500


def test_pretrain_config_rejects_unknown_keys(tmp_path):
    """Unknown TOML keys raise ValueError (typo protection)."""
    from alphazero.supervised import load_pretrain_config

    toml_path = tmp_path / "bad.toml"
    toml_path.write_text("""
n_blocks = 4
n_channels = 8
typo_field = 999
""")
    with pytest.raises(ValueError, match="Unknown"):
        load_pretrain_config(toml_path)
