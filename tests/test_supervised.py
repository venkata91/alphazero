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


def test_load_corpus_splits_shards_into_train_and_val(tmp_path):
    """load_corpus discovers all shards and reserves holdout_fraction for val."""
    from alphazero.supervised import load_corpus

    for i in range(20):
        _write_fake_shard(tmp_path / f"shard_w0_s{i:04d}.npz", 100, seed=i)

    train, val = load_corpus(tmp_path, holdout_fraction=0.1)
    assert len(train) == 18
    assert len(val) == 2
    assert all(t.name < v.name for t in train for v in val)


def test_load_corpus_minimum_one_val_shard(tmp_path):
    """Even with a tiny corpus, at least 1 shard is held out."""
    from alphazero.supervised import load_corpus

    for i in range(3):
        _write_fake_shard(tmp_path / f"shard_w0_s{i:04d}.npz", 100, seed=i)

    train, val = load_corpus(tmp_path, holdout_fraction=0.01)
    assert len(val) >= 1


def test_iter_batches_yields_correct_shape_and_dtype(tmp_path):
    """iter_batches reads shards and yields (states, moves, z) tensors."""
    from alphazero.supervised import iter_batches

    _write_fake_shard(tmp_path / "shard_w0_s0000.npz", 100, seed=0)
    _write_fake_shard(tmp_path / "shard_w0_s0001.npz", 100, seed=1)
    shards = sorted(tmp_path.glob("*.npz"))

    batches = list(iter_batches(shards, batch_size=32, device=torch.device("cpu"), shuffle=False))
    # iter_batches works PER SHARD: 100/32 = 3 full + 1 partial per shard, ×2 shards = 8 batches
    assert len(batches) == 8

    # Check the 6 full batches (3 per shard × 2 shards)
    full_batches = [b for b in batches if b[0].shape[0] == 32]
    assert len(full_batches) == 6
    for states, moves, zs in full_batches:
        assert states.shape == (32, 20, 8, 8)
        assert states.dtype == torch.float32
        assert moves.shape == (32,)
        assert moves.dtype == torch.int64
        assert zs.shape == (32,)
        assert zs.dtype == torch.float32

    # The 2 partial batches (last in each shard, size 100 - 96 = 4)
    partial_batches = [b for b in batches if b[0].shape[0] != 32]
    assert len(partial_batches) == 2
    for states, _, _ in partial_batches:
        assert states.shape == (4, 20, 8, 8)


def test_iter_batches_shuffles_when_requested(tmp_path):
    """shuffle=True produces a different order than shuffle=False."""
    from alphazero.supervised import iter_batches

    _write_fake_shard(tmp_path / "shard_w0_s0000.npz", 100, seed=0)
    shards = sorted(tmp_path.glob("*.npz"))

    torch.manual_seed(42)
    no_shuffle = next(iter(iter_batches(shards, batch_size=100, device=torch.device("cpu"), shuffle=False)))
    torch.manual_seed(42)
    yes_shuffle = next(iter(iter_batches(shards, batch_size=100, device=torch.device("cpu"), shuffle=True)))

    assert not torch.equal(no_shuffle[1], yes_shuffle[1])


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
