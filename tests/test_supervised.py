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


def test_lr_schedule_warmup_ramps_linearly_from_zero():
    """During warmup, lr ramps linearly from ~0 to peak_lr."""
    from alphazero.supervised import lr_schedule

    lr_at_0 = lr_schedule(step=0, warmup_steps=100, total_steps=1000, peak_lr=1e-3, end_lr=1e-4)
    lr_at_50 = lr_schedule(step=50, warmup_steps=100, total_steps=1000, peak_lr=1e-3, end_lr=1e-4)
    lr_at_99 = lr_schedule(step=99, warmup_steps=100, total_steps=1000, peak_lr=1e-3, end_lr=1e-4)

    assert lr_at_0 == pytest.approx(1e-5, rel=0.01)
    assert lr_at_50 == pytest.approx(0.51 * 1e-3, rel=0.01)
    assert lr_at_99 == pytest.approx(1.0 * 1e-3, rel=0.01)


def test_lr_schedule_after_warmup_starts_at_peak():
    """At step == warmup_steps, lr equals peak_lr."""
    from alphazero.supervised import lr_schedule

    lr = lr_schedule(step=100, warmup_steps=100, total_steps=1000, peak_lr=1e-3, end_lr=1e-4)
    assert lr == pytest.approx(1e-3, rel=0.01)


def test_lr_schedule_at_total_steps_reaches_end_lr():
    """At step == total_steps, lr equals end_lr (cosine has fully decayed)."""
    from alphazero.supervised import lr_schedule

    lr = lr_schedule(step=1000, warmup_steps=100, total_steps=1000, peak_lr=1e-3, end_lr=1e-4)
    assert lr == pytest.approx(1e-4, rel=0.01)


def test_lr_schedule_monotonic_decay_after_warmup():
    """After warmup, lr only decreases."""
    from alphazero.supervised import lr_schedule

    lrs = [
        lr_schedule(step=s, warmup_steps=100, total_steps=1000, peak_lr=1e-3, end_lr=1e-4)
        for s in range(100, 1001, 50)
    ]
    for i in range(len(lrs) - 1):
        assert lrs[i] >= lrs[i + 1] - 1e-9


def test_pretrain_step_returns_losses_and_decreases_loss():
    """One pretrain step on a tiny net + tiny batch should produce a finite loss
    and reduce the loss when run twice on the same batch (overfitting check)."""
    from alphazero.network import AlphaZeroNet
    from alphazero.supervised import pretrain_step

    net = AlphaZeroNet(input_shape=(20, 8, 8), action_size=4672, n_blocks=1, n_channels=8)
    net.train()
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)

    states = torch.zeros((4, 20, 8, 8), dtype=torch.float32)
    moves = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    zs = torch.tensor([1.0, -1.0, 0.0, 1.0], dtype=torch.float32)

    total_1, policy_1, value_1 = pretrain_step(net, (states, moves, zs), optimizer, lr=1e-3)
    total_2, policy_2, value_2 = pretrain_step(net, (states, moves, zs), optimizer, lr=1e-3)

    assert all(math.isfinite(x) for x in (total_1, policy_1, value_1, total_2, policy_2, value_2))
    assert total_2 < total_1


def test_compute_val_loss_runs_without_grad(tmp_path):
    """compute_val_loss should evaluate a model on a corpus without producing gradients."""
    from alphazero.network import AlphaZeroNet
    from alphazero.supervised import compute_val_loss

    _write_fake_shard(tmp_path / "shard_w0_s0000.npz", 64, seed=0)
    shards = [tmp_path / "shard_w0_s0000.npz"]

    net = AlphaZeroNet(input_shape=(20, 8, 8), action_size=4672, n_blocks=1, n_channels=8)
    net.eval()

    val_loss = compute_val_loss(net, shards, batch_size=16, device=torch.device("cpu"))
    assert math.isfinite(val_loss)
    assert val_loss > 0


def test_pretrain_supervised_writes_checkpoint(tmp_path):
    """End-to-end on a tiny corpus: pretrain runs and writes a checkpoint."""
    from alphazero.supervised import PretrainConfig, pretrain_supervised

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    for i in range(3):
        _write_fake_shard(corpus_dir / f"shard_w0_s{i:04d}.npz", 64, seed=i)

    ckpt_path = tmp_path / "test_pretrained.pt"

    cfg = PretrainConfig(
        n_blocks=1,
        n_channels=8,
        num_epochs=1,
        batch_size=32,
        warmup_steps=2,
        holdout_fraction=0.34,
        early_stopping_patience=10,
        corpus_dir=str(corpus_dir),
        output_checkpoint=str(ckpt_path),
        log_dir=str(tmp_path / "runs"),
        device="cpu",
    )

    pretrain_supervised(cfg)

    assert ckpt_path.exists()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert set(ckpt.keys()) >= {"iteration", "config", "best_net", "candidate_net", "optimizer"}
    assert ckpt["iteration"] == 0


def test_pretrain_supervised_early_stops_when_val_loss_diverges(tmp_path, monkeypatch):
    """If val_loss never improves, early stopping triggers."""
    from alphazero.supervised import PretrainConfig, pretrain_supervised
    import alphazero.supervised as sup

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    for i in range(3):
        _write_fake_shard(corpus_dir / f"shard_w0_s{i:04d}.npz", 64, seed=i)

    counter = [0.5]
    def fake_val_loss(*args, **kwargs):
        counter[0] += 1.0
        return counter[0]
    monkeypatch.setattr(sup, "compute_val_loss", fake_val_loss)

    cfg = PretrainConfig(
        n_blocks=1,
        n_channels=8,
        num_epochs=10,
        batch_size=32,
        warmup_steps=2,
        holdout_fraction=0.34,
        early_stopping_patience=2,
        corpus_dir=str(corpus_dir),
        output_checkpoint=str(tmp_path / "p.pt"),
        log_dir=str(tmp_path / "runs"),
        device="cpu",
    )
    pretrain_supervised(cfg)

    # First call sets best (1.5), then 2 worsening calls (2.5, 3.5) trip patience=2
    assert counter[0] <= 4.0


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
