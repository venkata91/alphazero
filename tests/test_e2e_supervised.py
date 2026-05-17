"""End-to-end test for the SP3b pipeline: corpus gen → pretrain → 1 AZ refinement iter.

This is a SLOW test — actually spawns Stockfish, generates a tiny corpus, pretrains,
runs one AZ iteration. Marked slow so it doesn't run in default pytest.

Run with:
    pytest -m slow tests/test_e2e_supervised.py -v -s
"""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.slow
def test_e2e_corpus_pretrain_refine_pipeline(tmp_path, monkeypatch):
    """Mini run of the full SP3b pipeline. Wall-clock: ~3 minutes."""
    from alphazero.config import TrainingConfig
    from alphazero.corpus import generate_corpus
    from alphazero.games.chess_game import Chess
    from alphazero.supervised import PretrainConfig, pretrain_supervised
    from alphazero.trainer import Trainer

    monkeypatch.chdir(tmp_path)

    corpus_dir = tmp_path / "corpus"

    # Phase 1: tiny corpus (4 games, 2 workers)
    generate_corpus(
        target_games=4,
        num_workers=2,
        output_dir=corpus_dir,
        time_per_move=0.01,
        seed=42,
    )
    shards = sorted(corpus_dir.glob("shard_*.npz"))
    assert len(shards) >= 2

    # Phase 2: pretrain a tiny network
    pretrained_path = tmp_path / "pretrained.pt"
    pre_cfg = PretrainConfig(
        n_blocks=1,
        n_channels=8,
        num_epochs=1,
        batch_size=16,
        warmup_steps=2,
        holdout_fraction=0.34,
        early_stopping_patience=10,
        corpus_dir=str(corpus_dir),
        output_checkpoint=str(pretrained_path),
        log_dir=str(tmp_path / "runs_pre"),
        device="cpu",
    )
    pretrain_supervised(pre_cfg)
    assert pretrained_path.exists()

    # Phase 3: 1 iteration of AZ refinement (single-worker, serial path)
    train_cfg = TrainingConfig(
        n_blocks=1,
        n_channels=8,
        num_iterations=1,
        games_per_iteration=2,
        num_simulations=3,
        training_steps_per_iteration=2,
        batch_size=4,
        min_buffer_size=4,
        replay_buffer_capacity=100,
        eval_interval=999,
        device="cpu",
        checkpoint_dir=str(tmp_path / "ckpts"),
        log_dir=str(tmp_path / "runs_refine"),
    )
    trainer = Trainer(Chess(), train_cfg)
    trainer.load_from_checkpoint(pretrained_path)
    trainer.run(verbose=False)

    assert (tmp_path / "ckpts" / "iter_0001.pt").exists()
