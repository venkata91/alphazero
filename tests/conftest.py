"""Shared pytest fixtures. Tiny config so unit tests stay fast."""
from __future__ import annotations

import pytest

from alphazero.config import TrainingConfig


@pytest.fixture
def tiny_config() -> TrainingConfig:
    """A small TrainingConfig used by unit tests that need one.

    Trainer / network / MCTS unit tests use this so they stay sub-second.
    The end-to-end test uses the full TTT config from configs/tictactoe.toml.
    """
    return TrainingConfig(
        n_blocks=1,
        n_channels=8,
        num_simulations=10,
        c_puct=1.5,
        dirichlet_alpha=1.0,
        dirichlet_weight=0.25,
        games_per_iteration=4,
        temperature_threshold=6,
        training_steps_per_iteration=5,
        batch_size=8,
        learning_rate=1e-3,
        weight_decay=1e-4,
        replay_buffer_capacity=200,
        min_buffer_size=20,
        arena_interval=1,
        arena_games=4,
        arena_threshold=0.55,
        eval_interval=1,
        eval_games=8,
        num_iterations=2,
        seed=42,
        device="cpu",
        checkpoint_dir="checkpoints_test",
        log_dir="runs_test",
    )
