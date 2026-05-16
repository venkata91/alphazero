"""End-to-end test: train an AlphaZero agent on Connect 4 and verify it beats
the minimax-depth-8 opponent in ≥80% of 200 games.

This is the kill criterion for Sub-project 2. Marked slow — a full training
run takes ~3-6 hours on M4 Pro MPS / ~12-20 hours on CPU / ~1-2 hours on
Colab T4 GPU.

Run with:
    pytest -m slow tests/test_e2e_connect4.py -v -s
"""
from __future__ import annotations

from pathlib import Path

import pytest

from alphazero.arena import play_match
from alphazero.config import load_config
from alphazero.games.connect4 import Connect4
from alphazero.opponents.connect4_minimax import Connect4MinimaxOpponent
from alphazero.trainer import Trainer

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.slow
def test_e2e_connect4_beats_minimax_depth_8(tmp_path, monkeypatch):
    """Train, then play 200 games vs the minimax-depth-8 opponent.
    Pass criterion: win_rate >= 0.80 (alternating colors)."""
    monkeypatch.chdir(tmp_path)

    config = load_config(REPO_ROOT / "configs/connect4.toml")
    opponent = Connect4MinimaxOpponent(depth=8)

    trainer = Trainer(Connect4(), config)
    trainer.run(eval_opponent=opponent)

    agent = trainer._make_argmax_mcts_agent(trainer.best_net)
    result = play_match(Connect4(), agent, opponent, num_games=200)

    print(
        f"\nFinal: wins={result.wins_a} draws={result.draws} "
        f"losses={result.losses_a} win_rate={result.win_rate:.2%}"
    )
    assert result.win_rate >= 0.80, (
        f"Agent win rate {result.win_rate:.2%} below 0.80 target. "
        "Increase num_iterations or num_simulations in configs/connect4.toml."
    )
