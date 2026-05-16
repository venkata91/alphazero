"""End-to-end test: train + ≥50% win rate vs Stockfish ELO=1500 over 100 games.

This is the kill criterion for Sub-project 3. Marked slow — a full
training run takes 3-8 hours on M4 Pro MPS / Colab L4 / Lambda A6000.

Run with:
    pytest -m slow tests/test_e2e_chess.py -v -s
"""
from __future__ import annotations

from pathlib import Path

import pytest

from alphazero.arena import play_match
from alphazero.config import load_config
from alphazero.games.chess_game import Chess
from alphazero.opponents.stockfish import StockfishOpponent
from alphazero.trainer import Trainer

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.slow
def test_e2e_chess_beats_stockfish_1500(tmp_path, monkeypatch):
    """Train an AlphaZero chess agent. Play 100 games vs Stockfish ELO=1500.
    Pass criterion: win_rate >= 0.50 (alternating colors)."""
    monkeypatch.chdir(tmp_path)

    config = load_config(REPO_ROOT / "configs/chess.toml")
    opponent = StockfishOpponent(elo=1500, time_per_move=0.5)

    trainer = Trainer(Chess(), config)
    trainer.run(eval_opponent=opponent)

    agent = trainer._make_argmax_mcts_agent(trainer.best_net)
    result = play_match(Chess(), agent, opponent, num_games=100)
    opponent.close()

    print(
        f"\nFinal: wins={result.wins_a} draws={result.draws} "
        f"losses={result.losses_a} win_rate={result.win_rate:.2%}"
    )
    assert result.win_rate >= 0.50, (
        f"Agent win rate {result.win_rate:.2%} below 0.50 target. "
        "Increase num_iterations or num_simulations in configs/chess.toml."
    )
