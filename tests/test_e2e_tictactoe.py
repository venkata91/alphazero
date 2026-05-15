"""End-to-end test: train an AlphaZero agent on TTT and verify it never
loses to a perfect minimax solver.

This is the kill criterion for Sub-project 1. Marked slow because a full
training run takes ~15–45 minutes on M4 Pro. Run with:
    pytest -m slow tests/test_e2e_tictactoe.py -v -s
"""
from __future__ import annotations

import pytest

from alphazero.arena import play_match
from alphazero.config import TrainingConfig
from alphazero.games.tictactoe import TicTacToe
from alphazero.solvers.tictactoe_solver import solve_tictactoe_action
from alphazero.trainer import Trainer


@pytest.mark.slow
def test_e2e_tictactoe_reaches_zero_losses_vs_perfect_solver(tmp_path, monkeypatch):
    """Train, then play 200 games vs the perfect TTT minimax solver.
    Pass criterion: 0 losses (TTT is a draw with perfect play)."""
    monkeypatch.chdir(tmp_path)

    config = TrainingConfig(
        n_blocks=4,
        n_channels=32,
        num_simulations=50,
        games_per_iteration=100,
        training_steps_per_iteration=500,
        num_iterations=30,
        arena_interval=5,
        eval_interval=5,
        eval_games=200,
        seed=42,
        device="cpu",
    )

    trainer = Trainer(TicTacToe(), config)
    trainer.run()

    agent = trainer._make_argmax_mcts_agent(trainer.best_net)
    def solver_agent(game, state):
        return solve_tictactoe_action(state)

    result = play_match(TicTacToe(), agent, solver_agent, num_games=200)

    print(
        f"\nE2E result: wins={result.wins_a} draws={result.draws} losses={result.losses_a}"
    )
    assert result.losses_a == 0, (
        f"Agent lost {result.losses_a} games to perfect play. "
        "Increase num_iterations or num_simulations."
    )
