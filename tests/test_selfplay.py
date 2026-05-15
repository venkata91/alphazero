from __future__ import annotations

import numpy as np
import pytest

from alphazero.games.tictactoe import TicTacToe
from alphazero.selfplay import run_one_game


def uniform_eval_fn(encoded: np.ndarray) -> tuple[np.ndarray, float]:
    return np.full(9, 1/9, dtype=np.float32), 0.0


def test_run_one_game_terminates():
    examples = run_one_game(
        TicTacToe(), uniform_eval_fn,
        num_simulations=5, temperature_threshold=6, augment=False,
    )
    assert 1 <= len(examples) <= 9


def test_run_one_game_emits_tuples_with_correct_shapes():
    examples = run_one_game(
        TicTacToe(), uniform_eval_fn,
        num_simulations=5, temperature_threshold=6, augment=False,
    )
    for s, pi, z in examples:
        assert s.shape == (3, 3, 3)
        assert s.dtype == np.float32
        assert pi.shape == (9,)
        assert pi.dtype == np.float32
        np.testing.assert_allclose(pi.sum(), 1.0, rtol=1e-5)
        assert -1.0 <= z <= 1.0


def test_z_assignment_x_wins_on_move_5():
    """Construct a 5-move X-wins game by hand, check z for every ply."""
    from alphazero.selfplay import _assign_z

    ttt = TicTacToe()
    history = [
        (np.zeros((3, 3, 3), dtype=np.float32), np.full(9, 1/9, dtype=np.float32), 1),
        (np.zeros((3, 3, 3), dtype=np.float32), np.full(9, 1/9, dtype=np.float32), -1),
        (np.zeros((3, 3, 3), dtype=np.float32), np.full(9, 1/9, dtype=np.float32), 1),
        (np.zeros((3, 3, 3), dtype=np.float32), np.full(9, 1/9, dtype=np.float32), -1),
        (np.zeros((3, 3, 3), dtype=np.float32), np.full(9, 1/9, dtype=np.float32), 1),
    ]
    final_state = np.array([[1, 1, 1], [-1, -1, 0], [0, 0, 0]], dtype=np.int8)
    final_value = -1.0

    zs = _assign_z(history, final_value, final_state, ttt)
    assert zs == [1.0, -1.0, 1.0, -1.0, 1.0]


def test_z_assignment_o_wins_on_move_6():
    from alphazero.selfplay import _assign_z

    ttt = TicTacToe()
    history = [
        (np.zeros((3, 3, 3), dtype=np.float32), np.full(9, 1/9, dtype=np.float32), 1),
        (np.zeros((3, 3, 3), dtype=np.float32), np.full(9, 1/9, dtype=np.float32), -1),
        (np.zeros((3, 3, 3), dtype=np.float32), np.full(9, 1/9, dtype=np.float32), 1),
        (np.zeros((3, 3, 3), dtype=np.float32), np.full(9, 1/9, dtype=np.float32), -1),
        (np.zeros((3, 3, 3), dtype=np.float32), np.full(9, 1/9, dtype=np.float32), 1),
        (np.zeros((3, 3, 3), dtype=np.float32), np.full(9, 1/9, dtype=np.float32), -1),
    ]
    final_state = np.array([[-1, -1, -1], [1, 1, 0], [1, 0, 0]], dtype=np.int8)
    final_value = -1.0

    zs = _assign_z(history, final_value, final_state, ttt)
    assert zs == [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]


def test_z_assignment_draw_assigns_zero_to_every_ply():
    from alphazero.selfplay import _assign_z

    ttt = TicTacToe()
    history = [
        (np.zeros((3, 3, 3), dtype=np.float32), np.full(9, 1/9, dtype=np.float32), p)
        for p in [1, -1, 1, -1, 1, -1, 1, -1, 1]
    ]
    final_state = np.array([[1, -1, 1], [1, -1, -1], [-1, 1, 1]], dtype=np.int8)
    final_value = 0.0

    zs = _assign_z(history, final_value, final_state, ttt)
    assert zs == [0.0] * 9
