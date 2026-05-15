import numpy as np
import pytest

from alphazero.games.tictactoe import TicTacToe
from alphazero.solvers.tictactoe_solver import (
    solve_tictactoe_value,
    solve_tictactoe_action,
)


@pytest.fixture
def ttt() -> TicTacToe:
    return TicTacToe()


def test_initial_state_is_a_draw_with_perfect_play(ttt: TicTacToe):
    s = ttt.initial_state()
    assert solve_tictactoe_value(s) == 0


def test_immediate_winning_move_returns_plus_one(ttt: TicTacToe):
    s = np.array([
        [1, 1, 0],
        [-1, -1, 0],
        [0, 0, 0],
    ], dtype=np.int8)
    assert solve_tictactoe_value(s) == 1


def test_immediate_losing_position_returns_minus_one(ttt: TicTacToe):
    s = np.array([
        [-1, -1, 0],
        [0, 0, 0],
        [0, 0, 0],
    ], dtype=np.int8)
    assert solve_tictactoe_value(s) == -1


def test_solver_action_completes_winning_move(ttt: TicTacToe):
    s = np.array([
        [1, 1, 0],
        [-1, -1, 0],
        [0, 0, 0],
    ], dtype=np.int8)
    assert solve_tictactoe_action(s) == 2


def test_solver_action_blocks_opponent_win(ttt: TicTacToe):
    s = np.array([
        [-1, -1, 0],
        [0, 0, 0],
        [0, 0, 0],
    ], dtype=np.int8)
    assert solve_tictactoe_action(s) == 2
