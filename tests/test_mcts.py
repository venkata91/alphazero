"""MCTS tests. We use a mock eval_fn so behavior is fully deterministic."""
from __future__ import annotations

import numpy as np
import pytest

from alphazero.games.tictactoe import TicTacToe
from alphazero.mcts import MCTS, Node


def make_eval_fn(prior: np.ndarray, value: float):
    """eval_fn that returns the same (prior, value) for any state."""
    def fn(state: np.ndarray) -> tuple[np.ndarray, float]:
        return prior.astype(np.float32), float(value)
    return fn


@pytest.fixture
def ttt() -> TicTacToe:
    return TicTacToe()


def test_node_default_values():
    n = Node(prior=0.3)
    assert n.prior == 0.3
    assert n.visit_count == 0
    assert n.value_sum == 0.0
    assert n.children == {}
    assert n.is_expanded is False
    assert n.Q == 0.0


def test_node_Q_after_visits():
    n = Node(prior=0.1)
    n.visit_count = 3
    n.value_sum = 1.5
    assert n.Q == pytest.approx(0.5)
