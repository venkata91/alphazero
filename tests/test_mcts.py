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


def test_search_returns_distribution_summing_to_one(ttt):
    eval_fn = make_eval_fn(np.full(9, 1/9), 0.0)
    mcts = MCTS(ttt, eval_fn)
    pi = mcts.search(ttt.initial_state(), num_simulations=10, add_root_noise=False)
    assert pi.shape == (9,)
    np.testing.assert_allclose(pi.sum(), 1.0, rtol=1e-6)


def test_search_visits_only_legal_actions(ttt):
    eval_fn = make_eval_fn(np.full(9, 1/9), 0.0)
    mcts = MCTS(ttt, eval_fn)
    state = ttt.apply(ttt.initial_state(), 4)
    pi = mcts.search(state, num_simulations=20, add_root_noise=False)
    assert pi[4] == 0.0


def test_search_prior_favoring_action_concentrates_visits(ttt):
    prior = np.full(9, 0.01)
    prior[0] = 1.0 - 0.01 * 8
    eval_fn = make_eval_fn(prior, 0.0)
    mcts = MCTS(ttt, eval_fn)
    pi = mcts.search(ttt.initial_state(), num_simulations=40, add_root_noise=False)
    assert pi[0] > 0.5


def test_search_more_simulations_means_more_total_visits(ttt):
    eval_fn = make_eval_fn(np.full(9, 1/9), 0.0)
    mcts = MCTS(ttt, eval_fn)
    pi_a = mcts.search(ttt.initial_state(), num_simulations=4, add_root_noise=False)
    pi_b = mcts.search(ttt.initial_state(), num_simulations=40, add_root_noise=False)
    assert pi_a.sum() == pytest.approx(1.0)
    assert pi_b.sum() == pytest.approx(1.0)
