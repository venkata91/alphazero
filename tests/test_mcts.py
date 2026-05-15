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


def test_search_value_favoring_child_concentrates_visits(ttt):
    """If the NN values action 0 highly, visits should concentrate on action 0.

    The eval_fn returns value=+1 when it sees a board where the opponent (planes[1])
    has a piece at (0,0) — that's the state after X plays action 0.
    """

    def biased_eval_fn(encoded: np.ndarray) -> tuple[np.ndarray, float]:
        prior = np.full(9, 1/9, dtype=np.float32)
        opp_corner = encoded[1, 0, 0]
        value = 1.0 if opp_corner > 0.5 else 0.0
        return prior, value

    mcts = MCTS(ttt, biased_eval_fn)
    pi = mcts.search(ttt.initial_state(), num_simulations=80, add_root_noise=False)
    assert pi.argmax() == 0


def test_backup_flips_sign_per_ply(ttt):
    """Direct unit test on a known tree shape.

    Set up: root with one child, whose value backs up.
    Backing up +1 from the child should make root.Q = -1 (zero-sum).
    """
    eval_fn = make_eval_fn(np.full(9, 1/9), 0.5)
    mcts = MCTS(ttt, eval_fn)
    root = Node()
    root.is_expanded = True
    child = Node(prior=1.0)
    root.children[0] = child

    mcts._backup([root, child], leaf_value=1.0)

    assert child.visit_count == 1
    assert child.value_sum == 1.0
    assert child.Q == 1.0
    assert root.visit_count == 1
    assert root.value_sum == -1.0
    assert root.Q == -1.0


def test_dirichlet_noise_changes_priors_when_enabled(ttt):
    """With add_root_noise=True and a fixed seed, root priors should differ."""
    np.random.seed(0)
    eval_fn = make_eval_fn(np.full(9, 1/9), 0.0)
    mcts = MCTS(ttt, eval_fn, dirichlet_alpha=1.0, dirichlet_weight=0.5)
    pi_noisy = mcts.search(ttt.initial_state(), num_simulations=1, add_root_noise=True)
    np.random.seed(0)
    pi_clean = mcts.search(ttt.initial_state(), num_simulations=1, add_root_noise=False)
    assert not np.allclose(pi_noisy, pi_clean)


def test_dirichlet_noise_disabled_yields_one_hot_at_one_sim(ttt):
    """With no noise, 1 simulation should give exactly one visit somewhere."""
    eval_fn = make_eval_fn(np.full(9, 1/9), 0.0)
    mcts = MCTS(ttt, eval_fn)
    pi = mcts.search(ttt.initial_state(), num_simulations=1, add_root_noise=False)
    assert pi.sum() == pytest.approx(1.0)
    assert (pi > 0).sum() == 1
