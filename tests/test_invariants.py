"""Named regression tests for the six critical invariants of AlphaZero.

These each guard a "silent failure" mode — training that appears to be
running but the agent never improves (or improves and then collapses).
"""
from __future__ import annotations

import copy
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from alphazero.games.tictactoe import TicTacToe
from alphazero.mcts import MCTS, Node
from alphazero.network import AlphaZeroNet
from alphazero.replay_buffer import ReplayBuffer
from alphazero.selfplay import run_one_game
from alphazero.trainer import Trainer


def test_invariant_1_selfplay_uses_best_net_not_candidate(tiny_config):
    """Wrap both nets in counting wrappers; run one self-play iteration.
    Assert: best_net.forward was called, candidate_net.forward was NOT."""
    trainer = Trainer(TicTacToe(), tiny_config)

    best_calls = {"n": 0}
    candidate_calls = {"n": 0}
    original_best_fwd = trainer.best_net.forward
    original_candidate_fwd = trainer.candidate_net.forward

    def count_best(*a, **k):
        best_calls["n"] += 1
        return original_best_fwd(*a, **k)

    def count_candidate(*a, **k):
        candidate_calls["n"] += 1
        return original_candidate_fwd(*a, **k)

    trainer.best_net.forward = count_best       # type: ignore[assignment]
    trainer.candidate_net.forward = count_candidate  # type: ignore[assignment]

    trainer._run_self_play_iteration()

    assert best_calls["n"] > 0
    assert candidate_calls["n"] == 0, (
        f"candidate_net was called {candidate_calls['n']} times during self-play; "
        "this would poison training data with a regressed network."
    )


def test_invariant_2_arena_disables_dirichlet_noise(tiny_config):
    """The arena agent must NOT add Dirichlet noise."""
    trainer = Trainer(TicTacToe(), tiny_config)
    agent = trainer._make_argmax_mcts_agent(trainer.best_net)

    captured = {"add_root_noise": None}
    from alphazero.mcts import MCTS
    original_search = MCTS.search
    def spy_search(self, root_state, num_simulations, add_root_noise):
        captured["add_root_noise"] = add_root_noise
        return original_search(self, root_state, num_simulations, add_root_noise)
    MCTS.search = spy_search  # type: ignore[method-assign]
    try:
        ttt = TicTacToe()
        agent(ttt, ttt.initial_state())
    finally:
        MCTS.search = original_search  # type: ignore[method-assign]

    assert captured["add_root_noise"] is False


def test_invariant_2_dirichlet_noise_affects_only_root(tiny_config):
    """Noise is applied at root expansion; not at deeper expansions."""
    import inspect
    from alphazero import mcts
    source = inspect.getsource(mcts.MCTS._simulate)
    assert "add_root_noise=False" in source, (
        "MCTS._simulate must call _expand with add_root_noise=False (only the "
        "initial root expansion gets noise)."
    )


def test_invariant_3_backup_flips_sign_per_ply():
    """Build a 4-deep path, verify sign flips per ply."""
    ttt = TicTacToe()
    mcts = MCTS(ttt, lambda s: (np.full(9, 1/9, dtype=np.float32), 0.5))

    root = Node()
    root.is_expanded = True
    n1 = Node(prior=1.0); root.children[0] = n1
    n2 = Node(prior=1.0); n1.children[0] = n2
    n3 = Node(prior=1.0); n2.children[0] = n3
    n4 = Node(prior=1.0); n3.children[0] = n4

    mcts._backup([root, n1, n2, n3, n4], leaf_value=+1.0)

    assert n4.value_sum == +1.0
    assert n3.value_sum == -1.0
    assert n2.value_sum == +1.0
    assert n1.value_sum == -1.0
    assert root.value_sum == +1.0


def test_invariant_4_encode_after_canonical_swaps_planes_when_player_changes():
    """After every move the perspective swap is consistent."""
    ttt = TicTacToe()
    s = ttt.initial_state()
    s = ttt.apply(s, 4)
    canon_o = ttt.canonical_state(s)
    enc_o = ttt.encode(canon_o)
    assert enc_o[0, 1, 1] == 0
    assert enc_o[1, 1, 1] == 1

    s = ttt.apply(s, 0)
    canon_x = ttt.canonical_state(s)
    enc_x = ttt.encode(canon_x)
    assert enc_x[0, 1, 1] == 1
    assert enc_x[1, 0, 0] == 1


def test_invariant_5_z_assignment_per_ply_mover_pov():
    from alphazero.selfplay import _assign_z
    ttt = TicTacToe()
    history = [
        (np.zeros((3, 3, 3), dtype=np.float32), np.full(9, 1/9, dtype=np.float32), p)
        for p in [1, -1, 1, -1, 1, -1]
    ]
    final_state = np.array([[-1, -1, -1], [1, 1, 0], [1, 0, 0]], dtype=np.int8)
    final_value = -1.0
    zs = _assign_z(history, final_value, final_state, ttt)
    assert zs == [-1.0, +1.0, -1.0, +1.0, -1.0, +1.0]


def test_invariant_6_replay_buffer_persists_across_iterations(tiny_config):
    """Two consecutive iterations should accumulate data in the same buffer."""
    trainer = Trainer(TicTacToe(), tiny_config)

    trainer._run_self_play_iteration()
    size_after_1 = len(trainer.replay_buffer)
    assert size_after_1 > 0

    trainer._run_self_play_iteration()
    size_after_2 = len(trainer.replay_buffer)

    assert size_after_2 >= size_after_1, (
        "Replay buffer shrank between iterations — this should never happen."
    )
