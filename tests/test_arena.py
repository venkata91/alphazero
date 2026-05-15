from __future__ import annotations

import numpy as np
import pytest

from alphazero.arena import MatchResult, play_match
from alphazero.games.tictactoe import TicTacToe
from alphazero.solvers.tictactoe_solver import solve_tictactoe_action


@pytest.fixture
def ttt() -> TicTacToe:
    return TicTacToe()


def random_agent_factory(seed: int):
    rng = np.random.default_rng(seed)
    def agent(game, state):
        legal = game.legal_actions_mask(state)
        choices = np.where(legal)[0]
        return int(rng.choice(choices))
    return agent


def perfect_agent(game, state):
    return solve_tictactoe_action(state)


def test_perfect_vs_random_perfect_never_loses(ttt: TicTacToe):
    np.random.seed(0)
    result = play_match(ttt, perfect_agent, random_agent_factory(42), num_games=20)
    assert isinstance(result, MatchResult)
    assert result.losses_a == 0
    assert result.total == 20


def test_perfect_vs_perfect_always_draws(ttt: TicTacToe):
    result = play_match(ttt, perfect_agent, perfect_agent, num_games=10)
    assert result.draws == 10
    assert result.wins_a == 0
    assert result.losses_a == 0


def test_play_match_alternates_colors(ttt: TicTacToe):
    np.random.seed(0)
    result = play_match(ttt, perfect_agent, random_agent_factory(7), num_games=20)
    assert result.total == 20


def test_match_result_win_rate(ttt: TicTacToe):
    np.random.seed(0)
    result = play_match(ttt, perfect_agent, random_agent_factory(11), num_games=10)
    rate = result.win_rate
    assert 0.5 <= rate <= 1.0
