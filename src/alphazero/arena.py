"""Head-to-head match between two agents.

An agent is a callable `(game, state) -> int` returning the chosen action.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .games.base import Game, State

Agent = Callable[[Game, State], int]


@dataclass
class MatchResult:
    wins_a: int = 0
    losses_a: int = 0
    draws: int = 0

    @property
    def total(self) -> int:
        return self.wins_a + self.losses_a + self.draws

    @property
    def win_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.wins_a + 0.5 * self.draws) / self.total


def play_match(
    game: Game,
    agent_a: Agent,
    agent_b: Agent,
    num_games: int,
) -> MatchResult:
    """Play num_games games alternating which agent moves first.

    Returns a MatchResult from agent_a's perspective.
    """
    result = MatchResult()
    for i in range(num_games):
        first, second = (agent_a, agent_b) if i % 2 == 0 else (agent_b, agent_a)
        winner_sign = _play_one_game(game, first, second)
        if winner_sign == 0:
            result.draws += 1
        else:
            first_is_a = (i % 2 == 0)
            a_won = (winner_sign == 1 and first_is_a) or (winner_sign == -1 and not first_is_a)
            if a_won:
                result.wins_a += 1
            else:
                result.losses_a += 1
    return result


def _play_one_game(game: Game, plus_agent: Agent, minus_agent: Agent) -> int:
    """Play one game; return winner sign (+1, -1) or 0 for draw."""
    state = game.initial_state()
    while game.terminal_value(state) is None:
        agent = plus_agent if game.current_player(state) == 1 else minus_agent
        action = agent(game, state)
        state = game.apply(state, action)

    final_v = game.terminal_value(state)
    if final_v == 0:
        return 0
    cur = game.current_player(state)
    if final_v == -1:
        return -cur
    else:
        return cur
