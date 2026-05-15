"""Perfect-play Tic-Tac-Toe solver via memoized minimax.

TTT has only ~5478 reachable positions, so full enumeration with caching is
cheap. Used as eval ground truth.
"""
from __future__ import annotations

import numpy as np

from alphazero.games.tictactoe import TicTacToe

_cache: dict[bytes, int] = {}
_ttt = TicTacToe()


def solve_tictactoe_value(state: np.ndarray) -> int:
    """Return the perfect-play value for current_player at this state.

    +1 if current_player wins with optimal play, -1 if loses, 0 if draw.
    """
    key = state.tobytes()
    cached = _cache.get(key)
    if cached is not None:
        return cached

    terminal = _ttt.terminal_value(state)
    if terminal is not None:
        v = int(terminal)
        _cache[key] = v
        return v

    legal = _ttt.legal_actions_mask(state)
    best = -2
    for action in np.where(legal)[0]:
        next_state = _ttt.apply(state, int(action))
        v = -solve_tictactoe_value(next_state)
        if v > best:
            best = v

    _cache[key] = best
    return best


def solve_tictactoe_action(state: np.ndarray) -> int:
    """Return an optimal action for current_player at this state.

    Ties broken by lowest action index (deterministic).
    """
    terminal = _ttt.terminal_value(state)
    if terminal is not None:
        raise ValueError("Cannot solve a terminal position")

    legal = _ttt.legal_actions_mask(state)
    best_val = -2
    best_action = -1
    for action in np.where(legal)[0]:
        next_state = _ttt.apply(state, int(action))
        v = -solve_tictactoe_value(next_state)
        if v > best_val:
            best_val = v
            best_action = int(action)
    return best_action
