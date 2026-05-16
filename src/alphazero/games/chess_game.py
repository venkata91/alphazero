"""Chess: AlphaZero-on-chess Game implementation.

Wraps python-chess for rules; uses AlphaZero's 8×8×73 = 4672 action space.
State is a chess.Board (mutable; we copy + push to advance).
"""
from __future__ import annotations

import chess
import numpy as np

from .base import Game
from .chess_move_encoding import ACTION_SIZE, index_to_move, move_to_index

State = chess.Board

NUM_PLANES = 20


class Chess(Game):
    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (NUM_PLANES, 8, 8)

    @property
    def action_size(self) -> int:
        return ACTION_SIZE

    def initial_state(self) -> State:
        return chess.Board()

    def current_player(self, state: State) -> int:
        return 1 if state.turn == chess.WHITE else -1

    def legal_actions_mask(self, state: State) -> np.ndarray:
        mask = np.zeros(ACTION_SIZE, dtype=bool)
        for move in state.legal_moves:
            try:
                idx = move_to_index(state, move)
            except ValueError:
                continue  # shouldn't happen for any legal move; safe fallback
            mask[idx] = True
        return mask

    def apply(self, state: State, action: int) -> State:
        move = index_to_move(state, action)
        new_state = state.copy()
        new_state.push(move)
        return new_state

    # Filled in by later tasks
    def terminal_value(self, state: State) -> float | None:
        if not state.is_game_over(claim_draw=True):
            return None
        outcome = state.outcome(claim_draw=True)
        if outcome is None:
            # is_game_over said True but no outcome: edge case, treat as draw
            return 0.0
        if outcome.winner is None:
            return 0.0  # draw (stalemate, 3-fold rep, 50-move, insufficient material)
        # winner is chess.WHITE or chess.BLACK; compare against current player to move
        winner_is_current = (outcome.winner == state.turn)
        return 1.0 if winner_is_current else -1.0

    def encode(self, state: State) -> np.ndarray:
        raise NotImplementedError  # Task 7

    def canonical_state(self, state: State) -> State:
        raise NotImplementedError  # Task 8
