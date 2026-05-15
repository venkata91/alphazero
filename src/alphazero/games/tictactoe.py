"""Tic-Tac-Toe: the smallest game we use to validate the AlphaZero pipeline.

State representation:
    np.ndarray shape (3, 3), dtype int8, values in {+1, 0, -1}.
    +1 = player X's piece; -1 = player O's piece; 0 = empty.
    Player to move is determined by piece-count parity (count_nonzero % 2).

Action space:
    Integers 0..8, row-major: action = 3*row + col.
"""
from __future__ import annotations

import numpy as np

from .base import Game

State = np.ndarray


class TicTacToe(Game):
    @property
    def input_shape(self) -> tuple[int, ...]:
        return (3, 3, 3)

    @property
    def action_size(self) -> int:
        return 9

    def initial_state(self) -> State:
        return np.zeros((3, 3), dtype=np.int8)

    def current_player(self, state: State) -> int:
        moves = int(np.count_nonzero(state))
        return 1 if moves % 2 == 0 else -1

    def legal_actions_mask(self, state: State) -> np.ndarray:
        return state.flatten() == 0

    def apply(self, state: State, action: int) -> State:
        r, c = divmod(action, 3)
        if state[r, c] != 0:
            raise ValueError(f"Illegal action {action}: square already occupied")
        new_state = state.copy()
        new_state[r, c] = self.current_player(state)
        return new_state

    # Filled in by later tasks:
    def terminal_value(self, state: State) -> float | None:
        raise NotImplementedError  # Task 5

    def encode(self, state: State) -> np.ndarray:
        raise NotImplementedError  # Task 6

    def canonical_state(self, state: State) -> State:
        raise NotImplementedError  # Task 6
