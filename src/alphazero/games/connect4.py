"""Connect 4: 6×7 board with gravity. Pieces drop to the lowest empty cell of the
chosen column.

State representation:
    np.ndarray shape (6, 7), dtype int8, values in {+1, 0, -1}.
    Row 0 is the top of the board; row 5 is the bottom (where pieces land).
    +1 = first player (red); -1 = second player (yellow); 0 = empty.

Action space: integers 0..6 — the column to drop a piece into.
"""
from __future__ import annotations

import numpy as np

from .base import Game

State = np.ndarray

ROWS = 6
COLS = 7


class Connect4(Game):
    @property
    def input_shape(self) -> tuple[int, ...]:
        return (3, ROWS, COLS)

    @property
    def action_size(self) -> int:
        return COLS

    def initial_state(self) -> State:
        return np.zeros((ROWS, COLS), dtype=np.int8)

    def current_player(self, state: State) -> int:
        moves = int(np.count_nonzero(state))
        return 1 if moves % 2 == 0 else -1

    def legal_actions_mask(self, state: State) -> np.ndarray:
        # A column is legal iff its top cell (row 0) is empty.
        return state[0, :] == 0

    def apply(self, state: State, action: int) -> State:
        if state[0, action] != 0:
            raise ValueError(f"Illegal action {action}: column {action} is full")
        new_state = state.copy()
        # Drop: find the lowest empty cell in this column.
        for row in range(ROWS - 1, -1, -1):
            if new_state[row, action] == 0:
                new_state[row, action] = self.current_player(state)
                break
        return new_state

    def terminal_value(self, state: State) -> float | None:
        winner = self._find_winner(state)
        if winner is not None:
            return 1.0 if winner == self.current_player(state) else -1.0
        if np.all(state != 0):
            return 0.0  # full board, no winner → draw
        return None

    def _find_winner(self, state: State) -> int | None:
        """Return +1 / -1 if either player has 4 in a row, else None."""
        for player in (1, -1):
            # Horizontal — 4 consecutive in a row
            for r in range(ROWS):
                for c in range(COLS - 3):
                    if np.all(state[r, c:c+4] == player):
                        return player
            # Vertical — 4 consecutive in a column
            for c in range(COLS):
                for r in range(ROWS - 3):
                    if np.all(state[r:r+4, c] == player):
                        return player
            # Diagonal ↘ (top-left to bottom-right)
            for r in range(ROWS - 3):
                for c in range(COLS - 3):
                    if all(state[r+i, c+i] == player for i in range(4)):
                        return player
            # Diagonal ↗ (bottom-left to top-right)
            for r in range(3, ROWS):
                for c in range(COLS - 3):
                    if all(state[r-i, c+i] == player for i in range(4)):
                        return player
        return None

    # Stubs filled in by subsequent tasks
    def encode(self, state: State) -> np.ndarray:
        raise NotImplementedError  # Task 3

    def canonical_state(self, state: State) -> State:
        raise NotImplementedError  # Task 3
