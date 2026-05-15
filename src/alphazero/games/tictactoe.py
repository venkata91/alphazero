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
        winner = self._find_winner(state)
        if winner is not None:
            return 1.0 if winner == self.current_player(state) else -1.0
        if np.all(state != 0):
            return 0.0  # board full, no winner → draw
        return None

    def _find_winner(self, state: State) -> int | None:
        for player in (1, -1):
            for r in range(3):
                if np.all(state[r, :] == player):
                    return player
            for c in range(3):
                if np.all(state[:, c] == player):
                    return player
            if np.all(np.diag(state) == player):
                return player
            if np.all(np.diag(np.fliplr(state)) == player):
                return player
        return None

    def canonical_state(self, state: State) -> State:
        """Rewrite so current player's pieces are +1, opponent's are -1."""
        return (state * self.current_player(state)).astype(np.int8)

    def encode(self, state: State) -> np.ndarray:
        """Three planes: my pieces (+1 in canonical state), opp pieces (-1), ones.

        Caller is expected to pass a canonical state. The ones plane is a
        constant feature that helps small CNNs learn positional reasoning
        about board edges (a standard trick).
        """
        my = (state == 1).astype(np.float32)
        opp = (state == -1).astype(np.float32)
        ones = np.ones((3, 3), dtype=np.float32)
        return np.stack([my, opp, ones], axis=0)

    def symmetries(
        self, encoded: np.ndarray, policy: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """All 8 D₄ symmetries of a TTT board (4 rotations × 2 mirrors).

        encoded shape: (C, H, W) = (3, 3, 3); policy shape: (9,) row-major.
        Returns list of (encoded_sym, policy_sym) pairs; element 0 is identity.
        """
        results: list[tuple[np.ndarray, np.ndarray]] = []
        pi_grid = policy.reshape(3, 3)
        # Generate all 8 D4 symmetries: rotations alternating with reflections
        for k in range(4):
            # Rotation by k*90 degrees
            rot_enc = np.rot90(encoded, k=k, axes=(1, 2)).copy()
            rot_pi = np.rot90(pi_grid, k=k).copy()
            results.append((rot_enc, rot_pi.flatten()))

            # Reflection after this rotation (vertical axis flip)
            refl_enc = np.flip(rot_enc, axis=2).copy()
            refl_pi = np.flip(rot_pi, axis=1).copy()
            results.append((refl_enc, refl_pi.flatten()))

        return results
