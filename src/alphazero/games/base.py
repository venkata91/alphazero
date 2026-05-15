"""Abstract base class for games.

Concrete Games encapsulate everything game-specific: rules, terminal
detection, state encoding, perspective normalization, and symmetry-based
data augmentation. The rest of the framework (network, MCTS, training)
consumes a Game purely through this interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

State = Any  # game-specific; e.g. np.ndarray for TTT


class Game(ABC):
    """Abstract base class for a two-player, zero-sum, perfect-information game.

    Convention:
        - Players are +1 and -1.
        - terminal_value returns the result from the CURRENT PLAYER'S POV.
        - encode is intended to be called on `canonical_state(state)` —
          canonical_state rewrites the board so the current player's pieces
          are +1, freeing the network from learning whose-turn-is-it.
    """

    @property
    @abstractmethod
    def input_shape(self) -> tuple[int, ...]:
        """Shape of `encode(state)` output: (channels, height, width)."""

    @property
    @abstractmethod
    def action_size(self) -> int:
        """Total number of distinct actions (legal or not) in the action space."""

    @abstractmethod
    def initial_state(self) -> State:
        """Return the starting position."""

    @abstractmethod
    def current_player(self, state: State) -> int:
        """Return +1 or -1 — whose turn is it to move."""

    @abstractmethod
    def legal_actions_mask(self, state: State) -> np.ndarray:
        """Boolean array of shape (action_size,) — True for legal actions."""

    @abstractmethod
    def apply(self, state: State, action: int) -> State:
        """Return a new state after applying `action` for current_player."""

    @abstractmethod
    def terminal_value(self, state: State) -> float | None:
        """Return None if game in progress; else +1/0/-1 from current_player's POV."""

    @abstractmethod
    def encode(self, state: State) -> np.ndarray:
        """Stack of float32 planes — input to the neural network. Shape == input_shape."""

    @abstractmethod
    def canonical_state(self, state: State) -> State:
        """Rewrite state so current player sees themselves as +1."""

    def symmetries(
        self, encoded: np.ndarray, policy: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return equivalent (encoded_state, policy) pairs for data augmentation.

        Default: identity only. Override for games with symmetries
        (TTT has 8; Connect 4 has 2; chess has none).
        """
        return [(encoded, policy)]
