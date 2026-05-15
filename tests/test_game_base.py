import numpy as np
import pytest

from alphazero.games.base import Game


def test_game_is_abstract_and_cannot_be_instantiated():
    with pytest.raises(TypeError):
        Game()  # type: ignore[abstract]


def test_default_symmetries_returns_identity_singleton():
    """A subclass that doesn't override symmetries() gets [(s, π)] back."""

    class DummyGame(Game):
        @property
        def input_shape(self):
            return (1, 1, 1)

        @property
        def action_size(self):
            return 1

        def initial_state(self):
            return np.zeros((1, 1))

        def current_player(self, state):
            return 1

        def legal_actions_mask(self, state):
            return np.array([True])

        def apply(self, state, action):
            return state

        def terminal_value(self, state):
            return None

        def encode(self, state):
            return np.zeros((1, 1, 1), dtype=np.float32)

        def canonical_state(self, state):
            return state

    g = DummyGame()
    encoded = np.zeros((1, 1, 1), dtype=np.float32)
    policy = np.array([1.0], dtype=np.float32)
    syms = g.symmetries(encoded, policy)
    assert len(syms) == 1
    np.testing.assert_array_equal(syms[0][0], encoded)
    np.testing.assert_array_equal(syms[0][1], policy)
