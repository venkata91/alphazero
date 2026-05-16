import numpy as np
import pytest

from alphazero.games.connect4 import Connect4


@pytest.fixture
def c4() -> Connect4:
    return Connect4()


def test_input_shape_and_action_size(c4: Connect4):
    assert c4.input_shape == (3, 6, 7)
    assert c4.action_size == 7


def test_initial_state_is_empty(c4: Connect4):
    s = c4.initial_state()
    assert s.shape == (6, 7)
    assert s.dtype == np.int8
    assert np.all(s == 0)


def test_current_player_starts_at_plus_one(c4: Connect4):
    assert c4.current_player(c4.initial_state()) == 1


def test_apply_drops_piece_to_bottom_row_of_column(c4: Connect4):
    """First piece in a column should land at row 5 (the bottom)."""
    s = c4.initial_state()
    s = c4.apply(s, action=3)  # player +1 drops in column 3
    assert s[5, 3] == 1
    assert np.count_nonzero(s) == 1
    assert c4.current_player(s) == -1


def test_apply_stacks_pieces_within_a_column(c4: Connect4):
    s = c4.initial_state()
    s = c4.apply(s, action=3)  # +1 at (5, 3)
    s = c4.apply(s, action=3)  # -1 stacks on top at (4, 3)
    s = c4.apply(s, action=3)  # +1 at (3, 3)
    assert s[5, 3] == 1
    assert s[4, 3] == -1
    assert s[3, 3] == 1
    assert s[2, 3] == 0


def test_apply_does_not_mutate_input(c4: Connect4):
    s = c4.initial_state()
    s_copy = s.copy()
    _ = c4.apply(s, 0)
    np.testing.assert_array_equal(s, s_copy)


def test_apply_raises_on_full_column(c4: Connect4):
    s = c4.initial_state()
    for _ in range(6):
        s = c4.apply(s, 0)
    assert s[0, 0] != 0
    with pytest.raises(ValueError, match="full"):
        c4.apply(s, 0)


def test_legal_actions_mask_initial_state_all_seven_legal(c4: Connect4):
    mask = c4.legal_actions_mask(c4.initial_state())
    assert mask.shape == (7,)
    assert mask.dtype == bool
    assert mask.all()


def test_legal_actions_mask_after_filling_one_column(c4: Connect4):
    s = c4.initial_state()
    for _ in range(6):
        s = c4.apply(s, 3)
    mask = c4.legal_actions_mask(s)
    assert mask.sum() == 6
    assert not mask[3]
