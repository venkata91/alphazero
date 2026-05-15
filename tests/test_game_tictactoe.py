import numpy as np
import pytest

from alphazero.games.tictactoe import TicTacToe


@pytest.fixture
def ttt() -> TicTacToe:
    return TicTacToe()


def test_input_shape_and_action_size(ttt: TicTacToe):
    assert ttt.input_shape == (3, 3, 3)
    assert ttt.action_size == 9


def test_initial_state_is_empty(ttt: TicTacToe):
    s = ttt.initial_state()
    assert s.shape == (3, 3)
    assert s.dtype == np.int8
    assert np.all(s == 0)


def test_current_player_starts_at_plus_one(ttt: TicTacToe):
    assert ttt.current_player(ttt.initial_state()) == 1


def test_apply_places_correct_piece(ttt: TicTacToe):
    s = ttt.initial_state()
    s1 = ttt.apply(s, action=4)  # X plays center
    assert s1[1, 1] == 1
    assert ttt.current_player(s1) == -1
    s2 = ttt.apply(s1, action=0)  # O plays corner
    assert s2[0, 0] == -1
    assert ttt.current_player(s2) == 1


def test_apply_does_not_mutate_input(ttt: TicTacToe):
    s = ttt.initial_state()
    s_copy = s.copy()
    _ = ttt.apply(s, 0)
    np.testing.assert_array_equal(s, s_copy)


def test_apply_raises_on_occupied_square(ttt: TicTacToe):
    s = ttt.apply(ttt.initial_state(), 4)
    with pytest.raises(ValueError, match="Illegal"):
        ttt.apply(s, 4)


def test_legal_actions_mask_initial_state_all_true(ttt: TicTacToe):
    mask = ttt.legal_actions_mask(ttt.initial_state())
    assert mask.shape == (9,)
    assert mask.dtype == bool
    assert mask.all()


def test_legal_actions_mask_after_one_move(ttt: TicTacToe):
    s = ttt.apply(ttt.initial_state(), 4)
    mask = ttt.legal_actions_mask(s)
    assert mask.sum() == 8
    assert not mask[4]
