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


def _board(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=np.int8)


def test_terminal_value_none_when_in_progress(ttt: TicTacToe):
    assert ttt.terminal_value(ttt.initial_state()) is None
    s = ttt.apply(ttt.initial_state(), 4)
    assert ttt.terminal_value(s) is None


def test_terminal_value_x_wins_row_returns_minus_one_for_o(ttt: TicTacToe):
    # X just played and won; now O's turn → from O's POV, X won → -1
    state = _board([
        [1, 1, 1],
        [-1, -1, 0],
        [0, 0, 0],
    ])
    assert ttt.current_player(state) == -1
    assert ttt.terminal_value(state) == -1.0


def test_terminal_value_x_wins_col(ttt: TicTacToe):
    state = _board([
        [1, -1, 0],
        [1, -1, 0],
        [1, 0, 0],
    ])
    assert ttt.terminal_value(state) == -1.0


def test_terminal_value_x_wins_diag(ttt: TicTacToe):
    state = _board([
        [1, -1, 0],
        [-1, 1, 0],
        [0, 0, 1],
    ])
    assert ttt.terminal_value(state) == -1.0


def test_terminal_value_x_wins_antidiag(ttt: TicTacToe):
    state = _board([
        [0, -1, 1],
        [-1, 1, 0],
        [1, 0, 0],
    ])
    assert ttt.terminal_value(state) == -1.0


def test_terminal_value_o_wins_returns_minus_one_for_x(ttt: TicTacToe):
    state = _board([
        [-1, -1, -1],
        [1, 1, 0],
        [1, 0, 0],
    ])
    assert ttt.current_player(state) == 1
    assert ttt.terminal_value(state) == -1.0


def test_terminal_value_draw_returns_zero(ttt: TicTacToe):
    state = _board([
        [1, -1, 1],
        [1, -1, -1],
        [-1, 1, 1],
    ])
    assert ttt.terminal_value(state) == 0.0


def test_canonical_state_is_identity_when_current_player_plus_one(ttt: TicTacToe):
    s = ttt.initial_state()
    np.testing.assert_array_equal(ttt.canonical_state(s), s)


def test_canonical_state_inverts_when_current_player_minus_one(ttt: TicTacToe):
    s = ttt.apply(ttt.initial_state(), 4)  # X plays center; now O's turn
    canon = ttt.canonical_state(s)
    expected = np.zeros((3, 3), dtype=np.int8)
    expected[1, 1] = -1
    np.testing.assert_array_equal(canon, expected)


def test_encode_shape_matches_input_shape(ttt: TicTacToe):
    s = ttt.initial_state()
    enc = ttt.encode(ttt.canonical_state(s))
    assert enc.shape == ttt.input_shape
    assert enc.dtype == np.float32


def test_encode_initial_state_my_and_opp_planes_empty(ttt: TicTacToe):
    enc = ttt.encode(ttt.canonical_state(ttt.initial_state()))
    assert enc[0].sum() == 0
    assert enc[1].sum() == 0
    np.testing.assert_array_equal(enc[2], np.ones((3, 3), dtype=np.float32))


def test_encode_treats_current_player_as_my_pieces(ttt: TicTacToe):
    """After X plays center, from O's canonical perspective:
       O's pieces (none) on plane 0; X's piece (center) on plane 1."""
    s = ttt.apply(ttt.initial_state(), 4)
    canon = ttt.canonical_state(s)
    enc = ttt.encode(canon)
    assert enc[0].sum() == 0
    assert enc[1, 1, 1] == 1
    assert enc[1].sum() == 1


def test_symmetries_yields_8_distinct_tuples(ttt: TicTacToe):
    """Asymmetric input → all 8 D₄ symmetries are distinct."""
    s = ttt.initial_state()
    s = ttt.apply(s, 0)  # X corner top-left — asymmetric
    canon = ttt.canonical_state(s)
    enc = ttt.encode(canon)
    policy = np.array([0.1, 0.05, 0.05, 0.05, 0.6, 0.05, 0.05, 0.05, 0.0], dtype=np.float32)
    syms = ttt.symmetries(enc, policy)
    assert len(syms) == 8
    seen = set()
    for enc_s, _ in syms:
        seen.add(enc_s.tobytes())
    assert len(seen) == 8


def test_symmetries_includes_identity_first(ttt: TicTacToe):
    """The first symmetry is (encoded, policy) unchanged."""
    enc = ttt.encode(ttt.canonical_state(ttt.initial_state()))
    pol = np.zeros(9, dtype=np.float32)
    pol[4] = 1.0
    syms = ttt.symmetries(enc, pol)
    np.testing.assert_array_equal(syms[0][0], enc)
    np.testing.assert_array_equal(syms[0][1], pol)


def test_symmetries_preserves_policy_sum(ttt: TicTacToe):
    enc = ttt.encode(ttt.canonical_state(ttt.initial_state()))
    pol = np.full(9, 1 / 9, dtype=np.float32)
    syms = ttt.symmetries(enc, pol)
    for _, p in syms:
        np.testing.assert_allclose(p.sum(), 1.0, rtol=1e-6)


def test_symmetries_action_index_follows_board_rotation(ttt: TicTacToe):
    """All-mass-on-corner stays all-mass-on-a-corner under symmetries."""
    enc = ttt.encode(ttt.canonical_state(ttt.initial_state()))
    pol = np.zeros(9, dtype=np.float32)
    pol[0] = 1.0  # top-left
    syms = ttt.symmetries(enc, pol)
    found = False
    for _, p in syms:
        if p[2] == 1.0:
            found = True
            break
    assert found
