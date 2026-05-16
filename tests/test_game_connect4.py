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


def _board(rows: list[list[int]]) -> np.ndarray:
    """Helper: literal board from a 2D list, row 0 at top."""
    return np.array(rows, dtype=np.int8)


def test_terminal_value_none_for_initial(c4: Connect4):
    assert c4.terminal_value(c4.initial_state()) is None


def test_terminal_value_none_when_in_progress(c4: Connect4):
    s = c4.initial_state()
    s = c4.apply(s, 3)
    s = c4.apply(s, 4)
    assert c4.terminal_value(s) is None


def test_terminal_value_horizontal_4_in_a_row(c4: Connect4):
    # +1 has 4 in a row at row 5, columns 0-3. -1 has 3 scattered.
    # Total nonzero = 7 (odd) → current_player = -1.
    state = _board([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [-1, -1, -1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0],
    ])
    assert c4.current_player(state) == -1
    assert c4.terminal_value(state) == -1.0


def test_terminal_value_vertical_4_in_a_row(c4: Connect4):
    # +1 has 4 stacked in column 0. -1 has 3 scattered.
    # Nonzero = 7, current_player = -1.
    state = _board([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, -1, -1, -1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
    ])
    assert c4.current_player(state) == -1
    assert c4.terminal_value(state) == -1.0


def test_terminal_value_diagonal_down_right(c4: Connect4):
    # +1 has 4 on the ↘ diagonal. 4 X + 5 O = 9 nonzero → -1's turn.
    state = _board([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [-1, 1, 0, 0, 0, 0, 0],
        [-1, -1, 1, 0, 0, 0, 0],
        [-1, -1, 0, 1, 0, 0, 0],
    ])
    assert c4.current_player(state) == -1
    assert c4.terminal_value(state) == -1.0


def test_terminal_value_diagonal_up_right(c4: Connect4):
    # +1 has 4 on the ↗ diagonal: (5,0), (4,1), (3,2), (2,3).
    # 5 X + 4 O = 9 nonzero → -1's turn.
    state = _board([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [-1, -1, 1, 0, 0, 0, 0],
        [-1, 1, -1, 0, 0, 0, 0],
        [1, -1, 1, 1, 0, 0, 0],
    ])
    assert c4.current_player(state) == -1
    assert c4.terminal_value(state) == -1.0


def test_terminal_value_draw_on_full_board_no_winner(c4: Connect4):
    """A block pattern where no player has 4 in a row."""
    state = _board([
        [ 1,  1, -1, -1,  1,  1, -1],
        [-1, -1,  1,  1, -1, -1,  1],
        [ 1,  1, -1, -1,  1,  1, -1],
        [-1, -1,  1,  1, -1, -1,  1],
        [ 1,  1, -1, -1,  1,  1, -1],
        [-1, -1,  1,  1, -1, -1,  1],
    ])
    assert c4.terminal_value(state) == 0.0


def test_canonical_state_is_identity_when_current_player_plus_one(c4: Connect4):
    s = c4.initial_state()
    np.testing.assert_array_equal(c4.canonical_state(s), s)


def test_canonical_state_inverts_when_current_player_minus_one(c4: Connect4):
    s = c4.apply(c4.initial_state(), 3)  # +1 drops in col 3; now -1's turn
    canon = c4.canonical_state(s)
    expected = np.zeros((6, 7), dtype=np.int8)
    expected[5, 3] = -1
    np.testing.assert_array_equal(canon, expected)


def test_encode_shape_and_dtype(c4: Connect4):
    enc = c4.encode(c4.canonical_state(c4.initial_state()))
    assert enc.shape == (3, 6, 7)
    assert enc.dtype == np.float32


def test_encode_initial_state_planes(c4: Connect4):
    enc = c4.encode(c4.canonical_state(c4.initial_state()))
    assert enc[0].sum() == 0
    assert enc[1].sum() == 0
    np.testing.assert_array_equal(enc[2], np.ones((6, 7), dtype=np.float32))


def test_encode_treats_canonical_player_as_my_pieces(c4: Connect4):
    """After +1 plays col 3, -1's canonical view: +1's piece appears on opp plane."""
    s = c4.apply(c4.initial_state(), 3)
    canon = c4.canonical_state(s)
    enc = c4.encode(canon)
    assert enc[0].sum() == 0
    assert enc[1, 5, 3] == 1
    assert enc[1].sum() == 1


def test_symmetries_yields_two_tuples(c4: Connect4):
    """Connect 4 has 2 symmetries: identity + horizontal (left-right) mirror."""
    s = c4.apply(c4.initial_state(), 0)
    enc = c4.encode(c4.canonical_state(s))
    policy = np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05], dtype=np.float32)
    syms = c4.symmetries(enc, policy)
    assert len(syms) == 2


def test_symmetries_first_is_identity(c4: Connect4):
    enc = c4.encode(c4.canonical_state(c4.initial_state()))
    policy = np.full(7, 1 / 7, dtype=np.float32)
    syms = c4.symmetries(enc, policy)
    np.testing.assert_array_equal(syms[0][0], enc)
    np.testing.assert_array_equal(syms[0][1], policy)


def test_symmetries_mirror_flips_columns(c4: Connect4):
    """Mirror reflects across the vertical axis: column 0 ↔ column 6."""
    s = c4.apply(c4.initial_state(), 0)  # +1's piece at (5, 0)
    enc = c4.encode(c4.canonical_state(s))
    policy = np.zeros(7, dtype=np.float32)
    policy[0] = 1.0
    syms = c4.symmetries(enc, policy)
    mirror_enc, mirror_policy = syms[1]
    # In mirrored encoding, piece is at column 6. Plane 1 = opp = X's pieces (from O's POV).
    assert mirror_enc[1, 5, 6] == 1
    assert mirror_enc[1, 5, 0] == 0
    np.testing.assert_array_equal(mirror_policy, [0, 0, 0, 0, 0, 0, 1])


def test_symmetries_preserve_policy_sum(c4: Connect4):
    enc = c4.encode(c4.canonical_state(c4.initial_state()))
    policy = np.full(7, 1 / 7, dtype=np.float32)
    for _, p in c4.symmetries(enc, policy):
        np.testing.assert_allclose(p.sum(), 1.0, rtol=1e-6)
