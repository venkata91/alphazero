# tests/test_game_chess.py
import chess
import numpy as np
import pytest

from alphazero.games.chess_game import Chess


@pytest.fixture
def game():
    return Chess()


def test_input_shape_and_action_size(game):
    assert game.input_shape == (20, 8, 8)
    assert game.action_size == 4672


def test_initial_state_is_starting_position(game):
    s = game.initial_state()
    assert isinstance(s, chess.Board)
    assert s.fen() == chess.STARTING_FEN


def test_current_player_white_first(game):
    assert game.current_player(game.initial_state()) == 1


def test_apply_advances_state(game):
    s = game.initial_state()
    # e2-e4 = action index 12*73 + 1 = 877 (queen direction N, distance 2 from e2)
    from alphazero.games.chess_move_encoding import move_to_index
    move = chess.Move.from_uci("e2e4")
    idx = move_to_index(s, move)
    s_after = game.apply(s, idx)
    assert s_after.turn == chess.BLACK
    assert s.turn == chess.WHITE  # original not mutated
    assert s_after.piece_at(chess.E4) is not None
    assert s_after.piece_at(chess.E4).piece_type == chess.PAWN


def test_legal_actions_mask_initial_position_has_20_legal(game):
    """At the starting position, white has 20 legal moves (16 pawn + 4 knight)."""
    s = game.initial_state()
    mask = game.legal_actions_mask(s)
    assert mask.shape == (4672,)
    assert mask.dtype == bool
    assert mask.sum() == 20


def test_terminal_value_initial_in_progress(game):
    assert game.terminal_value(game.initial_state()) is None


def test_terminal_value_white_checkmate_returns_minus_one_for_black(game):
    """Fool's mate: white plays f3, g4; black plays Qh4# winning.
    The terminal position has white to move (and being checkmated)."""
    s = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert s.is_checkmate()
    # current player (white) is the one being mated
    assert game.terminal_value(s) == -1.0


def test_terminal_value_stalemate_returns_zero(game):
    """Classic stalemate position."""
    s = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert s.is_stalemate()
    assert game.terminal_value(s) == 0.0


def test_terminal_value_insufficient_material_returns_zero(game):
    """King vs King: insufficient material, must be a draw."""
    s = chess.Board("8/8/8/4k3/8/8/8/4K3 w - - 0 1")
    assert s.is_insufficient_material()
    assert game.terminal_value(s) == 0.0


def test_terminal_value_50_move_rule_returns_zero(game):
    """Halfmove clock at 100 (50 full moves without capture/pawn move) is a draw."""
    s = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 100 75")
    # Note: chess.Board considers this a draw under can_claim_draw
    assert game.terminal_value(s) == 0.0


def test_encode_shape_and_dtype(game):
    s = game.initial_state()
    enc = game.encode(s)
    assert enc.shape == (20, 8, 8)
    assert enc.dtype == np.float32


def test_encode_initial_position_my_pawns_on_rank_2(game):
    """White to move at start: my pawns (plane 0) on rank 2 (board row index 1)."""
    s = game.initial_state()
    enc = game.encode(s)
    # Plane 0 = my pawns. python-chess square_rank(): a2=8 → rank 1 (0-indexed)
    # We pack squares into (8, 8) as enc[plane][rank][file]
    assert (enc[0, 1, :] == 1).all()           # all 8 files of rank 2 have my pawns
    assert (enc[0, 0, :] == 0).all()           # rank 1 has no my-pawns


def test_encode_initial_position_opp_pawns_on_rank_7(game):
    s = game.initial_state()
    enc = game.encode(s)
    # Plane 6 = opponent's pawns. Rank 7 = board index 6.
    assert (enc[6, 6, :] == 1).all()


def test_encode_ones_plane_all_ones(game):
    s = game.initial_state()
    enc = game.encode(s)
    # Plane 12 = ones plane
    assert (enc[12] == 1).all()


def test_encode_castling_rights_initial_all_one(game):
    """At the starting position, all 4 castling rights are present."""
    s = game.initial_state()
    enc = game.encode(s)
    # Planes 13-16: my-K, my-Q, opp-K, opp-Q castling
    for plane in (13, 14, 15, 16):
        assert (enc[plane] == 1).all()


def test_encode_castling_rights_after_white_kingside_zero(game):
    """After Ke1-e2, white has lost both castling rights."""
    s = game.initial_state()
    s.push(chess.Move.from_uci("e2e3"))
    s.push(chess.Move.from_uci("e7e6"))
    s.push(chess.Move.from_uci("e1e2"))
    s.push(chess.Move.from_uci("e8e7"))
    # Now back to white; current player's K and Q castling are both lost
    enc = game.encode(s)
    assert (enc[13] == 0).all()
    assert (enc[14] == 0).all()
