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
