# tests/test_move_encoding.py
import chess
import pytest

from alphazero.games.chess_move_encoding import (
    ACTION_SIZE,
    move_to_index,
    index_to_move,
)


def test_action_size_is_4672():
    assert ACTION_SIZE == 4672  # 64 squares × 73 move types


def test_white_pawn_e2_e4_encodes():
    """Famous opening: e2-e4 = queen-direction N, distance 2, source e2."""
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    idx = move_to_index(board, move)
    assert 0 <= idx < ACTION_SIZE


def test_white_pawn_e2_e4_round_trip():
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    idx = move_to_index(board, move)
    back = index_to_move(board, idx)
    assert back == move


def test_white_rook_a1_a8_round_trip():
    """Rook move along a-file (7 squares N)."""
    board = chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
    move = chess.Move.from_uci("a1a8")
    idx = move_to_index(board, move)
    back = index_to_move(board, idx)
    assert back == move


def test_white_bishop_diagonal_round_trip():
    """Bishop move along the long diagonal."""
    board = chess.Board("8/8/8/8/8/8/8/B6k w - - 0 1")
    move = chess.Move.from_uci("a1h8")
    idx = move_to_index(board, move)
    back = index_to_move(board, idx)
    assert back == move


def test_white_queen_all_8_directions_round_trip():
    """A queen in the middle can move in 8 directions. All encode/decode."""
    board = chess.Board("8/8/8/4Q3/8/8/8/4K2k w - - 0 1")
    # 8 destinations, one per direction
    destinations = ["e6", "f6", "f5", "f4", "e4", "d4", "d5", "d6"]
    for dst in destinations:
        move = chess.Move.from_uci(f"e5{dst}")
        idx = move_to_index(board, move)
        back = index_to_move(board, idx)
        assert back == move, f"Failed for queen move e5-{dst}"


def test_white_king_one_square_each_direction_round_trip():
    """King is queen-distance-1 in our encoding."""
    board = chess.Board("8/8/8/8/8/4K3/8/4k3 w - - 0 1")
    destinations = ["e4", "f4", "f3", "f2", "e2", "d2", "d3", "d4"]
    for dst in destinations:
        move = chess.Move.from_uci(f"e3{dst}")
        idx = move_to_index(board, move)
        back = index_to_move(board, idx)
        assert back == move, f"Failed for king move e3-{dst}"
