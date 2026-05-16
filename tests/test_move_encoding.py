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


def test_knight_round_trip_all_8_directions():
    """A knight at e4 has 8 possible destinations."""
    board = chess.Board("8/8/8/8/4N3/8/8/4K2k w - - 0 1")
    destinations = ["g5", "f6", "d6", "c5", "c3", "d2", "f2", "g3"]
    for dst in destinations:
        move = chess.Move.from_uci(f"e4{dst}")
        idx = move_to_index(board, move)
        back = index_to_move(board, idx)
        assert back == move, f"Failed for knight move e4-{dst}"


def test_knight_at_corner_partial_round_trip():
    """A knight at a1 has fewer legal moves but encoding must still work."""
    board = chess.Board("4k3/8/8/8/8/8/8/N6K w - - 0 1")
    for dst in ["b3", "c2"]:
        move = chess.Move.from_uci(f"a1{dst}")
        idx = move_to_index(board, move)
        back = index_to_move(board, idx)
        assert back == move


def test_underpromotion_round_trip_all_combinations():
    """Pawn on 7th rank: 3 promotion pieces × 3 file directions = 9 underpromotions."""
    # White pawn on e7; possible promotions: knight, bishop, rook (queen is auto via queen direction)
    # File directions: capture-left (d8), forward (e8), capture-right (f8)
    board = chess.Board("3rkr2/4P3/8/8/8/8/8/4K3 w - - 0 1")
    for piece in (chess.KNIGHT, chess.BISHOP, chess.ROOK):
        for dst_file in ("d", "e", "f"):
            try:
                move = chess.Move.from_uci(f"e7{dst_file}8{chess.piece_symbol(piece)}")
            except Exception:
                continue
            if move not in board.legal_moves:
                continue
            idx = move_to_index(board, move)
            back = index_to_move(board, idx)
            assert back == move, f"Underpromotion round-trip failed for {move.uci()}"


def test_bijective_every_legal_move_in_initial_position():
    """Every legal move in the starting position round-trips perfectly."""
    board = chess.Board()
    for move in board.legal_moves:
        idx = move_to_index(board, move)
        back = index_to_move(board, idx)
        assert back == move, f"Round-trip failed for {move.uci()}"


def test_bijective_after_random_play():
    """100 moves of random play; every legal move at every position round-trips."""
    import random
    rng = random.Random(42)
    board = chess.Board()
    for _ in range(100):
        if board.is_game_over():
            break
        for move in board.legal_moves:
            idx = move_to_index(board, move)
            back = index_to_move(board, idx)
            assert back == move, f"Round-trip failed for {move.uci()} at fen={board.fen()}"
        moves = list(board.legal_moves)
        board.push(rng.choice(moves))


def test_index_to_move_rejects_out_of_range():
    board = chess.Board()
    with pytest.raises(ValueError):
        index_to_move(board, -1)
    with pytest.raises(ValueError):
        index_to_move(board, ACTION_SIZE)
