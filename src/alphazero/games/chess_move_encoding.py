"""AlphaZero 8×8×73 = 4672 action-space encoding for chess.

Each chess move is encoded as (source_square, move_type) where:
- source_square ∈ [0, 64): standard python-chess square indexing (a1=0, ..., h8=63)
- move_type ∈ [0, 73): one of three categories
    - Queen-like (8 directions × 7 distances): indices [0, 56)
    - Knight moves (8 L-shapes):                 indices [56, 64)
    - Underpromotions (3 pieces × 3 file dirs):  indices [64, 73)

Flat action index = source_square * 73 + move_type, range [0, 4672).

NOTE: this module currently implements only queen-like moves. Knight moves
and underpromotions are added in Tasks 3 and 4.
"""
from __future__ import annotations

import chess

# 8 compass directions as (drank, dfile) deltas
# Order matters: matches the original AlphaZero paper for reproducibility
QUEEN_DIRECTIONS: list[tuple[int, int]] = [
    ( 1,  0),  # N    (rank increases)
    ( 1,  1),  # NE
    ( 0,  1),  # E    (file increases)
    (-1,  1),  # SE
    (-1,  0),  # S
    (-1, -1),  # SW
    ( 0, -1),  # W
    ( 1, -1),  # NW
]

# 8 knight L-shapes as (drank, dfile) deltas
KNIGHT_DELTAS: list[tuple[int, int]] = [
    ( 2,  1),  # NNE
    ( 1,  2),  # ENE
    (-1,  2),  # ESE
    (-2,  1),  # SSE
    (-2, -1),  # SSW
    (-1, -2),  # WSW
    ( 1, -2),  # WNW
    ( 2, -1),  # NNW
]

ACTION_SIZE = 64 * 73  # 4672


def _queen_move_to_type(drank: int, dfile: int) -> int | None:
    """Return move_type ∈ [0, 56) for a queen-like move, or None if not queen-like."""
    if drank == 0 and dfile == 0:
        return None  # null move
    distance = max(abs(drank), abs(dfile))
    if distance > 7:
        return None
    # Direction sign (normalized)
    dr_sign = (drank > 0) - (drank < 0)
    df_sign = (dfile > 0) - (dfile < 0)
    # Must move in a straight line (rank, file, or diagonal)
    if drank != 0 and dfile != 0 and abs(drank) != abs(dfile):
        return None  # not queen-like
    try:
        direction_idx = QUEEN_DIRECTIONS.index((dr_sign, df_sign))
    except ValueError:
        return None
    return direction_idx * 7 + (distance - 1)


def _type_to_queen_move(move_type: int) -> tuple[int, int] | None:
    """Inverse of _queen_move_to_type. Returns (drank, dfile) or None if not queen-like."""
    if not (0 <= move_type < 56):
        return None
    direction_idx, distance_idx = divmod(move_type, 7)
    distance = distance_idx + 1
    dr_sign, df_sign = QUEEN_DIRECTIONS[direction_idx]
    return dr_sign * distance, df_sign * distance


def move_to_index(board: chess.Board, move: chess.Move) -> int:
    src = move.from_square
    dst = move.to_square
    drank = chess.square_rank(dst) - chess.square_rank(src)
    dfile = chess.square_file(dst) - chess.square_file(src)

    # Queen-like (and queen-promotions)
    queen_type = _queen_move_to_type(drank, dfile)
    if queen_type is not None and move.promotion in (None, chess.QUEEN):
        return src * 73 + queen_type

    # Knight
    if (drank, dfile) in KNIGHT_DELTAS:
        knight_idx = KNIGHT_DELTAS.index((drank, dfile))
        return src * 73 + (56 + knight_idx)

    raise NotImplementedError(
        f"Underpromotion encoding not implemented yet "
        f"(move={move}, drank={drank}, dfile={dfile})"
    )


def index_to_move(board: chess.Board, index: int) -> chess.Move:
    if not (0 <= index < ACTION_SIZE):
        raise ValueError(f"index {index} out of range [0, {ACTION_SIZE})")
    src, move_type = divmod(index, 73)

    if 0 <= move_type < 56:
        delta = _type_to_queen_move(move_type)
        drank, dfile = delta
        return _build_move_with_promotion(board, src, drank, dfile)

    if 56 <= move_type < 64:
        knight_idx = move_type - 56
        drank, dfile = KNIGHT_DELTAS[knight_idx]
        dst_rank = chess.square_rank(src) + drank
        dst_file = chess.square_file(src) + dfile
        if not (0 <= dst_rank < 8 and 0 <= dst_file < 8):
            raise ValueError(f"index {index} decodes off-board")
        dst = chess.square(dst_file, dst_rank)
        return chess.Move(src, dst)

    raise NotImplementedError(f"index {index} (move_type {move_type}) is an underpromotion (not impl yet)")


def _build_move_with_promotion(board: chess.Board, src: int, drank: int, dfile: int) -> chess.Move:
    """Build a queen-like move from src + delta, auto-detecting queen promotion."""
    dst_rank = chess.square_rank(src) + drank
    dst_file = chess.square_file(src) + dfile
    if not (0 <= dst_rank < 8 and 0 <= dst_file < 8):
        raise ValueError("off-board destination")
    dst = chess.square(dst_file, dst_rank)
    promotion = None
    piece = board.piece_at(src)
    if piece is not None and piece.piece_type == chess.PAWN:
        if (piece.color == chess.WHITE and dst_rank == 7) or \
           (piece.color == chess.BLACK and dst_rank == 0):
            promotion = chess.QUEEN
    return chess.Move(src, dst, promotion=promotion)
