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
    """Convert a python-chess Move to a flat action index in [0, 4672).

    Only handles queen-like moves so far. Knights and underpromotions raise.
    """
    src = move.from_square
    dst = move.to_square
    drank = chess.square_rank(dst) - chess.square_rank(src)
    dfile = chess.square_file(dst) - chess.square_file(src)

    queen_type = _queen_move_to_type(drank, dfile)
    if queen_type is not None and move.promotion in (None, chess.QUEEN):
        # Queen promotion is encoded as a queen-direction move (not as an underpromotion)
        return src * 73 + queen_type

    raise NotImplementedError(
        f"Knight and underpromotion encoding not implemented yet "
        f"(move={move}, drank={drank}, dfile={dfile})"
    )


def index_to_move(board: chess.Board, index: int) -> chess.Move:
    """Convert a flat action index back to a python-chess Move.

    Only handles queen-like moves so far.
    """
    if not (0 <= index < ACTION_SIZE):
        raise ValueError(f"index {index} out of range [0, {ACTION_SIZE})")
    src, move_type = divmod(index, 73)
    delta = _type_to_queen_move(move_type)
    if delta is None:
        raise NotImplementedError(
            f"index {index} decodes to non-queen-like move_type {move_type}"
        )
    drank, dfile = delta
    dst_rank = chess.square_rank(src) + drank
    dst_file = chess.square_file(src) + dfile
    if not (0 <= dst_rank < 8 and 0 <= dst_file < 8):
        raise ValueError(f"index {index} decodes to off-board destination")
    dst = chess.square(dst_file, dst_rank)
    # Detect promotion: a pawn moving onto rank 0 or 7 (white pawn → rank 7, black pawn → rank 0)
    promotion = None
    piece = board.piece_at(src)
    if piece is not None and piece.piece_type == chess.PAWN:
        if (piece.color == chess.WHITE and dst_rank == 7) or \
           (piece.color == chess.BLACK and dst_rank == 0):
            promotion = chess.QUEEN  # default; underpromotions handled separately
    return chess.Move(src, dst, promotion=promotion)
