"""AlphaZero 8×8×73 = 4672 action-space encoding for chess.

Each chess move is encoded as (source_square, move_type) where:
- source_square ∈ [0, 64): standard python-chess square indexing (a1=0, ..., h8=63)
- move_type ∈ [0, 73): one of three categories
    - Queen-like (8 directions × 7 distances): indices [0, 56)
    - Knight moves (8 L-shapes):                 indices [56, 64)
    - Underpromotions (3 pieces × 3 file dirs):  indices [64, 73)

Flat action index = source_square * 73 + move_type, range [0, 4672).
Queen promotions are encoded via the queen-like path (drank=±1, dfile∈{-1,0,1});
underpromotions to knight/bishop/rook occupy move_type ∈ [64, 73).
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

# Underpromotion pieces (in this order): index 0=knight, 1=bishop, 2=rook
UNDERPROMO_PIECES: list[int] = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
# Underpromotion file deltas (relative to forward direction of the moving pawn)
# index 0 = capture-left, 1 = forward (push), 2 = capture-right
UNDERPROMO_FILE_DELTAS: list[int] = [-1, 0, 1]

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

    # Underpromotion (knight, bishop, rook only — queen is encoded as queen move)
    if move.promotion is not None and move.promotion != chess.QUEEN:
        piece = board.piece_at(src)
        if piece is None or piece.piece_type != chess.PAWN:
            raise ValueError(f"Underpromotion claim at square {src} but no pawn there (got {piece})")
        # Black pawns move "down" (drank=-1); flip drank to canonical "forward" for index
        forward_drank = drank if piece.color == chess.WHITE else -drank
        forward_dfile = dfile if piece.color == chess.WHITE else -dfile
        if forward_drank != 1:
            raise ValueError(f"Expected pawn promotion drank=1, got {forward_drank}")
        try:
            piece_idx = UNDERPROMO_PIECES.index(move.promotion)
        except ValueError:
            raise ValueError(f"Bad underpromotion piece: {move.promotion}")
        try:
            file_idx = UNDERPROMO_FILE_DELTAS.index(forward_dfile)
        except ValueError:
            raise ValueError(f"Bad underpromotion file delta: {forward_dfile}")
        move_type = 64 + piece_idx * 3 + file_idx
        return src * 73 + move_type

    # Queen-like (and queen-promotions)
    queen_type = _queen_move_to_type(drank, dfile)
    if queen_type is not None and move.promotion in (None, chess.QUEEN):
        return src * 73 + queen_type

    # Knight
    if (drank, dfile) in KNIGHT_DELTAS:
        knight_idx = KNIGHT_DELTAS.index((drank, dfile))
        return src * 73 + (56 + knight_idx)

    raise ValueError(f"Cannot encode move {move} (drank={drank}, dfile={dfile})")


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

    if 64 <= move_type < 73:
        underpromo = move_type - 64
        piece_idx, file_idx = divmod(underpromo, 3)
        promotion_piece = UNDERPROMO_PIECES[piece_idx]
        forward_dfile = UNDERPROMO_FILE_DELTAS[file_idx]
        # Determine pawn color from source square's piece
        piece = board.piece_at(src)
        if piece is None or piece.piece_type != chess.PAWN:
            raise ValueError(f"Underpromotion at square {src} but no pawn there")
        if piece.color == chess.WHITE:
            drank = 1
            dfile = forward_dfile
        else:
            drank = -1
            dfile = -forward_dfile  # flip for black pawn moving "down"
        dst_rank = chess.square_rank(src) + drank
        dst_file = chess.square_file(src) + dfile
        if not (0 <= dst_rank < 8 and 0 <= dst_file < 8):
            raise ValueError(f"index {index} decodes off-board")
        dst = chess.square(dst_file, dst_rank)
        return chess.Move(src, dst, promotion=promotion_piece)

    raise ValueError(f"Unreachable: move_type {move_type}")


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
