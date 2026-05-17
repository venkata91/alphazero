"""Chess corpus generation primitives.

Stockfish-vs-Stockfish game generation for supervised pre-training of
AlphaZeroNet. Each worker process plays games independently and writes
its own .npz shards.

A Position is a (encoded_state, move_index, z) tuple where:
    encoded_state: np.ndarray of shape (20, 8, 8), dtype int8
        — Chess.encode(board) cast to int8 for compact storage
    move_index: int in [0, 4672)
        — chess_move_encoding.move_to_index(board, move)
    z: int in {-1, 0, +1}
        — game outcome from the mover's POV at that position
"""
from __future__ import annotations

import random
from typing import NamedTuple

import chess
import numpy as np


class Position(NamedTuple):
    encoded_state: np.ndarray   # (20, 8, 8) int8
    move_index: int             # [0, 4672)
    z: int                      # {-1, 0, +1}


def random_opening_moves(
    board: chess.Board, n_plies: int, rng: random.Random
) -> None:
    """Play n_plies uniformly-random legal moves into the board (in place).

    Stops early if the game ends before n_plies (defensive; rare).
    """
    for _ in range(n_plies):
        if board.is_game_over(claim_draw=True):
            return
        legal = list(board.legal_moves)
        move = rng.choice(legal)
        board.push(move)
