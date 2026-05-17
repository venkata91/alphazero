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


from pathlib import Path

import chess.engine

from .games.chess_game import Chess
from .games.chess_move_encoding import move_to_index


def write_shard(
    output_dir: Path,
    worker_id: int,
    shard_counter: int,
    positions: list[Position],
) -> Path:
    """Write a list of Positions to a compressed .npz shard.

    Filename format: shard_w{worker_id}_s{shard_counter:04d}.npz
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_path = output_dir / f"shard_w{worker_id}_s{shard_counter:04d}.npz"

    states = np.stack([p.encoded_state for p in positions]).astype(np.int8)
    move_indices = np.array([p.move_index for p in positions], dtype=np.int32)
    outcomes = np.array([p.z for p in positions], dtype=np.int8)

    np.savez_compressed(
        shard_path,
        states=states,
        move_indices=move_indices,
        outcomes=outcomes,
    )
    return shard_path


def worker_generate(
    worker_id: int,
    n_games: int,
    output_dir: Path,
    time_per_move: float,
    seed: int,
    *,
    shard_size: int = 10_000,
    stockfish_path: str = "stockfish",
) -> None:
    """Worker process entry point. Plays n_games and writes shards.

    Each worker is fully independent — owns its own Stockfish subprocess,
    its own RNG, its own shard counter, writes to its own filenames.
    """
    rng = random.Random(seed + worker_id)
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"Threads": 1, "Hash": 16})

    shard_buffer: list[Position] = []
    shard_counter = 0

    try:
        for _ in range(n_games):
            positions = play_one_game(engine, time_per_move=time_per_move, rng=rng)
            shard_buffer.extend(positions)

            while len(shard_buffer) >= shard_size:
                write_shard(output_dir, worker_id, shard_counter, shard_buffer[:shard_size])
                shard_buffer = shard_buffer[shard_size:]
                shard_counter += 1

        if shard_buffer:
            write_shard(output_dir, worker_id, shard_counter, shard_buffer)
    finally:
        try:
            engine.quit()
        except (chess.engine.EngineTerminatedError, BrokenPipeError):
            pass


def play_one_game(
    engine: chess.engine.SimpleEngine,
    time_per_move: float,
    rng: random.Random,
    *,
    opening_plies: int = 4,
) -> list[Position]:
    """Play one Stockfish-vs-Stockfish game and return Position tuples.

    Flow:
        1. Random opening (4 plies) for diversity
        2. Stockfish plays both sides until game ends
        3. For each recorded position, compute z from the final outcome
           in the position-mover's POV
    """
    game = Chess()
    board = chess.Board()
    random_opening_moves(board, n_plies=opening_plies, rng=rng)

    history: list[tuple[np.ndarray, int, int]] = []

    while not board.is_game_over(claim_draw=True):
        result = engine.play(board, chess.engine.Limit(time=time_per_move))
        move = result.move
        if move is None:
            break
        encoded = game.encode(board).astype(np.int8)
        mv_idx = move_to_index(board, move)
        mover_player = 1 if board.turn == chess.WHITE else -1
        history.append((encoded, mv_idx, mover_player))
        board.push(move)

    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        white_pov_z = 0
    elif outcome.winner == chess.WHITE:
        white_pov_z = 1
    else:
        white_pov_z = -1

    positions = []
    for encoded, mv_idx, mover_player in history:
        z_from_mover_pov = white_pov_z * mover_player
        positions.append(Position(
            encoded_state=encoded,
            move_index=mv_idx,
            z=int(z_from_mover_pov),
        ))
    return positions
