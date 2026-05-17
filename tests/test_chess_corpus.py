"""Tests for the chess corpus generation primitives."""
import random

import chess
import numpy as np
import pytest

from alphazero.corpus import Position, random_opening_moves


def test_random_opening_moves_pushes_n_plies():
    """random_opening_moves should advance the board by exactly n_plies."""
    board = chess.Board()
    rng = random.Random(42)
    random_opening_moves(board, n_plies=4, rng=rng)
    assert len(board.move_stack) == 4


def test_random_opening_moves_produces_only_legal_moves():
    """Every move in the opening sequence must be legal at its time."""
    board = chess.Board()
    rng = random.Random(42)
    random_opening_moves(board, n_plies=4, rng=rng)
    replay = chess.Board()
    for m in board.move_stack:
        assert m in replay.legal_moves
        replay.push(m)


def test_random_opening_moves_diverse_across_seeds():
    """Different seeds should produce different opening sequences (usually)."""
    boards = []
    for seed in range(5):
        board = chess.Board()
        random_opening_moves(board, n_plies=4, rng=random.Random(seed))
        boards.append(tuple(m.uci() for m in board.move_stack))
    assert len(set(boards)) >= 2


def test_random_opening_moves_stops_if_game_ends():
    """If the game ends mid-opening (shouldn't happen, but defensive), stop early."""
    board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert board.is_checkmate()
    rng = random.Random(42)
    random_opening_moves(board, n_plies=4, rng=rng)
    assert len(board.move_stack) == 0


def test_play_one_game_returns_positions_with_correct_shapes():
    """A single game should return a list of Position with correct shapes."""
    import chess.engine
    from alphazero.corpus import play_one_game

    engine = chess.engine.SimpleEngine.popen_uci("stockfish")
    engine.configure({"Threads": 1, "Hash": 16})
    try:
        rng = random.Random(42)
        positions = play_one_game(engine, time_per_move=0.01, rng=rng)
    finally:
        engine.quit()

    assert len(positions) > 4
    for p in positions:
        assert p.encoded_state.shape == (20, 8, 8)
        assert p.encoded_state.dtype == np.int8
        assert 0 <= p.move_index < 4672
        assert p.z in (-1, 0, 1)


def test_play_one_game_z_values_alternate_within_game():
    """In a decisive game, z values alternate sign between plies (mover POV)."""
    import chess.engine
    from alphazero.corpus import play_one_game

    engine = chess.engine.SimpleEngine.popen_uci("stockfish")
    engine.configure({"Threads": 1, "Hash": 16})
    try:
        rng = random.Random(123)
        positions = play_one_game(engine, time_per_move=0.01, rng=rng)
    finally:
        engine.quit()

    z_values = [p.z for p in positions]
    if z_values[-1] == 0:
        assert all(z == 0 for z in z_values)
    else:
        for i in range(len(z_values) - 1):
            assert z_values[i] == -z_values[i + 1], (
                f"At ply {i}: z={z_values[i]} but next z={z_values[i+1]}, "
                f"expected alternation in decisive game"
            )
