"""Tests for the chess corpus generation primitives."""
import random

import chess
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
