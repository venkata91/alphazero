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


def test_write_shard_creates_npz_with_correct_keys_and_dtypes(tmp_path):
    """Writing a shard produces a .npz with states, move_indices, outcomes."""
    from alphazero.corpus import Position, write_shard

    positions = [
        Position(
            encoded_state=np.ones((20, 8, 8), dtype=np.int8),
            move_index=42,
            z=1,
        ),
        Position(
            encoded_state=np.zeros((20, 8, 8), dtype=np.int8),
            move_index=100,
            z=-1,
        ),
    ]
    shard_path = write_shard(tmp_path, worker_id=3, shard_counter=7, positions=positions)
    assert shard_path.exists()
    assert shard_path.name == "shard_w3_s0007.npz"

    data = np.load(shard_path)
    assert set(data.files) == {"states", "move_indices", "outcomes"}
    assert data["states"].shape == (2, 20, 8, 8)
    assert data["states"].dtype == np.int8
    assert data["move_indices"].shape == (2,)
    assert data["move_indices"].dtype == np.int32
    assert data["move_indices"].tolist() == [42, 100]
    assert data["outcomes"].shape == (2,)
    assert data["outcomes"].dtype == np.int8
    assert data["outcomes"].tolist() == [1, -1]


def test_write_shard_round_trip_preserves_data(tmp_path):
    """Round-trip: write a shard, load it, verify all values match."""
    from alphazero.corpus import Position, write_shard

    rng = np.random.default_rng(42)
    positions = []
    for i in range(50):
        positions.append(Position(
            encoded_state=rng.integers(0, 2, size=(20, 8, 8), dtype=np.int8),
            move_index=int(rng.integers(0, 4672)),
            z=int(rng.choice([-1, 0, 1])),
        ))
    shard_path = write_shard(tmp_path, worker_id=0, shard_counter=0, positions=positions)

    data = np.load(shard_path)
    for i, p in enumerate(positions):
        assert np.array_equal(data["states"][i], p.encoded_state)
        assert data["move_indices"][i] == p.move_index
        assert data["outcomes"][i] == p.z


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
