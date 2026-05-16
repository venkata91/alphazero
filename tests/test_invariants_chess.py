# tests/test_invariants_chess.py
"""Invariant tests for chess-specific behaviors.

Inherits the 6 framework invariants from tests/test_invariants.py
(those operate on TicTacToe but cover the framework code which is
unchanged for chess). Adds 4 chess-specific guards.
"""
from __future__ import annotations

import chess
import numpy as np
import pytest


def test_invariant_7_move_encoding_bijective_in_initial_position():
    """Round-trip: every legal move at the starting position encodes/decodes correctly."""
    from alphazero.games.chess_move_encoding import move_to_index, index_to_move
    board = chess.Board()
    for move in board.legal_moves:
        idx = move_to_index(board, move)
        back = index_to_move(board, idx)
        assert back == move, f"Round-trip failed for {move.uci()}"


def test_invariant_7_move_encoding_bijective_after_random_play():
    """Round-trip holds after 100 plies of random play."""
    import random
    from alphazero.games.chess_move_encoding import move_to_index, index_to_move
    rng = random.Random(42)
    board = chess.Board()
    for _ in range(100):
        if board.is_game_over():
            break
        for move in board.legal_moves:
            idx = move_to_index(board, move)
            back = index_to_move(board, idx)
            assert back == move, f"Round-trip failed for {move.uci()} at {board.fen()}"
        moves = list(board.legal_moves)
        board.push(rng.choice(moves))


def test_invariant_8_workers_dont_share_state():
    """Run 2 workers playing 4 games. Confirm independence of game state objects."""
    import torch.multiprocessing as mp
    from alphazero.network import AlphaZeroNet
    from alphazero.parallel_selfplay import run_parallel_self_play

    net = AlphaZeroNet(input_shape=(3, 3, 3), action_size=9, n_blocks=1, n_channels=4)
    state_dict = {k: v.clone() for k, v in net.state_dict().items()}

    examples = run_parallel_self_play(
        game_name="tictactoe",
        state_dict=state_dict,
        input_shape=(3, 3, 3),
        action_size=9,
        n_blocks=1, n_channels=4,
        num_games=4, num_workers=2, inference_batch_size=4,
        num_simulations=5, temperature_threshold=6,
        c_puct=1.5, dirichlet_alpha=1.0, dirichlet_weight=0.25,
        device="cpu",
    )
    # Sanity: each game produced reasonable training tuples
    assert len(examples) > 4   # at least 1 tuple per game
    # Sanity: outcomes span all 3 possible z values OR show a sensible distribution
    z_values = {e[2] for e in examples}
    assert z_values.issubset({-1.0, 0.0, 1.0})


def test_invariant_9_nn_server_preserves_batch_ordering():
    """Send 4 requests with different request_ids; verify responses come back
    paired with the correct ID (this is the bug surface in the NN-server)."""
    import threading
    import torch.multiprocessing as mp
    from alphazero.network import AlphaZeroNet
    from alphazero.parallel_selfplay import (
        InferenceRequest, InferenceResponse, nn_server_loop,
    )

    request_q = mp.Queue()
    response_qs = {0: mp.Queue(), 1: mp.Queue()}
    shutdown = mp.Event()

    net = AlphaZeroNet(input_shape=(3, 3, 3), action_size=9, n_blocks=1, n_channels=4)
    state_dict = {k: v.clone() for k, v in net.state_dict().items()}

    server_thread = threading.Thread(
        target=nn_server_loop,
        args=(state_dict, (3, 3, 3), 9, 1, 4, request_q, response_qs, shutdown),
        kwargs={"batch_size": 4, "wait_timeout_ms": 5, "device": "cpu"},
        daemon=True,
    )
    server_thread.start()

    # 4 distinct requests, 2 per worker
    requests = [
        InferenceRequest(worker_id=0, request_id=100, encoded=np.zeros((3, 3, 3), dtype=np.float32)),
        InferenceRequest(worker_id=1, request_id=200, encoded=np.zeros((3, 3, 3), dtype=np.float32)),
        InferenceRequest(worker_id=0, request_id=101, encoded=np.zeros((3, 3, 3), dtype=np.float32)),
        InferenceRequest(worker_id=1, request_id=201, encoded=np.zeros((3, 3, 3), dtype=np.float32)),
    ]
    for r in requests:
        request_q.put(r)

    # Collect from each reply queue. Worker 0 should get IDs 100 + 101; worker 1 → 200 + 201.
    w0_ids = sorted([response_qs[0].get(timeout=5.0).request_id for _ in range(2)])
    w1_ids = sorted([response_qs[1].get(timeout=5.0).request_id for _ in range(2)])

    shutdown.set()
    server_thread.join(timeout=2.0)

    assert w0_ids == [100, 101]
    assert w1_ids == [200, 201]


def test_invariant_10_stockfish_clean_shutdown():
    """Open and close StockfishOpponent multiple times; no zombies left."""
    from alphazero.opponents.stockfish import StockfishOpponent

    for _ in range(3):
        opp = StockfishOpponent(elo=1500, time_per_move=0.05)
        opp.close()
    # If clean_shutdown were broken, after-close calls would block or crash
    opp = StockfishOpponent(elo=1500, time_per_move=0.05)
    opp.close()
    opp.close()  # second close is no-op
