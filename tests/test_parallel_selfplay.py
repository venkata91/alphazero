# tests/test_parallel_selfplay.py
import numpy as np
import pytest
import torch
import torch.multiprocessing as mp

from alphazero.games.tictactoe import TicTacToe   # smaller game for fast tests
from alphazero.network import AlphaZeroNet
from alphazero.parallel_selfplay import (
    InferenceRequest,
    InferenceResponse,
    nn_server_loop,
)


def test_nn_server_responds_to_a_single_request():
    """Spin up an NN-server in a thread, send 1 request, get 1 response."""
    request_q = mp.Queue()
    response_qs = {0: mp.Queue()}   # one worker (id=0)
    shutdown = mp.Event()

    net = AlphaZeroNet(input_shape=(3, 3, 3), action_size=9, n_blocks=1, n_channels=4)
    net.eval()
    state_dict = {k: v.clone() for k, v in net.state_dict().items()}

    # Run server in a separate thread for testing (avoid spawn overhead)
    import threading
    t = threading.Thread(
        target=nn_server_loop,
        args=(state_dict, (3, 3, 3), 9, 1, 4, request_q, response_qs, shutdown),
        kwargs={"batch_size": 1, "wait_timeout_ms": 5, "device": "cpu"},
    )
    t.start()

    encoded = np.zeros((3, 3, 3), dtype=np.float32)
    request_q.put(InferenceRequest(worker_id=0, request_id=42, encoded=encoded))
    response: InferenceResponse = response_qs[0].get(timeout=5.0)

    assert response.request_id == 42
    assert response.priors.shape == (9,)
    assert isinstance(response.value, float)

    shutdown.set()
    t.join(timeout=5.0)


def test_worker_plays_one_complete_game():
    """A worker process plays one self-play game and returns training tuples.

    Use TicTacToe for speed. The worker uses a remote eval_fn that sends
    requests to a server thread.
    """
    import threading
    from alphazero.parallel_selfplay import worker_play_one_game

    request_q = mp.Queue()
    response_qs = {0: mp.Queue()}
    shutdown = mp.Event()

    net = AlphaZeroNet(input_shape=(3, 3, 3), action_size=9, n_blocks=1, n_channels=4)
    net.eval()
    state_dict = {k: v.clone() for k, v in net.state_dict().items()}

    server_thread = threading.Thread(
        target=nn_server_loop,
        args=(state_dict, (3, 3, 3), 9, 1, 4, request_q, response_qs, shutdown),
        kwargs={"batch_size": 4, "wait_timeout_ms": 2, "device": "cpu"},
        daemon=True,
    )
    server_thread.start()

    examples = worker_play_one_game(
        worker_id=0,
        game_name="tictactoe",
        num_simulations=5,
        temperature_threshold=6,
        c_puct=1.5,
        dirichlet_alpha=1.0,
        dirichlet_weight=0.25,
        request_q=request_q,
        response_q=response_qs[0],
        augment=True,
    )

    shutdown.set()
    server_thread.join(timeout=2.0)

    assert len(examples) > 0
    for s, pi, z in examples:
        assert s.shape == (3, 3, 3)
        assert pi.shape == (9,)
        assert z in (-1.0, 0.0, 1.0)


def test_run_parallel_self_play_returns_tuples():
    """End-to-end: spawn 2 workers, play 4 games of TTT, get all tuples back."""
    from alphazero.parallel_selfplay import run_parallel_self_play

    net = AlphaZeroNet(input_shape=(3, 3, 3), action_size=9, n_blocks=1, n_channels=4)
    state_dict = {k: v.clone() for k, v in net.state_dict().items()}

    examples = run_parallel_self_play(
        game_name="tictactoe",
        state_dict=state_dict,
        input_shape=(3, 3, 3),
        action_size=9,
        n_blocks=1,
        n_channels=4,
        num_games=4,
        num_workers=2,
        inference_batch_size=4,
        num_simulations=5,
        temperature_threshold=6,
        c_puct=1.5,
        dirichlet_alpha=1.0,
        dirichlet_weight=0.25,
        device="cpu",
    )
    assert len(examples) > 0
    for s, pi, z in examples:
        assert s.shape == (3, 3, 3)
        assert pi.shape == (9,)
        assert z in (-1.0, 0.0, 1.0)
