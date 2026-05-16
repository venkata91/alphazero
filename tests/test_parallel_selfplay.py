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
