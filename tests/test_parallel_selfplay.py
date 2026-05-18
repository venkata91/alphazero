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


def test_run_parallel_self_play_happy_path_with_heartbeat():
    """Happy path with tight heartbeat budget: 2 workers x 2 games of TTT
    finishes well within heartbeat_timeout_s and never raises."""
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
        heartbeat_timeout_s=30.0,
        poll_interval_s=0.5,
    )
    assert len(examples) > 0


def test_collect_with_heartbeat_monitor_raises_on_hung_worker():
    """Unit test the monitor: feed it a heartbeat_q where one worker is
    silent and verify WorkerHangError is raised promptly with the right
    worker_id and message."""
    import queue as stdlib_queue
    import time

    from alphazero.parallel_selfplay import (
        WorkerHangError,
        _collect_with_heartbeat_monitor,
    )

    # Plain Queue is fine — the helper only uses .get(timeout=) / .get_nowait().
    result_q = stdlib_queue.Queue()
    heartbeat_q = stdlib_queue.Queue()

    # Seed two workers as alive recently; worker 1 will stay silent.
    now = time.monotonic()
    last_heartbeat = {0: now, 1: now}

    # Worker 0 keeps signaling; worker 1 stays silent. Send a heartbeat
    # for worker 0 so its clock keeps advancing past the timeout.
    heartbeat_q.put((0, now + 0.05))
    heartbeat_q.put((0, now + 0.10))

    t0 = time.monotonic()
    with pytest.raises(WorkerHangError) as exc_info:
        _collect_with_heartbeat_monitor(
            num_games=10,                # never reached
            result_q=result_q,            # always empty
            heartbeat_q=heartbeat_q,
            last_heartbeat=last_heartbeat,
            heartbeat_timeout_s=0.5,
            poll_interval_s=0.05,
            sink=lambda b: None,
        )
    elapsed = time.monotonic() - t0

    assert exc_info.value.worker_id == 1
    assert "Worker 1" in str(exc_info.value)
    # Should detect well within a few seconds.
    assert elapsed < 5.0, f"hang detection took too long: {elapsed:.2f}s"


def test_collect_with_heartbeat_monitor_happy_path_no_raise():
    """Monitor returns cleanly when all num_games results arrive, even if a
    worker would otherwise be considered hung (no late-stage false positives)."""
    import queue as stdlib_queue
    import time

    from alphazero.parallel_selfplay import _collect_with_heartbeat_monitor

    result_q = stdlib_queue.Queue()
    heartbeat_q = stdlib_queue.Queue()

    # Pre-fill exactly num_games results so the loop never has to wait long.
    result_q.put([("game0_tuple",)])
    result_q.put([("game1_tuple",)])

    # Seed worker as alive but don't update — should not matter since we
    # collect everything before any hang check fires (num_games met).
    last_heartbeat = {0: time.monotonic()}

    collected: list = []
    _collect_with_heartbeat_monitor(
        num_games=2,
        result_q=result_q,
        heartbeat_q=heartbeat_q,
        last_heartbeat=last_heartbeat,
        heartbeat_timeout_s=60.0,
        poll_interval_s=0.05,
        sink=collected.extend,
    )
    assert len(collected) == 2


def _hanging_worker_target(worker_id, heartbeat_q, shutdown_event):
    """Top-level target for spawn: sends one heartbeat then sleeps until
    shutdown_event is set (or 60s elapse). Used by the end-to-end hang test."""
    import time as _time
    heartbeat_q.put((worker_id, _time.monotonic()))
    # Hang here. The parent should detect this and set shutdown_event.
    shutdown_event.wait(timeout=60.0)


def test_end_to_end_hang_detection_with_real_spawned_worker():
    """Spawn a real process that emits exactly one heartbeat then hangs.
    The parent monitor should raise WorkerHangError within a few seconds."""
    import time
    import torch.multiprocessing as mp

    from alphazero.parallel_selfplay import (
        WorkerHangError,
        _collect_with_heartbeat_monitor,
    )

    ctx = mp.get_context("spawn")
    result_q = ctx.Queue()
    heartbeat_q = ctx.Queue()
    shutdown_event = ctx.Event()

    p = ctx.Process(
        target=_hanging_worker_target,
        args=(0, heartbeat_q, shutdown_event),
    )
    p.start()

    last_heartbeat = {0: time.monotonic()}

    try:
        t0 = time.monotonic()
        with pytest.raises(WorkerHangError) as exc_info:
            _collect_with_heartbeat_monitor(
                num_games=1,
                result_q=result_q,
                heartbeat_q=heartbeat_q,
                last_heartbeat=last_heartbeat,
                heartbeat_timeout_s=2.0,
                poll_interval_s=0.2,
                sink=lambda b: None,
            )
        elapsed = time.monotonic() - t0
        assert exc_info.value.worker_id == 0
        assert "Worker 0" in str(exc_info.value)
        # Detection bound: a 2s timeout plus a poll interval and a bit of slack.
        assert elapsed < 30.0, f"hang detection took too long: {elapsed:.2f}s"
    finally:
        shutdown_event.set()
        p.join(timeout=5.0)
        if p.is_alive():
            p.terminate()
            p.join(timeout=2.0)
