"""Parallel self-play with central NN-server batching inference.

Architecture:
    N worker processes each run MCTS + Game logic on different self-play games.
    1 central NN-server process owns the GPU and batches inference requests
    from workers. Workers send (encoded_state, request_id) to a shared request
    queue; the server pulls batches up to `batch_size`, runs one forward pass,
    and routes (priors, value) back to each worker's reply queue.

This decouples Python-bound MCTS work (parallelized across CPU cores) from
GPU-bound NN inference (batched for utilization). Required for chess at
scale — serial self-play is wall-time infeasible.
"""
from __future__ import annotations

import queue
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.multiprocessing as mp

from .network import AlphaZeroNet


class WorkerHangError(RuntimeError):
    """Raised when a worker has not sent a heartbeat within the configured
    timeout. Carries the offending worker_id and elapsed silence (seconds)."""

    def __init__(self, worker_id: int, silent_for_s: float, heartbeat_timeout_s: float):
        self.worker_id = worker_id
        self.silent_for_s = silent_for_s
        self.heartbeat_timeout_s = heartbeat_timeout_s
        super().__init__(
            f"Worker {worker_id} has not sent a heartbeat for "
            f"{silent_for_s:.1f}s (timeout={heartbeat_timeout_s:.1f}s); "
            f"considering it hung."
        )


@dataclass
class InferenceRequest:
    worker_id: int
    request_id: int
    encoded: np.ndarray   # shape: (C, H, W); the canonical encoded state


@dataclass
class InferenceResponse:
    request_id: int
    priors: np.ndarray    # shape: (action_size,)
    value: float          # scalar


def nn_server_loop(
    state_dict: dict,
    input_shape: tuple[int, int, int],
    action_size: int,
    n_blocks: int,
    n_channels: int,
    request_q: mp.Queue,
    response_qs: dict[int, mp.Queue],
    shutdown_event: Any,                    # mp.Event or threading.Event
    *,
    batch_size: int = 64,
    wait_timeout_ms: int = 5,
    device: str = "cpu",
) -> None:
    """Run as a thread or process: serve batched inference until shutdown_event is set.

    Workers send InferenceRequest objects to request_q. The server collects
    up to `batch_size` requests OR waits `wait_timeout_ms`, runs one batched
    forward pass, and routes InferenceResponse back via response_qs[worker_id].
    """
    dev = torch.device(device)
    net = AlphaZeroNet(
        input_shape=input_shape,
        action_size=action_size,
        n_blocks=n_blocks,
        n_channels=n_channels,
    )
    net.load_state_dict(state_dict)
    net.to(dev)
    net.eval()

    wait_timeout_s = wait_timeout_ms / 1000.0
    while not shutdown_event.is_set():
        batch: list[InferenceRequest] = []
        # Block briefly for first request; non-blocking for the rest
        try:
            first = request_q.get(timeout=wait_timeout_s)
            batch.append(first)
        except queue.Empty:
            continue
        while len(batch) < batch_size:
            try:
                req = request_q.get_nowait()
                batch.append(req)
            except queue.Empty:
                break

        # Stack and run
        x = torch.from_numpy(
            np.stack([r.encoded for r in batch]).astype(np.float32, copy=False)
        ).to(dev)
        with torch.inference_mode():
            logits, values = net(x)
            priors = torch.softmax(logits, dim=-1).cpu().numpy()
            values = values.cpu().numpy()

        # Route back
        for i, req in enumerate(batch):
            response_qs[req.worker_id].put(
                InferenceResponse(
                    request_id=req.request_id,
                    priors=priors[i].astype(np.float32),
                    value=float(values[i]),
                )
            )


# Game-name → Game instance. Single source of truth in games/__init__.py.
def _make_game(game_name: str):
    from .games import make_game

    return make_game(game_name)


def worker_play_one_game(
    *,
    worker_id: int,
    game_name: str,
    num_simulations: int,
    temperature_threshold: int,
    c_puct: float,
    dirichlet_alpha: float,
    dirichlet_weight: float,
    request_q: mp.Queue,
    response_q: mp.Queue,
    augment: bool = True,
    request_id_offset: int = 0,
    heartbeat_q: Any = None,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Run one self-play game using a remote NN-server for inference.

    Returns a list of (state, π, z) training tuples (symmetry-augmented if
    augment=True and the game has symmetries).
    """
    from .mcts import MCTS
    from .selfplay import play_one_selfplay_game

    game = _make_game(game_name)
    request_counter = [request_id_offset]

    def eval_fn(encoded: np.ndarray) -> tuple[np.ndarray, float]:
        rid = request_counter[0]
        request_counter[0] += 1
        request_q.put(InferenceRequest(worker_id=worker_id, request_id=rid, encoded=encoded))
        # Wait for the response with our exact request_id (the server returns
        # ID-tagged responses; the queue is per-worker so we just pull next)
        response: InferenceResponse = response_q.get(timeout=300.0)
        assert response.request_id == rid, (
            f"Worker {worker_id} expected request_id {rid}, got {response.request_id}"
        )
        return response.priors, response.value

    mcts = MCTS(
        game, eval_fn,
        c_puct=c_puct,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_weight=dirichlet_weight,
    )

    on_step = None
    if heartbeat_q is not None:
        def on_step() -> None:
            try:
                heartbeat_q.put((worker_id, time.monotonic()))
            except Exception:
                pass

    return play_one_selfplay_game(
        game,
        mcts,
        num_simulations=num_simulations,
        temperature_threshold=temperature_threshold,
        augment=augment,
        on_step=on_step,
    )


def run_parallel_self_play(
    *,
    game_name: str,
    state_dict: dict,
    input_shape: tuple[int, int, int],
    action_size: int,
    n_blocks: int,
    n_channels: int,
    num_games: int,
    num_workers: int,
    inference_batch_size: int,
    num_simulations: int,
    temperature_threshold: int,
    c_puct: float,
    dirichlet_alpha: float,
    dirichlet_weight: float,
    device: str = "cpu",
    heartbeat_timeout_s: float = 120.0,
    poll_interval_s: float = 5.0,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Run num_games self-play games across num_workers processes.

    Spawns one NN-server thread (owns the model) and num_workers worker
    processes (each runs one game at a time until num_games are done).
    Workers report back via a result queue; orchestrator collects.

    Workers also emit a (worker_id, monotonic_timestamp) heartbeat at the
    start and end of every game. The parent polls result_q on a short
    interval (poll_interval_s) and checks heartbeats; any worker silent
    for more than heartbeat_timeout_s is treated as hung and raises
    WorkerHangError after attempting shutdown.

    Returns a flat list of (state, π, z) training tuples.
    """
    import threading

    # Use spawn (safer with CUDA + cleaner than fork)
    ctx = mp.get_context("spawn")
    request_q = ctx.Queue()
    response_qs = {wid: ctx.Queue() for wid in range(num_workers)}
    result_q = ctx.Queue()
    heartbeat_q = ctx.Queue()
    work_q = ctx.Queue()   # holds "play one game" job tokens

    # Pre-load work queue with num_games jobs (just dummy tokens)
    for game_idx in range(num_games):
        work_q.put(game_idx)
    # Sentinel: each worker exits when it pulls one
    for _ in range(num_workers):
        work_q.put(None)

    shutdown_event = ctx.Event()

    # Start the NN-server in a thread (in this process — server uses GPU)
    server_thread = threading.Thread(
        target=nn_server_loop,
        args=(state_dict, input_shape, action_size, n_blocks, n_channels,
              request_q, response_qs, shutdown_event),
        kwargs={"batch_size": inference_batch_size,
                "wait_timeout_ms": 5, "device": device},
        daemon=True,
    )
    server_thread.start()

    # Seed heartbeats to "now" so we don't false-positive before workers
    # have a chance to send their first ping (spawn startup can take >1s).
    now = time.monotonic()
    last_heartbeat: dict[int, float] = {wid: now for wid in range(num_workers)}

    # Start workers
    workers = []
    for wid in range(num_workers):
        p = ctx.Process(
            target=_worker_loop,
            args=(wid, game_name, num_simulations, temperature_threshold,
                  c_puct, dirichlet_alpha, dirichlet_weight,
                  work_q, request_q, response_qs[wid], result_q,
                  heartbeat_q, shutdown_event),
        )
        p.start()
        workers.append(p)

    # Collect num_games batches of training tuples
    all_examples: list[tuple[np.ndarray, np.ndarray, float]] = []
    try:
        _collect_with_heartbeat_monitor(
            num_games=num_games,
            result_q=result_q,
            heartbeat_q=heartbeat_q,
            last_heartbeat=last_heartbeat,
            heartbeat_timeout_s=heartbeat_timeout_s,
            poll_interval_s=poll_interval_s,
            sink=all_examples.extend,
        )
    finally:
        # Always shutdown — even if collection raised
        shutdown_event.set()
        for p in workers:
            p.join(timeout=10.0)
            if p.is_alive():
                p.terminate()
                p.join(timeout=2.0)
        server_thread.join(timeout=5.0)

    return all_examples


def _collect_with_heartbeat_monitor(
    *,
    num_games: int,
    result_q: Any,
    heartbeat_q: Any,
    last_heartbeat: dict[int, float],
    heartbeat_timeout_s: float,
    poll_interval_s: float,
    sink,
) -> None:
    """Collect num_games results from result_q while monitoring heartbeats.

    Calls `sink(batch)` for each batch pulled from result_q. Raises
    WorkerHangError if any worker is silent for more than
    heartbeat_timeout_s before all games are collected. Exits cleanly
    (no raise) once all num_games batches are collected.
    """
    games_collected = 0
    while games_collected < num_games:
        # Short-poll result_q so we can check heartbeats frequently.
        try:
            batch = result_q.get(timeout=poll_interval_s)
            sink(batch)
            games_collected += 1
        except queue.Empty:
            pass

        # Drain any pending heartbeats.
        while True:
            try:
                wid, ts = heartbeat_q.get_nowait()
            except queue.Empty:
                break
            # Keep only the most recent timestamp per worker.
            if ts > last_heartbeat.get(wid, 0.0):
                last_heartbeat[wid] = ts

        # If we've already collected everything, exit happy regardless
        # of any worker that may still be winding down.
        if games_collected >= num_games:
            return

        # Hang check: any worker silent for too long?
        now = time.monotonic()
        for wid, last in last_heartbeat.items():
            silent_for = now - last
            if silent_for > heartbeat_timeout_s:
                err = WorkerHangError(
                    worker_id=wid,
                    silent_for_s=silent_for,
                    heartbeat_timeout_s=heartbeat_timeout_s,
                )
                print(
                    f"[parallel_selfplay] {err}",
                    file=sys.stderr,
                    flush=True,
                )
                raise err


def _worker_loop(
    worker_id: int,
    game_name: str,
    num_simulations: int,
    temperature_threshold: int,
    c_puct: float,
    dirichlet_alpha: float,
    dirichlet_weight: float,
    work_q: mp.Queue,
    request_q: mp.Queue,
    response_q: mp.Queue,
    result_q: mp.Queue,
    heartbeat_q: mp.Queue,
    shutdown_event: Any,
) -> None:
    """Worker entry point. Pulls 'play a game' tokens from work_q, runs one
    game using worker_play_one_game, pushes results back via result_q. Exits
    when it pulls a None sentinel or when shutdown_event is set.

    Emits a (worker_id, monotonic_timestamp) heartbeat on heartbeat_q at the
    start and end of every game so the parent can detect hangs."""
    request_id_offset = worker_id * 1_000_000  # avoid request_id collisions across workers

    # Initial heartbeat: announce we're alive immediately after spawn.
    try:
        heartbeat_q.put((worker_id, time.monotonic()))
    except Exception:
        pass

    while True:
        if shutdown_event.is_set():
            break
        token = work_q.get()
        if token is None:
            break

        # Heartbeat: about to start a game.
        try:
            heartbeat_q.put((worker_id, time.monotonic()))
        except Exception:
            pass

        try:
            examples = worker_play_one_game(
                worker_id=worker_id,
                game_name=game_name,
                num_simulations=num_simulations,
                temperature_threshold=temperature_threshold,
                c_puct=c_puct,
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_weight=dirichlet_weight,
                request_q=request_q,
                response_q=response_q,
                augment=True,
                request_id_offset=request_id_offset,
                heartbeat_q=heartbeat_q,
            )
        except (queue.Empty, AssertionError) as e:
            # NN inference server didn't respond (response_q timeout) or
            # request_id desync. Skip this game so the iteration doesn't stall,
            # but log clearly so the user sees the real cause rather than a
            # downstream WorkerHangError.
            sys.stderr.write(
                f"worker {worker_id}: game failed ({type(e).__name__}: {e}); "
                f"skipping. Likely NN inference timeout — check parent-side "
                f"MPS/GPU saturation.\n"
            )
            sys.stderr.flush()
            # Drain any late-arriving responses for the failed request_ids
            # so the next game doesn't immediately assertion-fail on stale data.
            while True:
                try:
                    response_q.get_nowait()
                except queue.Empty:
                    break
            examples = []
        request_id_offset += 100_000   # bump for next game from this worker

        # Heartbeat: game done, about to publish.
        try:
            heartbeat_q.put((worker_id, time.monotonic()))
        except Exception:
            pass

        result_q.put(examples)
