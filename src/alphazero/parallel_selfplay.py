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
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.multiprocessing as mp

from .network import AlphaZeroNet


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


# Game registry — extend as new games are added
def _make_game(game_name: str):
    if game_name == "tictactoe":
        from .games.tictactoe import TicTacToe
        return TicTacToe()
    if game_name == "connect4":
        from .games.connect4 import Connect4
        return Connect4()
    if game_name == "chess":
        from .games.chess_game import Chess
        return Chess()
    raise ValueError(f"Unknown game: {game_name}")


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
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Run one self-play game using a remote NN-server for inference.

    Returns a list of (state, π, z) training tuples (symmetry-augmented if
    augment=True and the game has symmetries).
    """
    from .mcts import MCTS
    from .selfplay import _assign_z

    game = _make_game(game_name)
    request_counter = [request_id_offset]

    def eval_fn(encoded: np.ndarray) -> tuple[np.ndarray, float]:
        rid = request_counter[0]
        request_counter[0] += 1
        request_q.put(InferenceRequest(worker_id=worker_id, request_id=rid, encoded=encoded))
        # Wait for the response with our exact request_id (the server returns
        # ID-tagged responses; the queue is per-worker so we just pull next)
        response: InferenceResponse = response_q.get(timeout=60.0)
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

    state = game.initial_state()
    history: list[tuple[np.ndarray, np.ndarray, int]] = []
    move_idx = 0
    while game.terminal_value(state) is None:
        pi = mcts.search(state, num_simulations=num_simulations, add_root_noise=True)
        if move_idx < temperature_threshold:
            action = int(np.random.choice(len(pi), p=pi))
        else:
            action = int(np.argmax(pi))
        canon = game.canonical_state(state)
        encoded = game.encode(canon)
        history.append((encoded, pi.astype(np.float32), game.current_player(state)))
        state = game.apply(state, action)
        move_idx += 1

    z_per_ply = _assign_z(history, game.terminal_value(state), state, game)
    examples = []
    for (encoded, pi, _player), z in zip(history, z_per_ply):
        if augment:
            for sym_enc, sym_pi in game.symmetries(encoded, pi):
                examples.append((sym_enc.astype(np.float32), sym_pi.astype(np.float32), z))
        else:
            examples.append((encoded, pi, z))
    return examples
