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
