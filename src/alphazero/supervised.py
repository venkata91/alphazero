"""Supervised pre-training of AlphaZeroNet on a Stockfish-generated corpus.

Pipeline:
    PretrainConfig (loaded from TOML)
    ├─ load_corpus() — discover shards, split into train/val
    ├─ iter_batches() — stream batches from shards
    ├─ lr_schedule() — linear warmup + cosine decay
    ├─ pretrain_supervised() — main loop with early stopping
    └─ save checkpoint in Trainer-compatible format
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class PretrainConfig:
    """Configuration for supervised pre-training.

    Mirrors TrainingConfig style (frozen dataclass, TOML-loadable) but
    holds only the knobs needed for pre-training. AZ refinement uses
    TrainingConfig + chess-refine.toml separately.
    """
    n_blocks: int
    n_channels: int

    num_epochs: int = 3
    batch_size: int = 256
    peak_lr: float = 5e-4
    end_lr: float = 5e-5
    warmup_steps: int = 500
    weight_decay: float = 1e-4

    holdout_fraction: float = 0.05
    early_stopping_patience: int = 2

    # Data loader parallelism. num_workers=0 → single-threaded synchronous reads
    # (old behavior). num_workers>0 → torch DataLoader with that many worker
    # processes prefetching batches off the critical path. pin_memory only
    # matters on CUDA (no-op on MPS/CPU).
    num_workers: int = 0
    pin_memory: bool = True

    corpus_dir: str = "data/chess_corpus"
    output_checkpoint: str = "pretrained.pt"
    log_dir: str = "runs_pretrain"

    seed: int = 42
    device: str = "auto"


def load_pretrain_config(path: Path | str) -> PretrainConfig:
    """Load PretrainConfig from a TOML file. Unknown keys raise ValueError."""
    data = tomllib.loads(Path(path).read_text())
    field_names = {f.name for f in fields(PretrainConfig)}
    unknown = set(data) - field_names
    if unknown:
        raise ValueError(f"Unknown pretrain config keys: {sorted(unknown)}")
    return PretrainConfig(**data)


import numpy as np
import torch
from typing import Iterator


def load_corpus(
    shard_dir: Path | str, holdout_fraction: float
) -> tuple[list[Path], list[Path]]:
    """Discover all shards in shard_dir and split into train/val.

    Returns (train_shards, val_shards) — both sorted by filename. The
    LAST holdout_fraction of shards (by sorted order) becomes val.
    Always reserves at least 1 shard for val.
    """
    shard_dir = Path(shard_dir)
    all_shards = sorted(shard_dir.glob("shard_*.npz"))
    if not all_shards:
        raise ValueError(f"No shards found in {shard_dir}")
    n_val = max(1, int(len(all_shards) * holdout_fraction))
    return all_shards[:-n_val], all_shards[-n_val:]


def iter_batches(
    shards: list[Path],
    batch_size: int,
    device: torch.device,
    shuffle: bool,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Yield (states, move_indices, outcomes) batches from a list of shards.

    Loads one shard at a time, optionally shuffles within the shard,
    then yields batches. The final batch may be smaller than batch_size.

    Returns tensors on `device` with dtypes:
        states: float32 (B, 20, 8, 8)
        move_indices: int64 (B,)
        outcomes: float32 (B,)
    """
    for shard_path in shards:
        data = np.load(shard_path)
        states = data["states"]
        move_indices = data["move_indices"]
        outcomes = data["outcomes"]
        n = states.shape[0]

        order = torch.randperm(n).numpy() if shuffle else np.arange(n)

        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            s = torch.from_numpy(states[idx]).to(torch.float32).to(device)
            m = torch.from_numpy(move_indices[idx]).to(torch.int64).to(device)
            z = torch.from_numpy(outcomes[idx]).to(torch.float32).to(device)
            yield s, m, z


from torch.utils.data import IterableDataset, DataLoader, get_worker_info


class _ShardDataset(IterableDataset):
    """IterableDataset yielding (state, move, z) batches from .npz shards.

    Yields CPU tensors — the consumer (or DataLoader's pin_memory path)
    handles the device transfer. Designed for multi-worker DataLoader:
    when N workers are spawned, each takes shards[worker_id::N] so there
    is no inter-worker overlap.
    """

    def __init__(self, shards: list[Path], batch_size: int, shuffle: bool, seed: int):
        self.shards = list(shards)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            my_shards = self.shards
            worker_id = 0
        else:
            my_shards = self.shards[worker_info.id :: worker_info.num_workers]
            worker_id = worker_info.id

        if self.shuffle:
            # Per-worker RNG seeded deterministically — same seed on retry,
            # different seed per worker so workers don't shuffle identically.
            import random as _random

            rng = _random.Random(self.seed + worker_id)
            my_shards = list(my_shards)
            rng.shuffle(my_shards)

        for shard_path in my_shards:
            data = np.load(shard_path)
            states = data["states"]
            move_indices = data["move_indices"]
            outcomes = data["outcomes"]
            n = states.shape[0]
            if self.shuffle:
                order = np.random.default_rng(self.seed + worker_id).permutation(n)
            else:
                order = np.arange(n)
            for start in range(0, n, self.batch_size):
                idx = order[start : start + self.batch_size]
                s = torch.from_numpy(states[idx].astype(np.float32, copy=False))
                m = torch.from_numpy(move_indices[idx].astype(np.int64, copy=False))
                z = torch.from_numpy(outcomes[idx].astype(np.float32, copy=False))
                yield s, m, z


def iter_batches_workers(
    shards: list[Path],
    batch_size: int,
    device: torch.device,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool = True,
    seed: int = 0,
) -> "Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]":
    """Multi-worker batch loader using torch DataLoader prefetching.

    Workers read shards from disk + cast int8 → float32 off the critical
    path; main thread handles only the CPU→GPU transfer. pin_memory only
    affects CUDA — MPS and CPU ignore it.

    For num_workers=0 callers should use iter_batches instead — this
    function still works at num_workers=0 (synchronous mode) but adds
    needless DataLoader overhead.
    """
    dataset = _ShardDataset(shards, batch_size, shuffle, seed=seed)
    # pin_memory is meaningful only on CUDA; suppress otherwise to avoid warnings
    effective_pin = pin_memory and device.type == "cuda"
    loader = DataLoader(
        dataset,
        batch_size=None,  # _ShardDataset already yields batches
        num_workers=num_workers,
        pin_memory=effective_pin,
        persistent_workers=num_workers > 0,
    )
    for s, m, z in loader:
        s = s.to(device, non_blocking=effective_pin)
        m = m.to(device, non_blocking=effective_pin)
        z = z.to(device, non_blocking=effective_pin)
        yield s, m, z


def make_batch_iter(
    shards: list[Path],
    batch_size: int,
    device: torch.device,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = True,
    seed: int = 0,
) -> "Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]":
    """Dispatch to iter_batches or iter_batches_workers based on num_workers.

    num_workers == 0 → single-threaded iter_batches (lowest overhead)
    num_workers >  0 → DataLoader with that many prefetch workers
    """
    if num_workers <= 0:
        return iter_batches(shards, batch_size, device, shuffle)
    return iter_batches_workers(
        shards, batch_size, device, shuffle, num_workers, pin_memory, seed
    )


import math


def lr_schedule(
    *,
    step: int,
    warmup_steps: int,
    total_steps: int,
    peak_lr: float,
    end_lr: float,
) -> float:
    """Linear warmup over warmup_steps, then cosine decay to end_lr at total_steps.

    During warmup (step < warmup_steps):
        lr = peak_lr * (step + 1) / warmup_steps

    After warmup (step >= warmup_steps):
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        lr = end_lr + 0.5 * (peak_lr - end_lr) * (1 + cos(pi * progress))
    """
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(progress, 1.0)
    return end_lr + 0.5 * (peak_lr - end_lr) * (1 + math.cos(math.pi * progress))


import torch.nn.functional as F

from .network import AlphaZeroNet


def pretrain_step(
    net: AlphaZeroNet,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    lr: float,
) -> tuple[float, float, float]:
    """One SGD step on (states, move_indices, outcomes). Returns scalar losses.

    Policy loss = cross-entropy(logits, move_indices) — the target is the
    one-hot of Stockfish's chosen move (passed as integer index).
    Value loss = MSE(predicted_value, outcomes) — outcomes are the game
    result from each position's mover POV.
    """
    states, move_indices, outcomes = batch
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    optimizer.zero_grad()
    logits, values = net(states)
    policy_loss = F.cross_entropy(logits, move_indices)
    value_loss = F.mse_loss(values, outcomes)
    total = policy_loss + value_loss
    total.backward()
    optimizer.step()
    return total.item(), policy_loss.item(), value_loss.item()


def compute_val_loss(
    net: AlphaZeroNet,
    val_shards: list[Path],
    batch_size: int,
    device: torch.device,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> float:
    """Compute average (policy + value) loss over val_shards without gradients."""
    net.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.inference_mode():
        for states, move_indices, outcomes in make_batch_iter(
            val_shards,
            batch_size=batch_size,
            device=device,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ):
            logits, values = net(states)
            policy_loss = F.cross_entropy(logits, move_indices, reduction="sum")
            value_loss = F.mse_loss(values, outcomes, reduction="sum")
            total_loss += (policy_loss + value_loss).item()
            total_samples += states.shape[0]
    net.train()
    if total_samples == 0:
        return float("nan")
    return total_loss / total_samples


import copy
import time
from dataclasses import asdict


def _resolve_device(device_str: str) -> torch.device:
    """Shared with Trainer via alphazero.device — see that module."""
    from .device import resolve_device

    return resolve_device(device_str)


def pretrain_supervised(config: PretrainConfig) -> None:
    """Main pre-training loop.

    Loads corpus, trains for up to config.num_epochs with early stopping
    on val loss, saves checkpoint in Trainer-compatible format.
    """
    torch.manual_seed(config.seed)
    device = _resolve_device(config.device)

    train_shards, val_shards = load_corpus(config.corpus_dir, config.holdout_fraction)
    print(
        f"Pre-training: {len(train_shards)} train shards, "
        f"{len(val_shards)} val shards on device={device}",
        flush=True,
    )

    sample_shard = np.load(train_shards[0])
    positions_per_shard = sample_shard["states"].shape[0]
    total_train_positions = positions_per_shard * len(train_shards)
    steps_per_epoch = max(1, total_train_positions // config.batch_size)
    total_steps = steps_per_epoch * config.num_epochs

    net = AlphaZeroNet(
        input_shape=(20, 8, 8),
        action_size=4672,
        n_blocks=config.n_blocks,
        n_channels=config.n_channels,
    ).to(device)
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=config.peak_lr, weight_decay=config.weight_decay
    )

    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0
    lr = config.peak_lr

    for epoch in range(1, config.num_epochs + 1):
        net.train()
        epoch_start = time.time()
        epoch_loss_sum = 0.0
        epoch_samples = 0

        for batch in make_batch_iter(
            train_shards,
            config.batch_size,
            device,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            seed=config.seed + epoch,
        ):
            lr = lr_schedule(
                step=global_step,
                warmup_steps=config.warmup_steps,
                total_steps=total_steps,
                peak_lr=config.peak_lr,
                end_lr=config.end_lr,
            )
            total_loss, _, _ = pretrain_step(net, batch, optimizer, lr=lr)
            epoch_loss_sum += total_loss * batch[0].shape[0]
            epoch_samples += batch[0].shape[0]
            global_step += 1

        train_loss = epoch_loss_sum / max(1, epoch_samples)
        val_loss = compute_val_loss(
            net,
            val_shards,
            config.batch_size,
            device,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )
        elapsed = time.time() - epoch_start

        # Save checkpoint BEFORE printing the epoch summary, so that
        # whenever a user sees "epoch N/M done", the file on disk is
        # guaranteed to reflect that epoch's weights (no race window
        # between print and save flush).
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            patience_counter = 0
            _save_pretrained_checkpoint(net, optimizer, config, epoch)
        else:
            patience_counter += 1

        saved_marker = " (checkpoint saved)" if improved else ""
        print(
            f"  epoch {epoch}/{config.num_epochs}: "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"lr={lr:.2e} elapsed={elapsed:.1f}s{saved_marker}",
            flush=True,
        )

        if not improved and patience_counter >= config.early_stopping_patience:
            print(
                f"  early stopping: val_loss did not improve for "
                f"{patience_counter} epochs (best={best_val_loss:.4f})",
                flush=True,
            )
            break

    print(f"Done. best_val_loss={best_val_loss:.4f}", flush=True)


def _save_pretrained_checkpoint(
    net: AlphaZeroNet,
    optimizer: torch.optim.Optimizer,
    config: PretrainConfig,
    epoch: int,
) -> None:
    """Save checkpoint in Trainer-compatible format.

    Trainer expects keys: iteration, config, best_net, candidate_net, optimizer.
    We populate both best_net and candidate_net with the pretrained weights so
    that --resume-from loads them as the starting best_net.
    """
    output = Path(config.output_checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    sd = net.state_dict()
    torch.save({
        "iteration": 0,
        "config": asdict(config),
        "best_net": sd,
        "candidate_net": copy.deepcopy(sd),
        "optimizer": optimizer.state_dict(),
        "_pretrain_epoch": epoch,
    }, output)
