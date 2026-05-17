# Supervised Pre-Training + AlphaZero Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement supervised pre-training of `AlphaZeroNet` on a corpus of Stockfish-vs-Stockfish games, then refine with a short AlphaZero loop, to reach ≥50% win rate vs Stockfish ELO=1500 in ~15 hours wall-clock on M4 Pro (vs 13 days for cold-start AZ).

**Architecture:** Three phases producing three checkpoints — (1) embarrassingly parallel corpus generation via 10 worker processes each running their own Stockfish subprocess at `Threads=1`, (2) supervised pre-training with held-out validation + early stopping + lr warmup/cosine decay, (3) AZ refinement (10 iters × 50 sims) starting from the pretrained checkpoint via existing `--resume-from` flag.

**Tech Stack:** Same as SP3 + `multiprocessing` (already imported elsewhere) + numpy `.npz` shards for corpus storage. All framework code (`AlphaZeroNet`, `Chess(Game)`, `chess_move_encoding`, `StockfishOpponent`, `parallel_selfplay`, `Trainer.run()`) reused unchanged.

**Spec:** [`docs/superpowers/specs/2026-05-16-supervised-pretraining-design.md`](../specs/2026-05-16-supervised-pretraining-design.md)

---

## File Structure

```
scripts/generate_chess_corpus.py        ← new (~180 lines) CLI + orchestrator
src/alphazero/corpus.py                 ← new (~150 lines) worker, single-game, shard writer
src/alphazero/supervised.py             ← new (~200 lines) load_corpus, lr_schedule, pretrain_supervised, PretrainConfig
src/alphazero/cli.py                    ← extend (~40 lines) new `pretrain` subcommand
configs/chess-pretrain.toml             ← new
configs/chess-refine.toml               ← new
tests/test_chess_corpus.py              ← new (~120 lines)
tests/test_supervised.py                ← new (~150 lines)
tests/test_cli.py                       ← extend (~25 lines) regression test for pretrain wiring
tests/test_e2e_supervised.py            ← new (~80 lines, @pytest.mark.slow)
```

Tasks proceed in dependency order: corpus primitives → corpus orchestrator → corpus generator CLI → small corpus smoke test → supervised loaders → lr schedule → pretrain step/epoch → early stopping + main loop → pretrain CLI subcommand → configs → refinement config + verify --resume-from → E2E test → final sweep.

---

## Conventions

- Commit messages: HEREDOC with `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` footer.
- TDD: red → green → commit. Each task has its own commit.
- All paths absolute from repo root unless noted.
- Run pytest from repo root: `cd /Users/vsowrira/git/alphazero && python3 -m pytest <args>`.

---

## Task 1: Position type + random opening helper

**Files:**
- Create: `src/alphazero/corpus.py`
- Create: `tests/test_chess_corpus.py`

- [ ] **Step 1: Write failing tests**

Create `/Users/vsowrira/git/alphazero/tests/test_chess_corpus.py`:

```python
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
    # Walk through the move stack and verify each is legal-at-its-time
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
    # At least 2 of the 5 should differ
    assert len(set(boards)) >= 2


def test_random_opening_moves_stops_if_game_ends():
    """If the game ends mid-opening (shouldn't happen, but defensive), stop early."""
    # Construct a position where black is checkmated and only legal "move" is none
    board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert board.is_checkmate()
    rng = random.Random(42)
    random_opening_moves(board, n_plies=4, rng=rng)
    # No moves should have been pushed since game is over
    assert len(board.move_stack) == 0
```

- [ ] **Step 2: Verify failure**

```bash
cd /Users/vsowrira/git/alphazero && python3 -m pytest tests/test_chess_corpus.py -v
```

Expected: ImportError on `alphazero.corpus`.

- [ ] **Step 3: Implement `src/alphazero/corpus.py`**

Create `/Users/vsowrira/git/alphazero/src/alphazero/corpus.py`:

```python
"""Chess corpus generation primitives.

Stockfish-vs-Stockfish game generation for supervised pre-training of
AlphaZeroNet. Each worker process plays games independently and writes
its own .npz shards.

A Position is a (encoded_state, move_index, z) tuple where:
    encoded_state: np.ndarray of shape (20, 8, 8), dtype int8
        — Chess.encode(board) cast to int8 for compact storage
    move_index: int in [0, 4672)
        — chess_move_encoding.move_to_index(board, move)
    z: int in {-1, 0, +1}
        — game outcome from the mover's POV at that position
"""
from __future__ import annotations

import random
from typing import NamedTuple

import chess
import numpy as np


class Position(NamedTuple):
    encoded_state: np.ndarray   # (20, 8, 8) int8
    move_index: int             # [0, 4672)
    z: int                      # {-1, 0, +1}


def random_opening_moves(
    board: chess.Board, n_plies: int, rng: random.Random
) -> None:
    """Play n_plies uniformly-random legal moves into the board (in place).

    Stops early if the game ends before n_plies (defensive; rare).
    """
    for _ in range(n_plies):
        if board.is_game_over(claim_draw=True):
            return
        legal = list(board.legal_moves)
        move = rng.choice(legal)
        board.push(move)
```

- [ ] **Step 4: Verify tests pass**

```bash
cd /Users/vsowrira/git/alphazero && python3 -m pytest tests/test_chess_corpus.py -v
```

Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/vsowrira/git/alphazero
git add src/alphazero/corpus.py tests/test_chess_corpus.py
git commit -m "$(cat <<'EOF'
Add corpus.py with Position type + random_opening_moves

Foundation for supervised pre-training corpus generation. Position is
the (encoded_state, move_index, z) tuple stored per training sample.
random_opening_moves provides game diversity by playing 4 uniformly
random legal plies before Stockfish takes over — without this, all
Stockfish-vs-Stockfish games would share the same opening line.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Single-game generator

**Files:**
- Modify: `src/alphazero/corpus.py`
- Modify: `tests/test_chess_corpus.py`

- [ ] **Step 1: Append failing tests**

Add to `/Users/vsowrira/git/alphazero/tests/test_chess_corpus.py`:

```python
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

    assert len(positions) > 4   # at least the random opening + some Stockfish moves
    for p in positions:
        assert p.encoded_state.shape == (20, 8, 8)
        assert p.encoded_state.dtype == np.int8
        assert 0 <= p.move_index < 4672
        assert p.z in (-1, 0, 1)


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
        # Draw: all z = 0
        assert all(z == 0 for z in z_values)
    else:
        # Decisive: adjacent positions have opposite signs (or both zero, but not here)
        for i in range(len(z_values) - 1):
            assert z_values[i] == -z_values[i + 1], (
                f"At ply {i}: z={z_values[i]} but next z={z_values[i+1]}, "
                f"expected alternation in decisive game"
            )
```

- [ ] **Step 2: Verify failure**

```bash
cd /Users/vsowrira/git/alphazero && python3 -m pytest tests/test_chess_corpus.py -v
```

Expected: ImportError on `play_one_game`.

- [ ] **Step 3: Implement `play_one_game` in `src/alphazero/corpus.py`**

Append to `/Users/vsowrira/git/alphazero/src/alphazero/corpus.py`:

```python
import chess.engine

from .games.chess_game import Chess
from .games.chess_move_encoding import move_to_index


def play_one_game(
    engine: chess.engine.SimpleEngine,
    time_per_move: float,
    rng: random.Random,
    *,
    opening_plies: int = 4,
) -> list[Position]:
    """Play one Stockfish-vs-Stockfish game and return Position tuples.

    Flow:
        1. Random opening (4 plies) for diversity
        2. Stockfish plays both sides until game ends
        3. For each recorded position, compute z from the final outcome
           in the position-mover's POV
    """
    game = Chess()
    board = chess.Board()
    random_opening_moves(board, n_plies=opening_plies, rng=rng)

    # Record (encoded_state, move_index, mover_player) for each ply
    history: list[tuple[np.ndarray, int, int]] = []

    while not board.is_game_over(claim_draw=True):
        result = engine.play(board, chess.engine.Limit(time=time_per_move))
        move = result.move
        if move is None:
            # Defensive: engine.play normally returns a move; if not, stop
            break
        encoded = game.encode(board).astype(np.int8)
        mv_idx = move_to_index(board, move)
        mover_player = 1 if board.turn == chess.WHITE else -1
        history.append((encoded, mv_idx, mover_player))
        board.push(move)

    # Determine final outcome from the white-POV signed value
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        white_pov_z = 0
    elif outcome.winner == chess.WHITE:
        white_pov_z = 1
    else:
        white_pov_z = -1

    # Build Position tuples — z is from the mover's POV at that position
    positions = []
    for encoded, mv_idx, mover_player in history:
        z_from_mover_pov = white_pov_z * mover_player
        positions.append(Position(
            encoded_state=encoded,
            move_index=mv_idx,
            z=int(z_from_mover_pov),
        ))
    return positions
```

- [ ] **Step 4: Verify tests pass**

```bash
cd /Users/vsowrira/git/alphazero && python3 -m pytest tests/test_chess_corpus.py -v
```

Expected: 6 tests pass (4 from Task 1 + 2 new).

- [ ] **Step 5: Commit**

```bash
cd /Users/vsowrira/git/alphazero
git add src/alphazero/corpus.py tests/test_chess_corpus.py
git commit -m "$(cat <<'EOF'
Add play_one_game: Stockfish vs Stockfish single-game generator

Plays one full game using a Stockfish engine for both sides, after a
random 4-ply opening. Records (encoded_state, move_index, z) tuples
with z assigned from each ply's mover POV — alternating ±1 in decisive
games, 0 throughout in draws. The encoded_state is cast to int8 for
compact storage in the corpus shards.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Shard writer

**Files:**
- Modify: `src/alphazero/corpus.py`
- Modify: `tests/test_chess_corpus.py`

- [ ] **Step 1: Append failing tests**

Add to `/Users/vsowrira/git/alphazero/tests/test_chess_corpus.py`:

```python
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
```

- [ ] **Step 2: Verify failure**

```bash
cd /Users/vsowrira/git/alphazero && python3 -m pytest tests/test_chess_corpus.py -v
```

Expected: ImportError on `write_shard`.

- [ ] **Step 3: Implement `write_shard` in `src/alphazero/corpus.py`**

Append to `/Users/vsowrira/git/alphazero/src/alphazero/corpus.py`:

```python
from pathlib import Path


def write_shard(
    output_dir: Path,
    worker_id: int,
    shard_counter: int,
    positions: list[Position],
) -> Path:
    """Write a list of Positions to a compressed .npz shard.

    Filename format: shard_w{worker_id}_s{shard_counter:04d}.npz
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_path = output_dir / f"shard_w{worker_id}_s{shard_counter:04d}.npz"

    states = np.stack([p.encoded_state for p in positions]).astype(np.int8)
    move_indices = np.array([p.move_index for p in positions], dtype=np.int32)
    outcomes = np.array([p.z for p in positions], dtype=np.int8)

    np.savez_compressed(
        shard_path,
        states=states,
        move_indices=move_indices,
        outcomes=outcomes,
    )
    return shard_path
```

- [ ] **Step 4: Verify tests pass**

```bash
cd /Users/vsowrira/git/alphazero && python3 -m pytest tests/test_chess_corpus.py -v
```

Expected: 8 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/vsowrira/git/alphazero
git add src/alphazero/corpus.py tests/test_chess_corpus.py
git commit -m "$(cat <<'EOF'
Add write_shard: compressed .npz writer for corpus shards

Shards use int8 states + int32 move indices + int8 outcomes for
compact on-disk size (~13MB compressed per 10k positions). Filename
format shard_w{wid}_s{counter:04d}.npz lets each worker write
independently without filename collisions.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Worker entry point

**Files:**
- Modify: `src/alphazero/corpus.py`
- Modify: `tests/test_chess_corpus.py`

- [ ] **Step 1: Append failing test**

Add to `/Users/vsowrira/git/alphazero/tests/test_chess_corpus.py`:

```python
def test_worker_generate_writes_shards(tmp_path):
    """A worker should produce at least one shard for a small game count."""
    from alphazero.corpus import worker_generate

    worker_generate(
        worker_id=0,
        n_games=3,
        output_dir=tmp_path,
        time_per_move=0.01,
        seed=42,
        shard_size=10_000,   # large so we get exactly one shard at the end
    )

    shards = sorted(tmp_path.glob("shard_w0_*.npz"))
    assert len(shards) >= 1
    # Total positions across all shards should be > 0
    total_positions = sum(np.load(s)["states"].shape[0] for s in shards)
    assert total_positions > 6   # ~3 games × ~few plies each minimum
```

- [ ] **Step 2: Verify failure**

```bash
cd /Users/vsowrira/git/alphazero && python3 -m pytest tests/test_chess_corpus.py::test_worker_generate_writes_shards -v
```

Expected: ImportError on `worker_generate`.

- [ ] **Step 3: Implement `worker_generate` in `src/alphazero/corpus.py`**

Append to `/Users/vsowrira/git/alphazero/src/alphazero/corpus.py`:

```python
def worker_generate(
    worker_id: int,
    n_games: int,
    output_dir: Path,
    time_per_move: float,
    seed: int,
    *,
    shard_size: int = 10_000,
    stockfish_path: str = "stockfish",
) -> None:
    """Worker process entry point. Plays n_games and writes shards.

    Each worker is fully independent — owns its own Stockfish subprocess,
    its own RNG, its own shard counter, writes to its own filenames.
    """
    rng = random.Random(seed + worker_id)
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"Threads": 1, "Hash": 16})

    shard_buffer: list[Position] = []
    shard_counter = 0

    try:
        for _ in range(n_games):
            positions = play_one_game(engine, time_per_move=time_per_move, rng=rng)
            shard_buffer.extend(positions)

            while len(shard_buffer) >= shard_size:
                write_shard(output_dir, worker_id, shard_counter, shard_buffer[:shard_size])
                shard_buffer = shard_buffer[shard_size:]
                shard_counter += 1

        # Flush remainder
        if shard_buffer:
            write_shard(output_dir, worker_id, shard_counter, shard_buffer)
    finally:
        try:
            engine.quit()
        except (chess.engine.EngineTerminatedError, BrokenPipeError):
            pass
```

- [ ] **Step 4: Verify tests pass**

```bash
cd /Users/vsowrira/git/alphazero && python3 -m pytest tests/test_chess_corpus.py -v
```

Expected: 9 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/vsowrira/git/alphazero
git add src/alphazero/corpus.py tests/test_chess_corpus.py
git commit -m "$(cat <<'EOF'
Add worker_generate: per-worker game-playing loop

Each worker owns its own Stockfish subprocess (configured Threads=1 to
prevent thread oversubscription across workers) and its own RNG seeded
by base_seed + worker_id. Flushes shards every shard_size positions
plus a final remainder shard at the end. Defensive engine.quit() in a
finally block ensures the subprocess is reaped even on worker failure.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Orchestrator (multi-process spawn + join)

**Files:**
- Modify: `src/alphazero/corpus.py`
- Modify: `tests/test_chess_corpus.py`

- [ ] **Step 1: Append failing test**

Add to `/Users/vsowrira/git/alphazero/tests/test_chess_corpus.py`:

```python
def test_generate_corpus_spawns_workers_and_distributes_games(tmp_path):
    """End-to-end: 2 workers each play 2 games; shards exist for both workers."""
    from alphazero.corpus import generate_corpus

    generate_corpus(
        target_games=4,
        num_workers=2,
        output_dir=tmp_path,
        time_per_move=0.01,
        seed=42,
    )

    # Both workers should have produced at least one shard each
    w0_shards = sorted(tmp_path.glob("shard_w0_*.npz"))
    w1_shards = sorted(tmp_path.glob("shard_w1_*.npz"))
    assert len(w0_shards) >= 1
    assert len(w1_shards) >= 1
```

- [ ] **Step 2: Verify failure**

```bash
cd /Users/vsowrira/git/alphazero && python3 -m pytest tests/test_chess_corpus.py::test_generate_corpus_spawns_workers_and_distributes_games -v
```

Expected: ImportError on `generate_corpus`.

- [ ] **Step 3: Implement `generate_corpus` in `src/alphazero/corpus.py`**

Append to `/Users/vsowrira/git/alphazero/src/alphazero/corpus.py`:

```python
import multiprocessing as mp


def generate_corpus(
    *,
    target_games: int,
    num_workers: int,
    output_dir: Path,
    time_per_move: float,
    seed: int,
    shard_size: int = 10_000,
    stockfish_path: str = "stockfish",
) -> None:
    """Spawn num_workers processes, distribute target_games across them.

    Embarrassingly parallel — workers don't communicate. Each writes its
    own shards to output_dir.
    """
    ctx = mp.get_context("spawn")
    games_per_worker = target_games // num_workers
    remainder = target_games % num_workers

    processes = []
    for wid in range(num_workers):
        # Distribute the remainder games to the first `remainder` workers
        n = games_per_worker + (1 if wid < remainder else 0)
        if n == 0:
            continue
        p = ctx.Process(
            target=worker_generate,
            args=(wid, n, output_dir, time_per_move, seed),
            kwargs={"shard_size": shard_size, "stockfish_path": stockfish_path},
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(
                f"Worker process exited with code {p.exitcode}; "
                f"check stderr for traceback"
            )
```

- [ ] **Step 4: Verify tests pass**

```bash
cd /Users/vsowrira/git/alphazero && python3 -m pytest tests/test_chess_corpus.py -v
```

Expected: 10 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/vsowrira/git/alphazero
git add src/alphazero/corpus.py tests/test_chess_corpus.py
git commit -m "$(cat <<'EOF'
Add generate_corpus orchestrator: spawn N workers, partition work

Spawn context (CUDA-safe, also avoids fork issues on macOS). Games are
partitioned via integer division + remainder so the total never deviates
from target_games. Each worker is fully independent — no IPC, no shared
state, no queues. Orchestrator just joins and propagates non-zero exit
codes as RuntimeError.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: CLI script for corpus generation

**Files:**
- Create: `scripts/generate_chess_corpus.py`

- [ ] **Step 1: Create the CLI script**

Create `/Users/vsowrira/git/alphazero/scripts/generate_chess_corpus.py`:

```python
#!/usr/bin/env python3
"""Generate a Stockfish-vs-Stockfish corpus for supervised pre-training.

Usage:
    python3 scripts/generate_chess_corpus.py \\
        --num-games 50000 \\
        --num-workers 10 \\
        --output data/chess_corpus/ \\
        --time-per-move 0.05 \\
        --seed 42
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from alphazero.corpus import generate_corpus


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-games", type=int, required=True,
                        help="Total games to generate across all workers")
    parser.add_argument("--num-workers", type=int, default=10,
                        help="Number of parallel worker processes (default: 10)")
    parser.add_argument("--output", type=Path, default=Path("data/chess_corpus"),
                        help="Output directory for shard files")
    parser.add_argument("--time-per-move", type=float, default=0.05,
                        help="Stockfish thinking time per move in seconds (default: 0.05)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base RNG seed; per-worker seed = base + worker_id")
    parser.add_argument("--shard-size", type=int, default=10_000,
                        help="Positions per shard file (default: 10000)")
    parser.add_argument("--stockfish-path", type=str, default="stockfish",
                        help="Path to stockfish binary (default: 'stockfish' on PATH)")
    args = parser.parse_args()

    start = time.time()
    print(
        f"Generating {args.num_games} games with {args.num_workers} workers; "
        f"time_per_move={args.time_per_move}s; output={args.output}",
        flush=True,
    )

    generate_corpus(
        target_games=args.num_games,
        num_workers=args.num_workers,
        output_dir=args.output,
        time_per_move=args.time_per_move,
        seed=args.seed,
        shard_size=args.shard_size,
        stockfish_path=args.stockfish_path,
    )

    elapsed = time.time() - start
    shards = sorted(args.output.glob("shard_*.npz"))
    print(
        f"Done. {len(shards)} shards written in {elapsed:.1f}s "
        f"({elapsed/60:.1f} min).",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Make it executable**

```bash
cd /Users/vsowrira/git/alphazero && chmod +x scripts/generate_chess_corpus.py
```

- [ ] **Step 3: Smoke-test it with a tiny corpus**

```bash
cd /Users/vsowrira/git/alphazero
rm -rf /tmp/test_corpus
python3 scripts/generate_chess_corpus.py --num-games 4 --num-workers 2 --output /tmp/test_corpus --time-per-move 0.01
ls /tmp/test_corpus/
```

Expected: prints `Done. N shards written...` and `ls` shows at least 2 shard files (`shard_w0_*.npz`, `shard_w1_*.npz`).

- [ ] **Step 4: Commit**

```bash
cd /Users/vsowrira/git/alphazero
git add scripts/generate_chess_corpus.py
git commit -m "$(cat <<'EOF'
Add scripts/generate_chess_corpus.py: CLI entry point for corpus gen

Wraps corpus.generate_corpus with argparse. Defaults tuned for M4 Pro
(10 workers, 0.05s/move). Prints summary line at end with shard count
and wall-clock for monitoring overnight runs.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: PretrainConfig dataclass + TOML loader

**Files:**
- Create: `src/alphazero/supervised.py`
- Create: `tests/test_supervised.py`

- [ ] **Step 1: Write failing tests**

Create `/Users/vsowrira/git/alphazero/tests/test_supervised.py`:

```python
"""Tests for supervised pre-training."""
from __future__ import annotations

from pathlib import Path

import pytest


def test_pretrain_config_defaults():
    """PretrainConfig has expected default values."""
    from alphazero.supervised import PretrainConfig

    cfg = PretrainConfig(n_blocks=4, n_channels=8)
    assert cfg.n_blocks == 4
    assert cfg.n_channels == 8
    assert cfg.num_epochs == 3
    assert cfg.batch_size == 256
    assert cfg.peak_lr == 5e-4
    assert cfg.end_lr == 5e-5
    assert cfg.warmup_steps == 500
    assert cfg.weight_decay == 1e-4
    assert cfg.holdout_fraction == 0.05
    assert cfg.early_stopping_patience == 2
    assert cfg.corpus_dir == "data/chess_corpus"
    assert cfg.output_checkpoint == "pretrained.pt"
    assert cfg.log_dir == "runs_pretrain"
    assert cfg.seed == 42
    assert cfg.device == "auto"


def test_pretrain_config_loads_from_toml(tmp_path):
    """load_pretrain_config reads a TOML file and constructs a PretrainConfig."""
    from alphazero.supervised import load_pretrain_config

    toml_path = tmp_path / "test.toml"
    toml_path.write_text("""
n_blocks = 10
n_channels = 64
num_epochs = 5
batch_size = 128
peak_lr = 1e-3
""")
    cfg = load_pretrain_config(toml_path)
    assert cfg.n_blocks == 10
    assert cfg.n_channels == 64
    assert cfg.num_epochs == 5
    assert cfg.batch_size == 128
    assert cfg.peak_lr == 1e-3
    # Defaults preserved for unspecified fields
    assert cfg.warmup_steps == 500


def test_pretrain_config_rejects_unknown_keys(tmp_path):
    """Unknown TOML keys raise ValueError (typo protection)."""
    from alphazero.supervised import load_pretrain_config

    toml_path = tmp_path / "bad.toml"
    toml_path.write_text("""
n_blocks = 4
n_channels = 8
typo_field = 999
""")
    with pytest.raises(ValueError, match="Unknown"):
        load_pretrain_config(toml_path)
```

- [ ] **Step 2: Verify failure**

```bash
cd /Users/vsowrira/git/alphazero && python3 -m pytest tests/test_supervised.py -v
```

Expected: ImportError on `alphazero.supervised`.

- [ ] **Step 3: Implement `src/alphazero/supervised.py`**

Create `/Users/vsowrira/git/alphazero/src/alphazero/supervised.py`:

```python
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
    # Network — must match what the refinement phase will load
    n_blocks: int
    n_channels: int

    # Training
    num_epochs: int = 3
    batch_size: int = 256
    peak_lr: float = 5e-4
    end_lr: float = 5e-5
    warmup_steps: int = 500
    weight_decay: float = 1e-4

    # Validation
    holdout_fraction: float = 0.05
    early_stopping_patience: int = 2

    # I/O
    corpus_dir: str = "data/chess_corpus"
    output_checkpoint: str = "pretrained.pt"
    log_dir: str = "runs_pretrain"

    # Misc
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
```

- [ ] **Step 4: Verify tests pass**

```bash
cd /Users/vsowrira/git/alphazero && python3 -m pytest tests/test_supervised.py -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/vsowrira/git/alphazero
git add src/alphazero/supervised.py tests/test_supervised.py
git commit -m "$(cat <<'EOF'
Add PretrainConfig + load_pretrain_config

Frozen dataclass mirroring TrainingConfig style. Holds only the
pretraining-specific knobs (epochs, lr schedule, holdout, early
stopping). Refinement uses TrainingConfig separately with
chess-refine.toml — keeping the two config types disjoint avoids
forcing pretrain-only fields into TrainingConfig.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: load_corpus + iter_batches

**Files:**
- Modify: `src/alphazero/supervised.py`
- Modify: `tests/test_supervised.py`

- [ ] **Step 1: Append failing tests**

Add to `/Users/vsowrira/git/alphazero/tests/test_supervised.py`:

```python
import numpy as np
import torch


def _write_fake_shard(path: Path, n_positions: int, seed: int):
    """Helper: write a fake shard with random data of the right shapes/dtypes."""
    rng = np.random.default_rng(seed)
    np.savez_compressed(
        path,
        states=rng.integers(0, 2, size=(n_positions, 20, 8, 8), dtype=np.int8),
        move_indices=rng.integers(0, 4672, size=(n_positions,), dtype=np.int32),
        outcomes=rng.choice([-1, 0, 1], size=(n_positions,)).astype(np.int8),
    )


def test_load_corpus_splits_shards_into_train_and_val(tmp_path):
    """load_corpus discovers all shards and reserves holdout_fraction for val."""
    from alphazero.supervised import load_corpus

    # Create 20 fake shards
    for i in range(20):
        _write_fake_shard(tmp_path / f"shard_w0_s{i:04d}.npz", 100, seed=i)

    train, val = load_corpus(tmp_path, holdout_fraction=0.1)
    assert len(train) == 18
    assert len(val) == 2
    # Train shards come before val shards (by sorted filename)
    assert all(t.name < v.name for t in train for v in val)


def test_load_corpus_minimum_one_val_shard(tmp_path):
    """Even with a tiny corpus, at least 1 shard is held out."""
    from alphazero.supervised import load_corpus

    for i in range(3):
        _write_fake_shard(tmp_path / f"shard_w0_s{i:04d}.npz", 100, seed=i)

    train, val = load_corpus(tmp_path, holdout_fraction=0.01)   # would round to 0
    assert len(val) >= 1


def test_iter_batches_yields_correct_shape_and_dtype(tmp_path):
    """iter_batches reads shards and yields (states, moves, z) tensors."""
    from alphazero.supervised import iter_batches

    _write_fake_shard(tmp_path / "shard_w0_s0000.npz", 100, seed=0)
    _write_fake_shard(tmp_path / "shard_w0_s0001.npz", 100, seed=1)
    shards = sorted(tmp_path.glob("*.npz"))

    batches = list(iter_batches(shards, batch_size=32, device=torch.device("cpu"), shuffle=False))
    # 200 positions / 32 = 6 full + 1 partial = 7 batches
    assert len(batches) == 7

    for states, moves, zs in batches[:6]:
        assert states.shape == (32, 20, 8, 8)
        assert states.dtype == torch.float32
        assert moves.shape == (32,)
        assert moves.dtype == torch.int64
        assert zs.shape == (32,)
        assert zs.dtype == torch.float32
    # Last batch has 200 - 192 = 8 positions
    last_states, last_moves, last_zs = batches[-1]
    assert last_states.shape == (8, 20, 8, 8)


def test_iter_batches_shuffles_when_requested(tmp_path):
    """shuffle=True produces a different order than shuffle=False."""
    from alphazero.supervised import iter_batches

    _write_fake_shard(tmp_path / "shard_w0_s0000.npz", 100, seed=0)
    shards = sorted(tmp_path.glob("*.npz"))

    torch.manual_seed(42)
    no_shuffle = next(iter(iter_batches(shards, batch_size=100, device=torch.device("cpu"), shuffle=False)))
    torch.manual_seed(42)
    yes_shuffle = next(iter(iter_batches(shards, batch_size=100, device=torch.device("cpu"), shuffle=True)))

    # Move indices should differ (with overwhelming probability)
    assert not torch.equal(no_shuffle[1], yes_shuffle[1])
```

- [ ] **Step 2: Verify failure**

```bash
cd /Users/vsowrira/git/alphazero && python3 -m pytest tests/test_supervised.py -v
```

Expected: ImportError on `load_corpus`, `iter_batches`.

- [ ] **Step 3: Implement `load_corpus` and `iter_batches` in `src/alphazero/supervised.py`**

Append to `/Users/vsowrira/git/alphazero/src/alphazero/supervised.py`:

```python
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
        states = data["states"]               # int8 (N, 20, 8, 8)
        move_indices = data["move_indices"]   # int32 (N,)
        outcomes = data["outcomes"]           # int8 (N,)
        n = states.shape[0]

        order = torch.randperm(n).numpy() if shuffle else np.arange(n)

        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            s = torch.from_numpy(states[idx]).to(torch.float32).to(device)
            m = torch.from_numpy(move_indices[idx]).to(torch.int64).to(device)
            z = torch.from_numpy(outcomes[idx]).to(torch.float32).to(device)
            yield s, m, z
```

- [ ] **Step 4: Verify tests pass**

```bash
cd /Users/vsowrira/git/alphazero && python3 -m pytest tests/test_supervised.py -v
```

Expected: 7 tests pass (3 from Task 7 + 4 new).

- [ ] **Step 5: Commit**

```bash
cd /Users/vsowrira/git/alphazero
git add src/alphazero/supervised.py tests/test_supervised.py
git commit -m "$(cat <<'EOF'
Add load_corpus + iter_batches for shard streaming

load_corpus splits sorted shard filenames into train/val using
holdout_fraction. iter_batches reads one shard at a time, optionally
shuffles within the shard, casts states to float32, and yields batches
on the target device. Streaming (vs loading all shards at once) keeps
memory bounded to ~13MB per shard regardless of corpus size.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: lr_schedule (warmup + cosine decay)

**Files:**
- Modify: `src/alphazero/supervised.py`
- Modify: `tests/test_supervised.py`

- [ ] **Step 1: Append failing tests**

Add to `/Users/vsowrira/git/alphazero/tests/test_supervised.py`:

```python
import math


def test_lr_schedule_warmup_ramps_linearly_from_zero():
    """During warmup, lr ramps linearly from ~0 to peak_lr."""
    from alphazero.supervised import lr_schedule

    # warmup over 100 steps, total 1000 steps
    lr_at_0 = lr_schedule(step=0, warmup_steps=100, total_steps=1000, peak_lr=1e-3, end_lr=1e-4)
    lr_at_50 = lr_schedule(step=50, warmup_steps=100, total_steps=1000, peak_lr=1e-3, end_lr=1e-4)
    lr_at_99 = lr_schedule(step=99, warmup_steps=100, total_steps=1000, peak_lr=1e-3, end_lr=1e-4)

    assert lr_at_0 == pytest.approx(1e-5, rel=0.01)     # step+1=1, 1/100 of peak
    assert lr_at_50 == pytest.approx(0.51 * 1e-3, rel=0.01)
    assert lr_at_99 == pytest.approx(1.0 * 1e-3, rel=0.01)


def test_lr_schedule_after_warmup_starts_at_peak():
    """At step == warmup_steps, lr equals peak_lr."""
    from alphazero.supervised import lr_schedule

    lr = lr_schedule(step=100, warmup_steps=100, total_steps=1000, peak_lr=1e-3, end_lr=1e-4)
    assert lr == pytest.approx(1e-3, rel=0.01)


def test_lr_schedule_at_total_steps_reaches_end_lr():
    """At step == total_steps, lr equals end_lr (cosine has fully decayed)."""
    from alphazero.supervised import lr_schedule

    lr = lr_schedule(step=1000, warmup_steps=100, total_steps=1000, peak_lr=1e-3, end_lr=1e-4)
    assert lr == pytest.approx(1e-4, rel=0.01)


def test_lr_schedule_monotonic_decay_after_warmup():
    """After warmup, lr only decreases."""
    from alphazero.supervised import lr_schedule

    lrs = [
        lr_schedule(step=s, warmup_steps=100, total_steps=1000, peak_lr=1e-3, end_lr=1e-4)
        for s in range(100, 1001, 50)
    ]
    for i in range(len(lrs) - 1):
        assert lrs[i] >= lrs[i + 1] - 1e-9   # allow tiny float error
```

- [ ] **Step 2: Verify failure**

```bash
cd /Users/vsowrira/git/alphazero && python3 -m pytest tests/test_supervised.py -v
```

Expected: ImportError on `lr_schedule`.

- [ ] **Step 3: Implement `lr_schedule` in `src/alphazero/supervised.py`**

Append to `/Users/vsowrira/git/alphazero/src/alphazero/supervised.py`:

```python
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
        lr = end_lr + 0.5 * (peak_lr - end_lr) * (1 + cos(π * progress))
    """
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(progress, 1.0)
    return end_lr + 0.5 * (peak_lr - end_lr) * (1 + math.cos(math.pi * progress))
```

- [ ] **Step 4: Verify tests pass**

```bash
cd /Users/vsowrira/git/alphazero && python3 -m pytest tests/test_supervised.py -v
```

Expected: 11 tests pass (7 from prior + 4 new).

- [ ] **Step 5: Commit**

```bash
cd /Users/vsowrira/git/alphazero
git add src/alphazero/supervised.py tests/test_supervised.py
git commit -m "$(cat <<'EOF'
Add lr_schedule: linear warmup + cosine decay

Standard schedule for pre-training larger networks. Warmup prevents
Adam's noisy first-step gradient estimates from blowing up the lr ×
random-gradient product. Cosine decay smoothly reduces lr toward end_lr
in the late phase where the optimizer is fine-tuning around a minimum.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Pretrain step + epoch runner

**Files:**
- Modify: `src/alphazero/supervised.py`
- Modify: `tests/test_supervised.py`

- [ ] **Step 1: Append failing test**

Add to `/Users/vsowrira/git/alphazero/tests/test_supervised.py`:

```python
def test_pretrain_step_returns_losses_and_decreases_loss():
    """One pretrain step on a tiny net + tiny batch should produce a finite loss
    and reduce the loss when run twice on the same batch (overfitting check)."""
    from alphazero.network import AlphaZeroNet
    from alphazero.supervised import pretrain_step

    net = AlphaZeroNet(input_shape=(20, 8, 8), action_size=4672, n_blocks=1, n_channels=8)
    net.train()
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)

    states = torch.zeros((4, 20, 8, 8), dtype=torch.float32)
    moves = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    zs = torch.tensor([1.0, -1.0, 0.0, 1.0], dtype=torch.float32)

    total_1, policy_1, value_1 = pretrain_step(net, (states, moves, zs), optimizer, lr=1e-3)
    total_2, policy_2, value_2 = pretrain_step(net, (states, moves, zs), optimizer, lr=1e-3)

    assert all(math.isfinite(x) for x in (total_1, policy_1, value_1, total_2, policy_2, value_2))
    # On the same batch, loss should decrease (model overfits trivially)
    assert total_2 < total_1


def test_compute_val_loss_runs_without_grad(tmp_path):
    """compute_val_loss should evaluate a model on a corpus without producing gradients."""
    from alphazero.network import AlphaZeroNet
    from alphazero.supervised import compute_val_loss

    _write_fake_shard(tmp_path / "shard_w0_s0000.npz", 64, seed=0)
    shards = [tmp_path / "shard_w0_s0000.npz"]

    net = AlphaZeroNet(input_shape=(20, 8, 8), action_size=4672, n_blocks=1, n_channels=8)
    net.eval()

    val_loss = compute_val_loss(net, shards, batch_size=16, device=torch.device("cpu"))
    assert math.isfinite(val_loss)
    assert val_loss > 0
```

- [ ] **Step 2: Verify failure**

```bash
cd /Users/vsowrira/git/alphazero && python3 -m pytest tests/test_supervised.py -v
```

Expected: ImportError on `pretrain_step`, `compute_val_loss`.

- [ ] **Step 3: Implement in `src/alphazero/supervised.py`**

Append to `/Users/vsowrira/git/alphazero/src/alphazero/supervised.py`:

```python
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
) -> float:
    """Compute average (policy + value) loss over val_shards without gradients."""
    net.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.inference_mode():
        for states, move_indices, outcomes in iter_batches(
            val_shards, batch_size=batch_size, device=device, shuffle=False
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
```

- [ ] **Step 4: Verify tests pass**

```bash
cd /Users/vsowrira/git/alphazero && python3 -m pytest tests/test_supervised.py -v
```

Expected: 13 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/vsowrira/git/alphazero
git add src/alphazero/supervised.py tests/test_supervised.py
git commit -m "$(cat <<'EOF'
Add pretrain_step + compute_val_loss

pretrain_step does one SGD update with the same loss structure as
Trainer._train_step (CE policy + MSE value, equal weight) — but with
hard one-hot policy targets (Stockfish's chosen move) instead of MCTS
visit distribution. Per-step lr is set from the schedule by the caller.

compute_val_loss runs the same loss in inference mode without gradient
updates — used for early stopping. Returns per-sample average so runs
with different val sizes are comparable.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: pretrain_supervised main loop (early stopping + checkpoint)

**Files:**
- Modify: `src/alphazero/supervised.py`
- Modify: `tests/test_supervised.py`

- [ ] **Step 1: Append failing test**

Add to `/Users/vsowrira/git/alphazero/tests/test_supervised.py`:

```python
def test_pretrain_supervised_writes_checkpoint(tmp_path):
    """End-to-end on a tiny corpus: pretrain runs and writes a checkpoint."""
    from dataclasses import replace
    from alphazero.supervised import PretrainConfig, pretrain_supervised

    # Make a tiny corpus
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    for i in range(3):
        _write_fake_shard(corpus_dir / f"shard_w0_s{i:04d}.npz", 64, seed=i)

    ckpt_path = tmp_path / "test_pretrained.pt"

    cfg = PretrainConfig(
        n_blocks=1,
        n_channels=8,
        num_epochs=1,
        batch_size=32,
        warmup_steps=2,
        holdout_fraction=0.34,    # 1 of 3 shards → val
        early_stopping_patience=10,
        corpus_dir=str(corpus_dir),
        output_checkpoint=str(ckpt_path),
        log_dir=str(tmp_path / "runs"),
        device="cpu",
    )

    pretrain_supervised(cfg)

    assert ckpt_path.exists()
    # Verify checkpoint format matches what Trainer.load_from_checkpoint expects
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert set(ckpt.keys()) >= {"iteration", "config", "best_net", "candidate_net", "optimizer"}
    assert ckpt["iteration"] == 0


def test_pretrain_supervised_early_stops_when_val_loss_diverges(tmp_path, monkeypatch):
    """If val_loss never improves, early stopping triggers."""
    from alphazero.supervised import PretrainConfig, pretrain_supervised
    import alphazero.supervised as sup

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    for i in range(3):
        _write_fake_shard(corpus_dir / f"shard_w0_s{i:04d}.npz", 64, seed=i)

    # Force compute_val_loss to return increasing values → triggers patience
    counter = [0.5]
    def fake_val_loss(*args, **kwargs):
        counter[0] += 1.0
        return counter[0]
    monkeypatch.setattr(sup, "compute_val_loss", fake_val_loss)

    cfg = PretrainConfig(
        n_blocks=1,
        n_channels=8,
        num_epochs=10,
        batch_size=32,
        warmup_steps=2,
        holdout_fraction=0.34,
        early_stopping_patience=2,
        corpus_dir=str(corpus_dir),
        output_checkpoint=str(tmp_path / "p.pt"),
        log_dir=str(tmp_path / "runs"),
        device="cpu",
    )
    pretrain_supervised(cfg)

    # Even though num_epochs=10, training stops within 3 epochs due to patience=2
    # (first call sets best_val_loss, then 2 worsening calls trip patience)
    assert counter[0] <= 4.0   # at most 3 epochs ran
```

- [ ] **Step 2: Verify failure**

```bash
cd /Users/vsowrira/git/alphazero && python3 -m pytest tests/test_supervised.py -v
```

Expected: ImportError on `pretrain_supervised`.

- [ ] **Step 3: Implement `pretrain_supervised` in `src/alphazero/supervised.py`**

Append to `/Users/vsowrira/git/alphazero/src/alphazero/supervised.py`:

```python
import copy
import time
from dataclasses import asdict


def _resolve_device(device_str: str) -> torch.device:
    """Mirrors Trainer._resolve_device for consistency."""
    if device_str == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_str)


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

    # Estimate total steps for the lr schedule (one-pass estimate; not exact when
    # shards have variable sizes, but close enough for the cosine schedule)
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

    for epoch in range(1, config.num_epochs + 1):
        net.train()
        epoch_start = time.time()
        epoch_loss_sum = 0.0
        epoch_samples = 0

        for batch in iter_batches(train_shards, config.batch_size, device, shuffle=True):
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
        val_loss = compute_val_loss(net, val_shards, config.batch_size, device)
        elapsed = time.time() - epoch_start

        print(
            f"  epoch {epoch}/{config.num_epochs}: "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"lr={lr:.2e} elapsed={elapsed:.1f}s",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            _save_pretrained_checkpoint(net, optimizer, config, epoch)
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
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
```

- [ ] **Step 4: Verify tests pass**

```bash
cd /Users/vsowrira/git/alphazero && python3 -m pytest tests/test_supervised.py -v
```

Expected: 15 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/vsowrira/git/alphazero
git add src/alphazero/supervised.py tests/test_supervised.py
git commit -m "$(cat <<'EOF'
Add pretrain_supervised main loop with early stopping

Trains for up to config.num_epochs, evaluates val_loss after each
epoch, saves checkpoint only when val_loss improves, and stops early
when val_loss stops improving for config.early_stopping_patience
epochs. Saves checkpoints in Trainer-compatible format (keys:
iteration, config, best_net, candidate_net, optimizer) so the AZ
refinement phase can load via --resume-from without modification.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: CLI `pretrain` subcommand

**Files:**
- Modify: `src/alphazero/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Append failing test**

Add to `/Users/vsowrira/git/alphazero/tests/test_cli.py`:

```python
def test_cmd_pretrain_calls_pretrain_supervised(tmp_path, monkeypatch):
    """Regression test: `pretrain` subcommand wires config + calls pretrain_supervised."""
    import argparse
    from unittest.mock import patch
    from alphazero.cli import _cmd_pretrain

    monkeypatch.chdir(tmp_path)

    # Write a minimal pretrain config
    cfg_path = tmp_path / "p.toml"
    cfg_path.write_text("""
n_blocks = 1
n_channels = 8
num_epochs = 1
batch_size = 32
corpus_dir = "fake_corpus"
output_checkpoint = "pretrained.pt"
log_dir = "runs"
device = "cpu"
""")

    captured = {}
    def fake_pretrain(config):
        captured["config"] = config

    with patch("alphazero.supervised.pretrain_supervised", fake_pretrain):
        args = argparse.Namespace(config=cfg_path)
        exit_code = _cmd_pretrain(args)

    assert exit_code == 0
    assert captured["config"].n_blocks == 1
    assert captured["config"].batch_size == 32
```

- [ ] **Step 2: Verify failure**

```bash
cd /Users/vsowrira/git/alphazero && python3 -m pytest tests/test_cli.py::test_cmd_pretrain_calls_pretrain_supervised -v
```

Expected: ImportError on `_cmd_pretrain`.

- [ ] **Step 3: Add `_cmd_pretrain` to `src/alphazero/cli.py`**

Edit `/Users/vsowrira/git/alphazero/src/alphazero/cli.py`. Find the end of `_cmd_play` (around line 208 — search for `return 0` after the final print statement of the play command). Add this new function AFTER `_cmd_play`:

```python
def _cmd_pretrain(args: argparse.Namespace) -> int:
    """Run supervised pre-training using a PretrainConfig from TOML."""
    from .supervised import load_pretrain_config, pretrain_supervised

    config = load_pretrain_config(args.config)
    pretrain_supervised(config)
    return 0
```

Then find the parser setup (around line 213 — search for `subs = parser.add_subparsers`). After the existing `p_play.set_defaults(func=_cmd_play)` line, add:

```python
    p_pretrain = subs.add_parser("pretrain", help="Supervised pre-training on a corpus")
    p_pretrain.add_argument("--config", type=Path, required=True,
                            help="Path to a chess-pretrain.toml file")
    p_pretrain.set_defaults(func=_cmd_pretrain)
```

- [ ] **Step 4: Verify tests pass**

```bash
cd /Users/vsowrira/git/alphazero && python3 -m pytest tests/test_cli.py -v
```

Expected: all CLI tests pass (existing + new test).

- [ ] **Step 5: Commit**

```bash
cd /Users/vsowrira/git/alphazero
git add src/alphazero/cli.py tests/test_cli.py
git commit -m "$(cat <<'EOF'
Add `pretrain` CLI subcommand

python -m alphazero pretrain --config configs/chess-pretrain.toml
Loads a PretrainConfig from TOML and runs pretrain_supervised. Mirrors
the existing train/eval/play subcommand pattern. Regression test
follows the same monkeypatch-on-the-implementation approach used for
the chess eval-opponent wiring.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: configs/chess-pretrain.toml

**Files:**
- Create: `configs/chess-pretrain.toml`

- [ ] **Step 1: Create the config**

Create `/Users/vsowrira/git/alphazero/configs/chess-pretrain.toml`:

```toml
# Chess supervised pre-training config — Sub-project 3b Phase 2.
# Run with: python -m alphazero pretrain --config configs/chess-pretrain.toml
# Reads corpus from data/chess_corpus/ (produced by scripts/generate_chess_corpus.py).
# Writes pretrained.pt for use by AZ refinement (--resume-from pretrained.pt).

# Network — must match configs/chess-refine.toml so --resume-from works
n_blocks = 15
n_channels = 192

# Training
num_epochs = 3
batch_size = 256
peak_lr = 5e-4
end_lr = 5e-5
warmup_steps = 500
weight_decay = 1e-4

# Validation + early stopping
holdout_fraction = 0.05
early_stopping_patience = 2

# I/O
corpus_dir = "data/chess_corpus"
output_checkpoint = "pretrained.pt"
log_dir = "runs_pretrain"

# Misc
seed = 42
device = "auto"
```

- [ ] **Step 2: Verify the config loads**

```bash
cd /Users/vsowrira/git/alphazero && python3 -c "from alphazero.supervised import load_pretrain_config; c = load_pretrain_config('configs/chess-pretrain.toml'); print(f'n_blocks={c.n_blocks} n_channels={c.n_channels} epochs={c.num_epochs} peak_lr={c.peak_lr}')"
```

Expected: prints `n_blocks=15 n_channels=192 epochs=3 peak_lr=0.0005`.

- [ ] **Step 3: Commit**

```bash
cd /Users/vsowrira/git/alphazero
git add configs/chess-pretrain.toml
git commit -m "$(cat <<'EOF'
Add configs/chess-pretrain.toml

15M-param net (n_blocks=15, n_channels=192), 3 epochs max with early
stopping (patience=2), lr 5e-4 with 500-step warmup and cosine decay
to 5e-5. Tuned for the ~3M-position corpus produced by 50K
Stockfish-vs-Stockfish games. Network shape matches chess-refine.toml
so the pretrained checkpoint loads cleanly via --resume-from.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: configs/chess-refine.toml

**Files:**
- Create: `configs/chess-refine.toml`

- [ ] **Step 1: Create the config**

Create `/Users/vsowrira/git/alphazero/configs/chess-refine.toml`:

```toml
# Chess AZ refinement config — Sub-project 3b Phase 3.
# Run with:
#   python -m alphazero train \
#     --game chess \
#     --config configs/chess-refine.toml \
#     --resume-from pretrained.pt

# Network — must match configs/chess-pretrain.toml (the pretrained checkpoint shape)
n_blocks = 15
n_channels = 192

# MCTS — reduced from chess.toml since pretrained priors compensate
num_simulations = 50
c_puct = 2.5
dirichlet_alpha = 0.3
dirichlet_weight = 0.25

# Self-play
games_per_iteration = 100
temperature_threshold = 30

# Training
training_steps_per_iteration = 2000
batch_size = 256
learning_rate = 1e-3
weight_decay = 1e-4

# Replay buffer — smaller since the pretrained net doesn't need a big ramp
replay_buffer_capacity = 200_000
min_buffer_size = 10_000

# Eval (vs Stockfish ELO=1500)
eval_interval = 2
eval_games = 100

# Schedule
num_iterations = 10

# Parallel self-play
num_workers = 10
inference_batch_size = 64

# Misc
seed = 42
device = "auto"
checkpoint_dir = "checkpoints_refine"
log_dir = "runs_refine"
```

- [ ] **Step 2: Verify the config loads**

```bash
cd /Users/vsowrira/git/alphazero && python3 -c "from alphazero.config import load_config; c = load_config('configs/chess-refine.toml'); print(f'n_blocks={c.n_blocks} n_channels={c.n_channels} num_iterations={c.num_iterations} num_workers={c.num_workers}')"
```

Expected: prints `n_blocks=15 n_channels=192 num_iterations=10 num_workers=10`.

- [ ] **Step 3: Commit**

```bash
cd /Users/vsowrira/git/alphazero
git add configs/chess-refine.toml
git commit -m "$(cat <<'EOF'
Add configs/chess-refine.toml

Reduced AZ knobs for the post-pretrain refinement phase: 10
iterations × 100 games × 50 sims (vs chess.toml's 80 × 100 × 200).
Pretrained priors mean MCTS converges faster, so fewer sims per move
are sufficient. 10 workers matches M4 Pro P-core count. Network shape
(n_blocks=15, n_channels=192) matches pretrain config so pretrained.pt
loads via --resume-from.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 15: Verify --resume-from loads pretrained checkpoint (integration smoke test)

**Files:**
- Modify: `tests/test_supervised.py`

- [ ] **Step 1: Append failing test**

Add to `/Users/vsowrira/git/alphazero/tests/test_supervised.py`:

```python
def test_pretrained_checkpoint_loads_in_trainer_via_resume(tmp_path):
    """A pretrained.pt should be loadable by Trainer.load_from_checkpoint."""
    from dataclasses import replace
    from alphazero.config import TrainingConfig
    from alphazero.games.chess_game import Chess
    from alphazero.supervised import PretrainConfig, pretrain_supervised
    from alphazero.trainer import Trainer

    # Generate a fake corpus
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    for i in range(3):
        _write_fake_shard(corpus_dir / f"shard_w0_s{i:04d}.npz", 64, seed=i)

    # Pretrain
    ckpt_path = tmp_path / "pretrained.pt"
    pre_cfg = PretrainConfig(
        n_blocks=1,
        n_channels=8,
        num_epochs=1,
        batch_size=32,
        warmup_steps=2,
        holdout_fraction=0.34,
        early_stopping_patience=10,
        corpus_dir=str(corpus_dir),
        output_checkpoint=str(ckpt_path),
        log_dir=str(tmp_path / "runs"),
        device="cpu",
    )
    pretrain_supervised(pre_cfg)

    # Build a Trainer with matching shape and load the pretrained ckpt
    train_cfg = TrainingConfig(
        n_blocks=1,
        n_channels=8,
        num_iterations=1,
        games_per_iteration=1,
        training_steps_per_iteration=1,
        batch_size=8,
        min_buffer_size=1,
        replay_buffer_capacity=100,
        num_simulations=2,
        device="cpu",
        checkpoint_dir=str(tmp_path / "ckpts"),
    )
    trainer = Trainer(Chess(), train_cfg)
    trainer.load_from_checkpoint(ckpt_path)

    # Verify best_net weights match what pretrain wrote
    import torch
    pretrained_sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["best_net"]
    for k in pretrained_sd:
        assert torch.equal(pretrained_sd[k], trainer.best_net.state_dict()[k]), (
            f"Weight mismatch at {k} — pretrained checkpoint did not load correctly"
        )
```

- [ ] **Step 2: Run the test**

```bash
cd /Users/vsowrira/git/alphazero && python3 -m pytest tests/test_supervised.py::test_pretrained_checkpoint_loads_in_trainer_via_resume -v
```

Expected: passes. If it fails with a key mismatch in the checkpoint, fix `_save_pretrained_checkpoint` in Task 11 to populate the missing keys, then re-run.

- [ ] **Step 3: Commit**

```bash
cd /Users/vsowrira/git/alphazero
git add tests/test_supervised.py
git commit -m "$(cat <<'EOF'
Add integration test: pretrained.pt loads via Trainer.load_from_checkpoint

Regression guard for the checkpoint format contract between
supervised pre-training and AZ refinement. If a future change to
_save_pretrained_checkpoint breaks compatibility with
Trainer.load_from_checkpoint, this test will fail immediately rather
than discovering the problem 5 hours into a refinement run.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 16: E2E slow test (corpus + pretrain + 1 refinement iter)

**Files:**
- Create: `tests/test_e2e_supervised.py`

- [ ] **Step 1: Write the E2E test**

Create `/Users/vsowrira/git/alphazero/tests/test_e2e_supervised.py`:

```python
"""End-to-end test for the SP3b pipeline: corpus gen → pretrain → 1 AZ refinement iter.

This is a SLOW test — actually spawns Stockfish, generates a tiny corpus, pretrains,
runs one AZ iteration. Marked slow so it doesn't run in default pytest.

Run with:
    pytest -m slow tests/test_e2e_supervised.py -v -s
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest


@pytest.mark.slow
def test_e2e_corpus_pretrain_refine_pipeline(tmp_path, monkeypatch):
    """Mini run of the full SP3b pipeline. Wall-clock: ~3 minutes."""
    from alphazero.config import TrainingConfig
    from alphazero.corpus import generate_corpus
    from alphazero.games.chess_game import Chess
    from alphazero.supervised import PretrainConfig, pretrain_supervised
    from alphazero.trainer import Trainer

    monkeypatch.chdir(tmp_path)

    corpus_dir = tmp_path / "corpus"

    # Phase 1: tiny corpus (4 games, 2 workers)
    generate_corpus(
        target_games=4,
        num_workers=2,
        output_dir=corpus_dir,
        time_per_move=0.01,
        seed=42,
    )
    shards = sorted(corpus_dir.glob("shard_*.npz"))
    assert len(shards) >= 2

    # Phase 2: pretrain a tiny network
    pretrained_path = tmp_path / "pretrained.pt"
    pre_cfg = PretrainConfig(
        n_blocks=1,
        n_channels=8,
        num_epochs=1,
        batch_size=16,
        warmup_steps=2,
        holdout_fraction=0.34,
        early_stopping_patience=10,
        corpus_dir=str(corpus_dir),
        output_checkpoint=str(pretrained_path),
        log_dir=str(tmp_path / "runs_pre"),
        device="cpu",
    )
    pretrain_supervised(pre_cfg)
    assert pretrained_path.exists()

    # Phase 3: 1 iteration of AZ refinement (single-worker, serial path)
    train_cfg = TrainingConfig(
        n_blocks=1,
        n_channels=8,
        num_iterations=1,
        games_per_iteration=2,
        num_simulations=3,
        training_steps_per_iteration=2,
        batch_size=4,
        min_buffer_size=4,
        replay_buffer_capacity=100,
        eval_interval=999,        # skip eval
        device="cpu",
        checkpoint_dir=str(tmp_path / "ckpts"),
        log_dir=str(tmp_path / "runs_refine"),
    )
    trainer = Trainer(Chess(), train_cfg)
    trainer.load_from_checkpoint(pretrained_path)
    trainer.run(verbose=False)

    # Verify a refinement checkpoint was written
    assert (tmp_path / "ckpts" / "iter_0001.pt").exists()
```

- [ ] **Step 2: Verify it's marked slow + collects correctly**

```bash
cd /Users/vsowrira/git/alphazero && python3 -m pytest --collect-only -m "not slow" 2>&1 | grep test_e2e_supervised || echo "(excluded ✓)"
cd /Users/vsowrira/git/alphazero && python3 -m pytest --collect-only -m slow tests/test_e2e_supervised.py 2>&1 | tail -5
```

Expected: first prints `(excluded ✓)`; second collects 1 test.

- [ ] **Step 3: Actually run it (sanity check the pipeline works)**

```bash
cd /Users/vsowrira/git/alphazero && python3 -m pytest -m slow tests/test_e2e_supervised.py -v -s
```

Expected: passes in ~1-3 minutes. If it fails, the failure tells you which phase broke.

- [ ] **Step 4: Commit**

```bash
cd /Users/vsowrira/git/alphazero
git add tests/test_e2e_supervised.py
git commit -m "$(cat <<'EOF'
Add E2E slow test: corpus → pretrain → 1 AZ refinement iter

Mini-run of the full SP3b pipeline: 4 games corpus, 1-epoch pretrain
on a tiny network, then 1 iter of AZ refinement. Wall-clock ~3 min.
Catches integration bugs that unit tests miss (e.g., checkpoint format
mismatch between pretrain and Trainer.load_from_checkpoint, or
Trainer crashing on a pretrained-format initial state).

Run with: pytest -m slow tests/test_e2e_supervised.py -v -s

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 17: Final sweep + push

- [ ] **Step 1: Run all non-slow tests**

```bash
cd /Users/vsowrira/git/alphazero && python3 -m pytest -m "not slow" 2>&1 | tail -10
```

Expected: all tests pass. ~165 baseline (post-SP3) + ~20 new SP3b tests = ~185 total.

- [ ] **Step 2: Run linter (informational; ignore minor style)**

```bash
cd /Users/vsowrira/git/alphazero && ruff check src tests scripts 2>&1 | tail -15
```

If there are real correctness errors (not style F401/F841), fix them.

- [ ] **Step 3: Push everything**

```bash
cd /Users/vsowrira/git/alphazero && git push 2>&1 | tail -3
```

Expected: push succeeds.

- [ ] **Step 4 (Optional): Run a small real-corpus smoke test**

This validates the full pipeline against actual Stockfish before committing to the 4-hour corpus generation. Wall-clock: ~2 minutes.

```bash
cd /Users/vsowrira/git/alphazero
rm -rf /tmp/sp3b_smoke
python3 scripts/generate_chess_corpus.py --num-games 20 --num-workers 4 --output /tmp/sp3b_smoke --time-per-move 0.05
ls /tmp/sp3b_smoke/
```

Expected: 4 worker shards, ~200-400 positions total.

- [ ] **Step 5 (Manual): Kick off the real pipeline**

Three commands, ~15 hours total wall-clock:

```bash
# Phase 1: ~4.2 hours
python3 scripts/generate_chess_corpus.py --num-games 50000 --num-workers 10 --output data/chess_corpus/

# Phase 2: ~30 minutes
python3 -m alphazero pretrain --config configs/chess-pretrain.toml

# Phase 3: ~10 hours
python3 -m alphazero train --game chess --config configs/chess-refine.toml --resume-from pretrained.pt
```

Then evaluate:

```bash
python3 -m alphazero eval --game chess --checkpoint checkpoints_refine/iter_0010.pt --num-games 100
```

Pass criterion: wins / (wins + losses + draws) ≥ 0.50 vs Stockfish ELO=1500.

---

## Final state

After Task 16, the repo has:

- `src/alphazero/corpus.py` — Stockfish-vs-Stockfish corpus generation primitives + orchestrator
- `src/alphazero/supervised.py` — PretrainConfig, lr_schedule, corpus loader, pretrain step/epoch, main loop
- `scripts/generate_chess_corpus.py` — CLI for corpus generation
- `src/alphazero/cli.py` — extended with `pretrain` subcommand
- `configs/chess-pretrain.toml` and `configs/chess-refine.toml`
- 3 new test files (~350 lines) + extension to `tests/test_cli.py`

The framework's reuse property is preserved — `AlphaZeroNet`, `Chess(Game)`, `chess_move_encoding`, `StockfishOpponent`, `parallel_selfplay`, `Trainer.run()` with `--resume-from` are all unchanged.

**Sub-project 3b is done when** `python -m alphazero eval --game chess --checkpoint checkpoints_refine/iter_0010.pt --num-games 100` reports `win_rate ≥ 0.50` vs Stockfish ELO=1500.
