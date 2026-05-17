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
