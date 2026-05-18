#!/usr/bin/env python3
"""UCI engine wrapper around an AlphaZero chess checkpoint.

Speaks the Universal Chess Interface protocol so any UCI-compatible host
can drive the engine: Lichess (via lichess-bot), Arena GUI, Cute Chess,
chess.com (via their bridge), etc.

Usage as a subprocess:
    python3 scripts/uci_engine.py --checkpoint pretrained.pt --num-simulations 200

The host writes UCI commands to stdin (one per line) and reads responses
from stdout. Supported commands:
    uci                  → respond with id + uciok
    isready              → respond readyok
    ucinewgame           → reset internal board
    position startpos|fen ... [moves m1 m2 ...]
                         → set the board state
    go [movetime N] ...  → search and respond with bestmove <UCI>
    quit                 → exit
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import chess
import numpy as np
import torch

from alphazero.device import resolve_device
from alphazero.games.chess_game import Chess
from alphazero.games.chess_move_encoding import index_to_move
from alphazero.mcts import MCTS
from alphazero.network import AlphaZeroNet

ENGINE_NAME = "AlphaZero-SP3b"
ENGINE_AUTHOR = "venkata91/alphazero"


class UCIEngine:
    def __init__(self, checkpoint_path: Path, num_simulations: int = 200, c_puct: float = 2.5):
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = resolve_device("auto")
        self._load_checkpoint(checkpoint_path)
        self.game = Chess()
        self.board = chess.Board()

    def _load_checkpoint(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        cfg = ckpt["config"]
        self.net = AlphaZeroNet(
            input_shape=(20, 8, 8),
            action_size=4672,
            n_blocks=cfg["n_blocks"],
            n_channels=cfg["n_channels"],
        ).to(self.device)
        self.net.load_state_dict(ckpt["best_net"])
        self.net.eval()
        # info string to stderr, so logs don't interfere with UCI stdout
        sys.stderr.write(
            f"loaded {cfg['n_blocks']}x{cfg['n_channels']} from {path} "
            f"on device={self.device}\n"
        )

    def _eval_fn(self, encoded: np.ndarray) -> tuple[np.ndarray, float]:
        x = torch.from_numpy(encoded).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.net(x)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return probs, float(value.item())

    def _search(self, max_seconds: float | None = None) -> chess.Move:
        """Run MCTS and return the best move.

        If max_seconds is given, scale num_simulations down so we finish
        roughly within the budget (very rough estimate: ~30ms per sim
        on Mac MPS for our 10.6M net at batch=1).
        """
        sims = self.num_simulations
        if max_seconds is not None and max_seconds > 0:
            # Conservative per-sim estimate; better to under-budget than time-out
            est_per_sim_s = 0.03
            budget_sims = max(8, int(max_seconds / est_per_sim_s))
            sims = min(sims, budget_sims)

        mcts = MCTS(self.game, self._eval_fn, c_puct=self.c_puct)
        pi = mcts.search(self.board, num_simulations=sims, add_root_noise=False)
        idx = int(np.argmax(pi))
        return index_to_move(self.board, idx)

    def _parse_position(self, tokens: list[str]) -> None:
        """Handle: position [startpos | fen <FEN>] [moves m1 m2 ...]"""
        if not tokens:
            return
        if tokens[0] == "startpos":
            self.board = chess.Board()
            remainder = tokens[1:]
        elif tokens[0] == "fen":
            # FEN is exactly 6 fields; the 'moves' keyword (if any) comes after
            fen_parts = tokens[1:7]
            if len(fen_parts) < 6:
                # malformed; leave board untouched
                return
            self.board = chess.Board(" ".join(fen_parts))
            remainder = tokens[7:]
        else:
            return

        if remainder and remainder[0] == "moves":
            for mv in remainder[1:]:
                try:
                    self.board.push_uci(mv)
                except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError):
                    # Stop on first malformed move
                    sys.stderr.write(f"warning: bad move {mv!r}; skipping rest\n")
                    break

    def _parse_go_budget(self, tokens: list[str]) -> float | None:
        """Extract a time budget (in seconds) from `go` arguments.

        Recognizes the common UCI time-control flags. Returns None if no
        budget is specified (uses full num_simulations).
        """
        i = 0
        wtime = btime = winc = binc = movetime = None
        movestogo = 30  # default if not provided
        while i < len(tokens):
            t = tokens[i]
            if t in ("wtime", "btime", "winc", "binc", "movetime") and i + 1 < len(tokens):
                try:
                    val = int(tokens[i + 1])
                except ValueError:
                    val = None
                if t == "wtime": wtime = val
                elif t == "btime": btime = val
                elif t == "winc": winc = val
                elif t == "binc": binc = val
                elif t == "movetime": movetime = val
                i += 2
            elif t == "movestogo" and i + 1 < len(tokens):
                try:
                    movestogo = int(tokens[i + 1])
                except ValueError:
                    pass
                i += 2
            else:
                i += 1

        if movetime is not None:
            return movetime / 1000.0

        # Allocate roughly remaining_time / movestogo + increment
        side_time = wtime if self.board.turn == chess.WHITE else btime
        side_inc = winc if self.board.turn == chess.WHITE else binc
        if side_time is None:
            return None
        budget_ms = side_time / max(1, movestogo) + (side_inc or 0)
        # Be conservative: spend at most half the per-move allocation
        return max(0.1, budget_ms / 1000.0 * 0.5)

    def handle_line(self, line: str) -> bool:
        """Process one UCI command line. Returns False if we should exit."""
        tokens = line.strip().split()
        if not tokens:
            return True
        cmd = tokens[0]

        if cmd == "uci":
            print(f"id name {ENGINE_NAME}")
            print(f"id author {ENGINE_AUTHOR}")
            # advertise tunable: number of MCTS sims per move
            print(f"option name MCTSSimulations type spin default {self.num_simulations} min 8 max 4000")
            print("uciok")
        elif cmd == "isready":
            print("readyok")
        elif cmd == "ucinewgame":
            self.board = chess.Board()
        elif cmd == "position":
            self._parse_position(tokens[1:])
        elif cmd == "go":
            budget = self._parse_go_budget(tokens[1:])
            try:
                move = self._search(max_seconds=budget)
                print(f"bestmove {move.uci()}")
            except Exception as e:
                sys.stderr.write(f"search failed: {e!r}\n")
                # Fall back to first legal move so we don't forfeit
                legal = list(self.board.legal_moves)
                if legal:
                    print(f"bestmove {legal[0].uci()}")
                else:
                    print("bestmove 0000")
        elif cmd == "setoption":
            # setoption name MCTSSimulations value 400
            if "value" in tokens:
                try:
                    name_idx = tokens.index("name") + 1
                    value_idx = tokens.index("value") + 1
                    name = tokens[name_idx]
                    value = tokens[value_idx]
                    if name == "MCTSSimulations":
                        self.num_simulations = int(value)
                        sys.stderr.write(f"set MCTSSimulations={self.num_simulations}\n")
                except (ValueError, IndexError):
                    pass
        elif cmd == "quit":
            return False
        # Unknown commands are silently ignored per UCI convention

        sys.stdout.flush()
        return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("pretrained.pt"))
    parser.add_argument("--num-simulations", type=int, default=200)
    parser.add_argument("--c-puct", type=float, default=2.5)
    args = parser.parse_args()

    engine = UCIEngine(args.checkpoint, args.num_simulations, args.c_puct)

    while True:
        try:
            line = sys.stdin.readline()
        except KeyboardInterrupt:
            break
        if not line:
            break
        if engine.handle_line(line) is False:
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
