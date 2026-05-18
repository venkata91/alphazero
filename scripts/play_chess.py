#!/usr/bin/env python3
"""Play a game of chess against an AlphaZero chess checkpoint.

Accepts a checkpoint produced by `python -m alphazero pretrain` (the
pretrained.pt) or the per-iteration files written by AZ refinement
(checkpoints_refine/iter_NNNN.pt). Uses MCTS for the agent's moves.

Usage:
    python3 scripts/play_chess.py --checkpoint pretrained.pt --as white

    # Make the agent stronger by giving it more search:
    python3 scripts/play_chess.py --checkpoint pretrained.pt --as white --num-simulations 800

    # Compare against the refined checkpoint:
    python3 scripts/play_chess.py --checkpoint checkpoints_refine/iter_0010.pt --as black

Moves can be entered in either UCI (e.g., 'e2e4', 'g1f3') or SAN ('e4', 'Nf3').
Type 'quit' or Ctrl-C to abort.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import chess
import numpy as np
import torch

from alphazero.device import resolve_device
from alphazero.games.chess_game import Chess
from alphazero.games.chess_move_encoding import index_to_move
from alphazero.mcts import MCTS
from alphazero.network import AlphaZeroNet


def make_eval_fn(net: AlphaZeroNet, device: torch.device):
    """Build the MCTS eval_fn closure: (encoded_state) -> (priors, value)."""

    def eval_fn(encoded: np.ndarray) -> tuple[np.ndarray, float]:
        x = torch.from_numpy(encoded).float().unsqueeze(0).to(device)
        with torch.no_grad():
            logits, value = net(x)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return probs, float(value.item())

    return eval_fn


def make_agent(net: AlphaZeroNet, device: torch.device, num_simulations: int, c_puct: float):
    """Build an MCTS argmax agent (deterministic play, no Dirichlet noise)."""
    eval_fn = make_eval_fn(net, device)
    game = Chess()

    def agent(_g, state):
        mcts = MCTS(game, eval_fn, c_puct=c_puct)
        pi = mcts.search(state, num_simulations=num_simulations, add_root_noise=False)
        return int(np.argmax(pi))

    return agent


def parse_user_move(user_input: str, board: chess.Board) -> chess.Move | None:
    """Try UCI then SAN. Return chess.Move if legal, None otherwise."""
    s = user_input.strip()
    if not s:
        return None
    try:
        move = chess.Move.from_uci(s)
        if move in board.legal_moves:
            return move
    except (ValueError, chess.InvalidMoveError):
        pass
    try:
        move = board.parse_san(s)
        if move in board.legal_moves:
            return move
    except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError):
        pass
    return None


def print_board(board: chess.Board, human_color: bool) -> None:
    """Render the board from the human's POV with file/rank labels."""
    flip = (human_color == chess.BLACK)
    print()
    for rank in (range(8) if flip else range(7, -1, -1)):
        line = f"{rank + 1} "
        for file in (range(7, -1, -1) if flip else range(8)):
            piece = board.piece_at(chess.square(file, rank))
            line += f" {piece.symbol() if piece else '.'}"
        print(line)
    file_labels = "   " + " ".join(reversed("abcdefgh")) if flip else "   " + " ".join("abcdefgh")
    print(file_labels)
    print()


def print_result(board: chess.Board, human_color: bool) -> None:
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        print("Game ended without an outcome.")
        return
    if outcome.winner is None:
        reason_map = {
            chess.Termination.STALEMATE: "stalemate",
            chess.Termination.INSUFFICIENT_MATERIAL: "insufficient material",
            chess.Termination.SEVENTYFIVE_MOVES: "75-move rule",
            chess.Termination.FIVEFOLD_REPETITION: "5-fold repetition",
            chess.Termination.FIFTY_MOVES: "50-move rule",
            chess.Termination.THREEFOLD_REPETITION: "3-fold repetition",
        }
        reason = reason_map.get(outcome.termination, "unknown")
        print(f"Draw — {reason}.")
    else:
        human_won = (outcome.winner == human_color)
        reason = "checkmate" if outcome.termination == chess.Termination.CHECKMATE else "termination"
        print(f"{'You win' if human_won else 'You lose'} by {reason}.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to a pretrained.pt or iter_NNNN.pt checkpoint")
    parser.add_argument("--as", dest="as_color", choices=["white", "black"], default="white",
                        help="Play as white (move first) or black. Default: white.")
    parser.add_argument("--num-simulations", type=int, default=200,
                        help="MCTS simulations per agent move. Default 200. "
                             "400-800 makes the agent meaningfully stronger but slower.")
    parser.add_argument("--c-puct", type=float, default=2.5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Loading {args.checkpoint} on {device}...", flush=True)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    net = AlphaZeroNet(
        input_shape=(20, 8, 8),
        action_size=4672,
        n_blocks=cfg["n_blocks"],
        n_channels=cfg["n_channels"],
    ).to(device)
    net.load_state_dict(ckpt["best_net"])
    net.eval()

    epoch_or_iter = ckpt.get("_pretrain_epoch") or ckpt.get("iteration", "?")
    n_params = sum(v.numel() for v in ckpt["best_net"].values())
    print(f"Loaded {cfg['n_blocks']} blocks × {cfg['n_channels']} channels "
          f"({n_params:,} params); checkpoint epoch/iter {epoch_or_iter}")
    print(f"MCTS: {args.num_simulations} sims per move, c_puct={args.c_puct}")

    human_color = chess.WHITE if args.as_color == "white" else chess.BLACK
    print(f"\nYou are {args.as_color}. "
          f"{'You move first.' if human_color == chess.WHITE else 'Agent moves first.'}")
    print("Enter moves as UCI ('e2e4') or SAN ('e4'). Type 'quit' to abort.")

    game = Chess()
    agent = make_agent(net, device, args.num_simulations, args.c_puct)

    board = chess.Board()
    while not board.is_game_over(claim_draw=True):
        print_board(board, human_color)

        if board.turn == human_color:
            while True:
                try:
                    user_in = input(f"Your move ({'white' if human_color == chess.WHITE else 'black'}): ")
                except (EOFError, KeyboardInterrupt):
                    print("\nGame aborted.")
                    return 130
                if user_in.strip().lower() in ("quit", "exit", "q"):
                    print("Game aborted.")
                    return 0
                move = parse_user_move(user_in, board)
                if move is not None:
                    break
                print(f"Illegal or unparseable: {user_in!r}. Try UCI ('e2e4') or SAN ('e4').")
            print(f"  You: {board.san(move)}")
            board.push(move)
        else:
            print(f"Agent thinking ({args.num_simulations} sims)...", flush=True)
            action_idx = agent(game, board)
            agent_move = index_to_move(board, action_idx)
            print(f"  Agent: {board.san(agent_move)}")
            board.push(agent_move)

    print_board(board, human_color)
    print_result(board, human_color)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
