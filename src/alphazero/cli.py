"""Command-line entry point: python -m alphazero <subcommand> [...]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from .arena import play_match
from .config import TrainingConfig, load_config
from .games.tictactoe import TicTacToe
from .solvers.tictactoe_solver import solve_tictactoe_action
from .trainer import Trainer


def _cmd_train(args: argparse.Namespace) -> int:
    config = load_config(args.config) if args.config else TrainingConfig()
    game = TicTacToe()
    trainer = Trainer(game, config)
    print(f"Starting training: {config.num_iterations} iterations on {trainer.device}")
    trainer.run()
    print("Training complete.")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    config = TrainingConfig(device="cpu")
    game = TicTacToe()
    trainer = Trainer(game, config)

    ckpt = torch.load(args.checkpoint, map_location=trainer.device)
    trainer.best_net.load_state_dict(ckpt["best_net"])
    trainer.best_net.eval()

    agent = trainer._make_argmax_mcts_agent(trainer.best_net)
    def solver_agent(_g, state):
        return solve_tictactoe_action(state)

    result = play_match(game, agent, solver_agent, num_games=args.num_games)
    print(
        f"vs perfect solver ({args.num_games} games): "
        f"wins={result.wins_a} draws={result.draws} losses={result.losses_a}"
    )
    return 0 if result.losses_a == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m alphazero")
    subs = parser.add_subparsers(dest="cmd", required=True)

    p_train = subs.add_parser("train", help="Run training loop")
    p_train.add_argument("--config", type=Path, default=Path("configs/tictactoe.toml"))
    p_train.set_defaults(func=_cmd_train)

    p_eval = subs.add_parser("eval", help="Evaluate a checkpoint vs perfect solver")
    p_eval.add_argument("--checkpoint", type=Path, required=True)
    p_eval.add_argument("--num-games", type=int, default=200)
    p_eval.set_defaults(func=_cmd_eval)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
