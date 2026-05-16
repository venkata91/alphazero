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


def _config_for_checkpoint(checkpoint_path: Path, override_path: Path | None) -> TrainingConfig:
    """Reconstruct the right config to load a checkpoint.

    Order of precedence: (1) --config override, (2) config saved in checkpoint,
    (3) defaults. The architecture knobs (n_blocks, n_channels) must match what
    the checkpoint was saved with or load_state_dict will fail.

    Filters out keys in the saved config dict that aren't recognized by the
    current TrainingConfig schema — keeps older checkpoints loadable after we
    remove deprecated fields (e.g., the old arena_* knobs).
    """
    from dataclasses import fields
    if override_path is not None:
        return load_config(override_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved = ckpt.get("config")
    if saved is None:
        return TrainingConfig(device="cpu")
    valid_keys = {f.name for f in fields(TrainingConfig)}
    saved = {k: v for k, v in saved.items() if k in valid_keys}
    saved["device"] = "cpu"  # Force CPU for inference
    return TrainingConfig(**saved)


def _cmd_eval(args: argparse.Namespace) -> int:
    config = _config_for_checkpoint(args.checkpoint, args.config)
    game = TicTacToe()
    trainer = Trainer(game, config)

    ckpt = torch.load(args.checkpoint, map_location=trainer.device, weights_only=False)
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


def _render_board(state: np.ndarray) -> str:
    """ASCII render of a 3x3 TTT board (X = +1, O = -1, . = empty)."""
    symbols = {1: " X ", -1: " O ", 0: "   "}
    rows = []
    for r in range(3):
        rows.append("|".join(symbols[int(state[r, c])] for c in range(3)))
    return ("\n" + "-" * 11 + "\n").join(rows)


def _render_action_help() -> str:
    return (
        "Action indices (row-major):\n"
        " 0 | 1 | 2 \n"
        "-----------\n"
        " 3 | 4 | 5 \n"
        "-----------\n"
        " 6 | 7 | 8 "
    )


def _cmd_play(args: argparse.Namespace) -> int:
    """Interactive play against best_net loaded from a checkpoint."""
    config = _config_for_checkpoint(args.checkpoint, args.config)
    if args.num_simulations is not None:
        # Override MCTS depth at inference time. More sims = stronger play
        # even with the same trained NN.
        from dataclasses import replace
        config = replace(config, num_simulations=args.num_simulations)
    game = TicTacToe()
    trainer = Trainer(game, config)

    ckpt = torch.load(args.checkpoint, map_location=trainer.device, weights_only=False)
    trainer.best_net.load_state_dict(ckpt["best_net"])
    trainer.best_net.eval()
    agent = trainer._make_argmax_mcts_agent(trainer.best_net)

    user_plays_x = args.as_player.lower() == "x"
    user_player = 1 if user_plays_x else -1

    print(f"\nLoaded best_net from iteration {ckpt.get('iteration', '?')}.")
    print(f"You are {'X (moves first)' if user_plays_x else 'O (moves second)'}.")
    print(_render_action_help())
    print()

    state = game.initial_state()
    while game.terminal_value(state) is None:
        print(_render_board(state))
        print()

        if game.current_player(state) == user_player:
            legal = game.legal_actions_mask(state)
            legal_indices = np.where(legal)[0]
            while True:
                try:
                    raw = input(f"Your move {sorted(legal_indices.tolist())}: ").strip()
                    action = int(raw)
                    if action in legal_indices:
                        break
                    print(f"  Illegal — must be one of {sorted(legal_indices.tolist())}")
                except ValueError:
                    print("  Enter a number 0-8.")
                except (EOFError, KeyboardInterrupt):
                    print("\nGame aborted.")
                    return 130
            print()
        else:
            action = agent(game, state)
            print(f"Agent plays {action}.")
            print()

        state = game.apply(state, action)

    # Game over
    print(_render_board(state))
    print()
    final_v = game.terminal_value(state)
    cur = game.current_player(state)
    if final_v == 0:
        print("Draw.")
    elif final_v == -1:
        # The current player (the one who would move next) has LOST.
        # So the previous-mover won.
        if cur == user_player:
            print("You lose.")
        else:
            print("You win!")
    else:
        # Shouldn't happen in TTT but handle for generality.
        if cur == user_player:
            print("You win!")
        else:
            print("You lose.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m alphazero")
    subs = parser.add_subparsers(dest="cmd", required=True)

    p_train = subs.add_parser("train", help="Run training loop")
    p_train.add_argument("--config", type=Path, default=Path("configs/tictactoe.toml"))
    p_train.set_defaults(func=_cmd_train)

    p_eval = subs.add_parser("eval", help="Evaluate a checkpoint vs perfect solver")
    p_eval.add_argument("--checkpoint", type=Path, required=True)
    p_eval.add_argument("--config", type=Path, default=None,
                        help="Override architecture config (needed for older checkpoints "
                             "that didn't save their config inline).")
    p_eval.add_argument("--num-games", type=int, default=200)
    p_eval.set_defaults(func=_cmd_eval)

    p_play = subs.add_parser("play", help="Play interactively against best_net")
    p_play.add_argument("--checkpoint", type=Path, required=True)
    p_play.add_argument("--config", type=Path, default=None,
                        help="Override architecture config (needed for older checkpoints "
                             "that didn't save their config inline).")
    p_play.add_argument("--num-simulations", type=int, default=None,
                        help="Override MCTS simulations per move at play time. "
                             "Default uses the config's value. Crank to 200-1000 for "
                             "much stronger play even with a weak network.")
    p_play.add_argument("--as", dest="as_player", choices=["x", "o", "X", "O"], default="x",
                        help="Play as X (moves first) or O (moves second). Default: x.")
    p_play.set_defaults(func=_cmd_play)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
