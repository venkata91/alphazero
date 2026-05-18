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
from .games.connect4 import Connect4
from .games.chess_game import Chess
from .solvers.tictactoe_solver import solve_tictactoe_action
from .trainer import Trainer

GAMES = {
    "tictactoe": TicTacToe,
    "connect4": Connect4,
    "chess": Chess,
}


def _cmd_train(args: argparse.Namespace) -> int:
    config = load_config(args.config) if args.config else TrainingConfig()
    game_cls = GAMES[args.game]
    game = game_cls()
    trainer = Trainer(game, config)

    if args.resume_from is not None:
        print(f"Resuming from {args.resume_from}", flush=True)
        trainer.load_from_checkpoint(args.resume_from)
        print(f"Loaded iteration {trainer.iteration}; will run {config.num_iterations} more.",
              flush=True)

    # Build the per-game eval opponent. Without this, Trainer falls through
    # to _eval_vs_solver (TTT minimax) — which crashes on Connect 4 states.
    eval_opponent = None
    if args.game == "connect4":
        from .opponents.connect4_minimax import Connect4MinimaxOpponent
        eval_opponent = Connect4MinimaxOpponent(depth=8)
    elif args.game == "chess":
        from .opponents.stockfish import StockfishOpponent
        eval_opponent = StockfishOpponent(elo=1500, time_per_move=0.5)
    # For tictactoe, leaving eval_opponent=None routes through _eval_vs_solver
    # (the perfect TTT minimax) which is correct.

    print(f"Starting training: {config.num_iterations} iterations on {trainer.device}")
    try:
        trainer.run(eval_opponent=eval_opponent)
        print("Training complete.")
        return 0
    finally:
        # Ensure the Stockfish subprocess is reaped even if training raises.
        # Connect4MinimaxOpponent has no .close() — hasattr keeps this safe
        # for both opponent types.
        if eval_opponent is not None and hasattr(eval_opponent, "close"):
            eval_opponent.close()


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
    game_cls = GAMES[args.game]
    game = game_cls()
    trainer = Trainer(game, config)

    ckpt = torch.load(args.checkpoint, map_location=trainer.device, weights_only=False)
    trainer.best_net.load_state_dict(ckpt["best_net"])
    trainer.best_net.eval()

    agent = trainer._make_argmax_mcts_agent(trainer.best_net)

    if args.game == "tictactoe":
        def opponent(_g, state):
            return solve_tictactoe_action(state)
        label = "perfect solver"
    elif args.game == "connect4":
        from .opponents.connect4_minimax import Connect4MinimaxOpponent
        opponent = Connect4MinimaxOpponent(depth=8)
        label = "minimax-depth-8"
    elif args.game == "chess":
        from .opponents.stockfish import StockfishOpponent
        opponent = StockfishOpponent(elo=1500, time_per_move=0.5)
        label = "Stockfish ELO=1500"
    else:
        raise ValueError(f"No eval opponent defined for game={args.game}")

    try:
        result = play_match(game, agent, opponent, num_games=args.num_games)
        print(
            f"vs {label} ({args.num_games} games): "
            f"wins={result.wins_a} draws={result.draws} losses={result.losses_a}"
        )
        return 0 if result.losses_a == 0 else 1
    finally:
        # Same Stockfish-leak protection as _cmd_train.
        if hasattr(opponent, "close"):
            opponent.close()


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


def _cmd_play_tictactoe(args: argparse.Namespace) -> int:
    """Interactive TTT play against best_net loaded from a checkpoint."""
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

    user_plays_x = args.as_player.lower() in ("x", "white", "w")
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


def _render_chess_board(board, human_color) -> str:
    """ASCII chess board from the human's POV (flipped if playing Black)."""
    import chess
    flip = (human_color == chess.BLACK)
    lines = [""]
    for rank in (range(8) if flip else range(7, -1, -1)):
        line = f"{rank + 1} "
        for file in (range(7, -1, -1) if flip else range(8)):
            piece = board.piece_at(chess.square(file, rank))
            line += f" {piece.symbol() if piece else '.'}"
        lines.append(line)
    file_labels = "   " + " ".join(reversed("abcdefgh")) if flip else "   " + " ".join("abcdefgh")
    lines.append(file_labels)
    return "\n".join(lines)


def _parse_chess_move(user_input: str, board):
    """Parse user input as UCI first, then SAN. Returns chess.Move or None."""
    import chess
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


def _print_chess_result(board, human_color) -> None:
    """Print game outcome from the human's perspective."""
    import chess
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


def _normalize_chess_color(as_player: str):
    """Map any of {white,black,w,b,x,o} to chess.WHITE / chess.BLACK.

    Accepting x/o keeps the --as flag uniform across games (x → first-mover →
    white; o → second-mover → black), so the CLI doesn't grow a separate flag
    per game.
    """
    import chess
    p = as_player.lower()
    if p in ("white", "w", "x"):
        return chess.WHITE
    if p in ("black", "b", "o"):
        return chess.BLACK
    raise ValueError(f"Unrecognized --as value: {as_player!r}")


def _cmd_play_chess(args: argparse.Namespace) -> int:
    """Interactive chess play against best_net loaded from a checkpoint.

    Mirrors `scripts/play_chess.py` but goes through Trainer's checkpoint
    load + agent construction so it stays in lockstep with `_cmd_eval`.
    """
    import chess
    from .games.chess_move_encoding import index_to_move

    config = _config_for_checkpoint(args.checkpoint, args.config)
    if args.num_simulations is not None:
        from dataclasses import replace
        config = replace(config, num_simulations=args.num_simulations)
    game = Chess()
    trainer = Trainer(game, config)

    ckpt = torch.load(args.checkpoint, map_location=trainer.device, weights_only=False)
    trainer.best_net.load_state_dict(ckpt["best_net"])
    trainer.best_net.eval()
    agent = trainer._make_argmax_mcts_agent(trainer.best_net)

    human_color = _normalize_chess_color(args.as_player)
    human_label = "white" if human_color == chess.WHITE else "black"
    human_player_id = 1 if human_color == chess.WHITE else -1

    epoch_or_iter = ckpt.get("_pretrain_epoch") or ckpt.get("iteration", "?")
    print(f"\nLoaded best_net from epoch/iter {epoch_or_iter}.")
    print(f"MCTS: {config.num_simulations} sims per move, c_puct={config.c_puct}")
    print(f"You are {human_label}. "
          f"{'You move first.' if human_color == chess.WHITE else 'Agent moves first.'}")
    print("Enter moves as UCI ('e2e4') or SAN ('e4'). Type 'quit' to abort.")

    state = game.initial_state()
    while game.terminal_value(state) is None:
        print(_render_chess_board(state, human_color))

        if game.current_player(state) == human_player_id:
            while True:
                try:
                    raw = input(f"Your move ({human_label}): ")
                except (EOFError, KeyboardInterrupt):
                    print("\nGame aborted.")
                    return 130
                if raw.strip().lower() in ("quit", "exit", "q"):
                    print("Game aborted.")
                    return 0
                move = _parse_chess_move(raw, state)
                if move is not None:
                    break
                print(f"  Illegal or unparseable: {raw!r}. Try UCI ('e2e4') or SAN ('e4').")
            print(f"  You: {state.san(move)}")
            new_state = state.copy()
            new_state.push(move)
            state = new_state
        else:
            print(f"Agent thinking ({config.num_simulations} sims)...", flush=True)
            action_idx = agent(game, state)
            agent_move = index_to_move(state, action_idx)
            print(f"  Agent: {state.san(agent_move)}")
            state = game.apply(state, action_idx)

    print(_render_chess_board(state, human_color))
    _print_chess_result(state, human_color)
    return 0


def _cmd_play(args: argparse.Namespace) -> int:
    """Interactive play dispatcher: routes by --game.

    TTT and chess have different state types (np.ndarray vs chess.Board),
    rendering, and input formats (action index vs UCI/SAN), so each game
    gets its own implementation. Connect 4 interactive play isn't wired up
    yet (no renderer or input scheme), so it's still rejected at the
    argparse layer.
    """
    if args.game == "tictactoe":
        return _cmd_play_tictactoe(args)
    if args.game == "chess":
        return _cmd_play_chess(args)
    raise ValueError(f"Unsupported --game for play: {args.game}")


def _cmd_pretrain(args: argparse.Namespace) -> int:
    """Run supervised pre-training using a PretrainConfig from TOML."""
    from .supervised import load_pretrain_config, pretrain_supervised

    config = load_pretrain_config(args.config)
    pretrain_supervised(config)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m alphazero")
    subs = parser.add_subparsers(dest="cmd", required=True)

    p_train = subs.add_parser("train", help="Run training loop")
    p_train.add_argument("--game", choices=list(GAMES), default="tictactoe")
    p_train.add_argument("--config", type=Path, default=Path("configs/tictactoe.toml"))
    p_train.add_argument("--resume-from", type=Path, default=None,
                         help="Path to a checkpoint to resume training from. "
                              "Loads best_net, candidate_net, optimizer, and iteration "
                              "counter; continues for `config.num_iterations` more iters.")
    p_train.set_defaults(func=_cmd_train)

    p_eval = subs.add_parser("eval", help="Evaluate a checkpoint vs perfect solver")
    p_eval.add_argument("--game", choices=list(GAMES), default="tictactoe")
    p_eval.add_argument("--checkpoint", type=Path, required=True)
    p_eval.add_argument("--config", type=Path, default=None,
                        help="Override architecture config (needed for older checkpoints "
                             "that didn't save their config inline).")
    p_eval.add_argument("--num-games", type=int, default=200)
    p_eval.set_defaults(func=_cmd_eval)

    p_play = subs.add_parser(
        "play",
        help="Play interactively against best_net (tictactoe or chess)",
    )
    # tictactoe + chess are wired up; connect4 isn't (no 6×7 renderer or
    # column-input scheme yet), so it stays rejected at argparse to avoid
    # a confusing mid-game crash.
    p_play.add_argument("--game", choices=["tictactoe", "chess"], default="tictactoe",
                        help="Game to play. tictactoe uses action-index input; "
                             "chess accepts UCI ('e2e4') or SAN ('e4').")
    p_play.add_argument("--checkpoint", type=Path, required=True)
    p_play.add_argument("--config", type=Path, default=None,
                        help="Override architecture config (needed for older checkpoints "
                             "that didn't save their config inline).")
    p_play.add_argument("--num-simulations", type=int, default=None,
                        help="Override MCTS simulations per move at play time. "
                             "Default uses the config's value. Crank to 200-1000 for "
                             "much stronger play even with a weak network.")
    # Accept both TTT vocabulary (x/o) and chess vocabulary (white/black/w/b)
    # for --as so a single flag works for both games. Normalization happens
    # inside _cmd_play_{tictactoe,chess}.
    p_play.add_argument("--as", dest="as_player",
                        choices=["x", "o", "X", "O", "white", "black", "w", "b"],
                        default="x",
                        help="Play as first-mover (x/white/w) or second-mover (o/black/b). "
                             "Default: x.")
    p_play.set_defaults(func=_cmd_play)

    p_pretrain = subs.add_parser("pretrain", help="Supervised pre-training on a corpus")
    p_pretrain.add_argument("--config", type=Path, required=True,
                            help="Path to a chess-pretrain.toml file")
    p_pretrain.set_defaults(func=_cmd_pretrain)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
