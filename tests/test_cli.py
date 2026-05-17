"""CLI integration tests.

These tests guard against bugs at the CLI dispatch layer — specifically,
making sure the right per-game wiring happens in _cmd_train (and friends)
before the Trainer takes over.

Mock-based for speed: we spy on Trainer.run() to capture what arguments
the CLI passes, without actually running a training loop.
"""
from __future__ import annotations

import argparse
from unittest.mock import patch

from alphazero.cli import _cmd_train
from alphazero.opponents.connect4_minimax import Connect4MinimaxOpponent


def _stub_args(game: str, config=None, resume_from=None) -> argparse.Namespace:
    """Build the args.Namespace _cmd_train expects, with sensible defaults."""
    return argparse.Namespace(game=game, config=config, resume_from=resume_from)


def test_cmd_train_wires_connect4_eval_opponent(tmp_path, monkeypatch):
    """Regression: _cmd_train must pass a Connect4MinimaxOpponent when --game connect4.

    The original bug: _cmd_train called trainer.run() with no eval_opponent,
    causing the Trainer to fall through to _eval_vs_solver (TTT minimax),
    which then crashed at iter 5 when fed a 6×7 Connect 4 board state.
    """
    monkeypatch.chdir(tmp_path)

    captured: dict = {}

    def fake_run(self, verbose=True, eval_opponent=None):
        captured["eval_opponent"] = eval_opponent

    with patch("alphazero.trainer.Trainer.run", fake_run):
        exit_code = _cmd_train(_stub_args(game="connect4"))

    assert exit_code == 0
    assert captured["eval_opponent"] is not None, (
        "_cmd_train must construct an eval_opponent for Connect 4; "
        "without it, Trainer falls through to the TTT solver which "
        "crashes on Connect 4 states."
    )
    assert isinstance(captured["eval_opponent"], Connect4MinimaxOpponent)


def test_cmd_train_leaves_tictactoe_eval_opponent_none(tmp_path, monkeypatch):
    """For tictactoe, _cmd_train should leave eval_opponent=None.

    The Trainer's default _eval_vs_solver path uses the perfect TTT solver,
    which is the correct eval baseline for TTT. We do NOT want to construct
    a Connect4MinimaxOpponent for TTT (game-agnostic Trainer would still
    crash on the type mismatch).
    """
    monkeypatch.chdir(tmp_path)

    captured: dict = {}

    def fake_run(self, verbose=True, eval_opponent=None):
        captured["eval_opponent"] = eval_opponent

    with patch("alphazero.trainer.Trainer.run", fake_run):
        exit_code = _cmd_train(_stub_args(game="tictactoe"))

    assert exit_code == 0
    assert captured["eval_opponent"] is None, (
        "_cmd_train should NOT pass an eval_opponent for tictactoe; "
        "the Trainer's default _eval_vs_solver path is correct for TTT."
    )


def test_cmd_train_wires_chess_eval_opponent(tmp_path, monkeypatch):
    """Regression test: --game chess must construct a StockfishOpponent."""
    from alphazero.opponents.stockfish import StockfishOpponent
    monkeypatch.chdir(tmp_path)

    captured: dict = {}
    def fake_run(self, verbose=True, eval_opponent=None):
        captured["eval_opponent"] = eval_opponent

    with patch("alphazero.trainer.Trainer.run", fake_run):
        exit_code = _cmd_train(_stub_args(game="chess"))

    assert exit_code == 0
    assert isinstance(captured["eval_opponent"], StockfishOpponent)
    captured["eval_opponent"].close()


def test_cmd_pretrain_calls_pretrain_supervised(tmp_path, monkeypatch):
    """Regression test: `pretrain` subcommand wires config + calls pretrain_supervised."""
    from alphazero.cli import _cmd_pretrain

    monkeypatch.chdir(tmp_path)

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
