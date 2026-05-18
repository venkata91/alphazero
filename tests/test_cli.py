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

import pytest

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


def test_cmd_train_closes_stockfish_even_on_run_failure(tmp_path, monkeypatch):
    """Regression for #15: if Trainer.run raises, the StockfishOpponent's
    subprocess must still be reaped. Verifies the try/finally in _cmd_train."""
    monkeypatch.chdir(tmp_path)

    captured: dict = {"close_called": False, "opponent": None}

    class FakeStockfish:
        def __init__(self, *args, **kwargs):
            captured["opponent"] = self

        def close(self):
            captured["close_called"] = True

        def __call__(self, *args, **kwargs):
            return 0

    def fake_run(self, verbose=True, eval_opponent=None):
        raise RuntimeError("simulated training failure")

    with patch("alphazero.opponents.stockfish.StockfishOpponent", FakeStockfish), \
         patch("alphazero.trainer.Trainer.run", fake_run):
        with pytest.raises(RuntimeError, match="simulated training failure"):
            _cmd_train(_stub_args(game="chess"))

    assert captured["opponent"] is not None, "Stockfish opponent should have been constructed"
    assert captured["close_called"], (
        "StockfishOpponent.close() must be called in _cmd_train's finally block "
        "even when Trainer.run raises — otherwise the subprocess leaks."
    )


def test_play_subcommand_rejects_chess_and_connect4(monkeypatch):
    """Regression for #17: `_render_board` is TTT-only (3×3, X/O symbols),
    so `play --game chess` would crash with IndexError mid-game. The
    parser must reject the choice up front with a clear error message
    rather than allowing the run to start and fail confusingly."""
    from alphazero.cli import main

    # argparse raises SystemExit(2) with a usage error when --game is rejected
    for bad_game in ("chess", "connect4"):
        with pytest.raises(SystemExit) as excinfo:
            main(["play", "--game", bad_game, "--checkpoint", "/dev/null"])
        assert excinfo.value.code != 0, (
            f"--game {bad_game} should be rejected by argparse, not silently accepted"
        )


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
