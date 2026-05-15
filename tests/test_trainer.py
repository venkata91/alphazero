from __future__ import annotations

import numpy as np
import pytest
import torch

from alphazero.games.tictactoe import TicTacToe
from alphazero.trainer import Trainer


def test_trainer_initializes_two_independent_nets(tiny_config):
    trainer = Trainer(TicTacToe(), tiny_config)
    assert trainer.best_net is not trainer.candidate_net
    for p_best, p_cand in zip(trainer.best_net.parameters(), trainer.candidate_net.parameters()):
        torch.testing.assert_close(p_best, p_cand)


def test_trainer_run_iteration_populates_buffer(tiny_config):
    trainer = Trainer(TicTacToe(), tiny_config)
    assert len(trainer.replay_buffer) == 0
    trainer._run_self_play_iteration()
    assert len(trainer.replay_buffer) > 0


def test_trainer_train_step_changes_candidate_weights(tiny_config):
    trainer = Trainer(TicTacToe(), tiny_config)
    trainer._run_self_play_iteration()
    before = [p.detach().clone() for p in trainer.candidate_net.parameters()]
    trainer._train_step()
    after = [p.detach().clone() for p in trainer.candidate_net.parameters()]
    changed = any(not torch.equal(b, a) for b, a in zip(before, after))
    assert changed


def test_trainer_train_step_does_not_change_best_net(tiny_config):
    trainer = Trainer(TicTacToe(), tiny_config)
    trainer._run_self_play_iteration()
    before = [p.detach().clone() for p in trainer.best_net.parameters()]
    trainer._train_step()
    after = [p.detach().clone() for p in trainer.best_net.parameters()]
    for b, a in zip(before, after):
        torch.testing.assert_close(b, a)


def test_trainer_run_advances_iteration_counter(tiny_config):
    trainer = Trainer(TicTacToe(), tiny_config)
    trainer.run()
    assert trainer.iteration == tiny_config.num_iterations


def test_trainer_run_writes_checkpoints(tiny_config, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trainer = Trainer(TicTacToe(), tiny_config)
    trainer.run()
    ckpts = list((tmp_path / tiny_config.checkpoint_dir).glob("iter_*.pt"))
    assert len(ckpts) >= 1


def test_arena_acceptance_replaces_best_net(tiny_config):
    trainer = Trainer(TicTacToe(), tiny_config)
    trainer._run_self_play_iteration()
    for p in trainer.candidate_net.parameters():
        with torch.no_grad():
            p.add_(torch.randn_like(p) * 0.5)
    win_rate_above = tiny_config.arena_threshold + 0.1
    trainer._arena_win_rate = lambda: win_rate_above  # type: ignore[method-assign]
    accepted = trainer._maybe_accept_candidate()
    assert accepted is True
    for p_b, p_c in zip(trainer.best_net.parameters(), trainer.candidate_net.parameters()):
        torch.testing.assert_close(p_b, p_c)


def test_arena_rejection_reverts_candidate_to_best(tiny_config):
    trainer = Trainer(TicTacToe(), tiny_config)
    trainer._run_self_play_iteration()
    best_snapshot = [p.detach().clone() for p in trainer.best_net.parameters()]
    for p in trainer.candidate_net.parameters():
        with torch.no_grad():
            p.add_(torch.randn_like(p) * 0.5)
    trainer._arena_win_rate = lambda: tiny_config.arena_threshold - 0.1  # type: ignore[method-assign]
    accepted = trainer._maybe_accept_candidate()
    assert accepted is False
    for snap, p in zip(best_snapshot, trainer.best_net.parameters()):
        torch.testing.assert_close(snap, p)
    for p_b, p_c in zip(trainer.best_net.parameters(), trainer.candidate_net.parameters()):
        torch.testing.assert_close(p_b, p_c)
