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
