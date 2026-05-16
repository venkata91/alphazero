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


def test_promote_candidate_to_best_copies_weights(tiny_config):
    """After promotion, best_net has the same weights as candidate."""
    trainer = Trainer(TicTacToe(), tiny_config)
    # Scramble candidate so it definitely differs from best
    for p in trainer.candidate_net.parameters():
        with torch.no_grad():
            p.add_(torch.randn_like(p) * 0.5)
    # Confirm divergence before promotion
    differing = any(
        not torch.equal(p_b, p_c)
        for p_b, p_c in zip(trainer.best_net.parameters(), trainer.candidate_net.parameters())
    )
    assert differing, "test setup failed: candidate already equals best"

    trainer._promote_candidate_to_best()

    # After promotion, all weights match
    for p_b, p_c in zip(trainer.best_net.parameters(), trainer.candidate_net.parameters()):
        torch.testing.assert_close(p_b, p_c)


def test_promote_creates_independent_best_net_copy(tiny_config):
    """Modifying candidate after promotion must NOT change best_net (deepcopy semantics)."""
    trainer = Trainer(TicTacToe(), tiny_config)
    trainer._promote_candidate_to_best()
    # Snapshot best
    best_snapshot = [p.detach().clone() for p in trainer.best_net.parameters()]
    # Mutate candidate
    with torch.no_grad():
        for p in trainer.candidate_net.parameters():
            p.add_(1.0)
    # best_net should be untouched
    for snap, p_b in zip(best_snapshot, trainer.best_net.parameters()):
        torch.testing.assert_close(snap, p_b)


def test_run_promotes_every_iteration(tiny_config):
    """After Trainer.run(), best_net should match the final candidate_net."""
    trainer = Trainer(TicTacToe(), tiny_config)
    trainer.run()
    # After the last iteration, best_net was just promoted from candidate.
    for p_b, p_c in zip(trainer.best_net.parameters(), trainer.candidate_net.parameters()):
        torch.testing.assert_close(p_b, p_c)


def test_eval_vs_opponent_uses_provided_agent(tiny_config):
    """Trainer._eval_vs_opponent should play best_net vs an arbitrary agent."""
    trainer = Trainer(TicTacToe(), tiny_config)

    # Sentinel opponent: always plays the first legal action.
    def sentinel_agent(game, state):
        legal = game.legal_actions_mask(state)
        return int(np.where(legal)[0][0])

    result = trainer._eval_vs_opponent(sentinel_agent, num_games=4)
    assert set(result.keys()) == {"wins", "draws", "losses"}
    assert result["wins"] + result["draws"] + result["losses"] == 4


def test_load_from_checkpoint_restores_iteration_and_weights(tiny_config, tmp_path, monkeypatch):
    """After load_from_checkpoint, iteration counter and net weights match the saved state."""
    monkeypatch.chdir(tmp_path)
    trainer1 = Trainer(TicTacToe(), tiny_config)
    trainer1.run()  # writes checkpoint at iter 2

    ckpt_path = tmp_path / tiny_config.checkpoint_dir / "iter_0002.pt"
    assert ckpt_path.exists()

    trainer2 = Trainer(TicTacToe(), tiny_config)
    trainer2.load_from_checkpoint(ckpt_path)

    assert trainer2.iteration == 2
    for p1, p2 in zip(trainer1.best_net.parameters(), trainer2.best_net.parameters()):
        torch.testing.assert_close(p1, p2)
    for p1, p2 in zip(trainer1.candidate_net.parameters(), trainer2.candidate_net.parameters()):
        torch.testing.assert_close(p1, p2)


def test_run_after_load_continues_from_checkpoint_iteration(tiny_config, tmp_path, monkeypatch):
    """Run after loading from iter_0002 should advance the counter further."""
    monkeypatch.chdir(tmp_path)
    trainer1 = Trainer(TicTacToe(), tiny_config)
    trainer1.run()  # writes through iter 2

    ckpt_path = tmp_path / tiny_config.checkpoint_dir / "iter_0002.pt"

    trainer2 = Trainer(TicTacToe(), tiny_config)
    trainer2.load_from_checkpoint(ckpt_path)
    # tiny_config.num_iterations == 2, so calling run() again should add 2 more iters → 4 total
    trainer2.run()
    assert trainer2.iteration == tiny_config.num_iterations + 2  # 2 + 2 = 4

    # The new checkpoints exist
    assert (tmp_path / tiny_config.checkpoint_dir / "iter_0003.pt").exists()
    assert (tmp_path / tiny_config.checkpoint_dir / "iter_0004.pt").exists()
