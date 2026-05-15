"""Trainer: the outer loop coordinating self-play, training, arena, eval."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .config import TrainingConfig
from .games.base import Game
from .network import AlphaZeroNet
from .replay_buffer import ReplayBuffer
from .selfplay import run_one_game


class Trainer:
    def __init__(self, game: Game, config: TrainingConfig):
        self.game = game
        self.config = config

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        self.device = self._resolve_device(config.device)

        self.candidate_net = AlphaZeroNet(
            input_shape=game.input_shape,
            action_size=game.action_size,
            n_blocks=config.n_blocks,
            n_channels=config.n_channels,
        ).to(self.device)
        self.best_net = copy.deepcopy(self.candidate_net)
        self.best_net.eval()

        self.replay_buffer = ReplayBuffer(capacity=config.replay_buffer_capacity)

        self.optimizer = torch.optim.AdamW(
            self.candidate_net.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.iteration = 0

    @staticmethod
    def _resolve_device(name: str) -> torch.device:
        if name == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(name)

    def _make_eval_fn(self, net: AlphaZeroNet) -> Callable[[np.ndarray], tuple[np.ndarray, float]]:
        """Wrap a network as an eval_fn for MCTS."""
        def fn(encoded: np.ndarray) -> tuple[np.ndarray, float]:
            with torch.no_grad():
                x = torch.from_numpy(encoded).float().unsqueeze(0).to(self.device)
                logits, value = net(x)
                priors = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                return priors.astype(np.float32), float(value.item())
        return fn

    def _run_self_play_iteration(self) -> None:
        """Generate games_per_iteration games using best_net; push to buffer."""
        eval_fn = self._make_eval_fn(self.best_net)
        for _ in range(self.config.games_per_iteration):
            examples = run_one_game(
                self.game, eval_fn,
                num_simulations=self.config.num_simulations,
                temperature_threshold=self.config.temperature_threshold,
                c_puct=self.config.c_puct,
                dirichlet_alpha=self.config.dirichlet_alpha,
                dirichlet_weight=self.config.dirichlet_weight,
                augment=True,
            )
            self.replay_buffer.add(examples)

    def _train_step(self) -> float:
        """One minibatch SGD update on candidate_net. Returns total loss."""
        if len(self.replay_buffer) < self.config.batch_size:
            return 0.0
        self.candidate_net.train()

        states, target_pis, target_zs = self.replay_buffer.sample(self.config.batch_size)
        states = states.to(self.device)
        target_pis = target_pis.to(self.device)
        target_zs = target_zs.to(self.device)

        logits, value = self.candidate_net(states)
        log_softmax = torch.log_softmax(logits, dim=-1)
        policy_loss = -(target_pis * log_softmax).sum(dim=-1).mean()
        value_loss = torch.nn.functional.mse_loss(value, target_zs)
        loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())
