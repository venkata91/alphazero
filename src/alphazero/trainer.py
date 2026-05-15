"""Trainer: the outer loop coordinating self-play, training, arena, eval."""
from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from tqdm import tqdm

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

    def _run_self_play_iteration(self, verbose: bool = True) -> None:
        """Generate games_per_iteration games using best_net; push to buffer."""
        eval_fn = self._make_eval_fn(self.best_net)
        iterator = range(self.config.games_per_iteration)
        if verbose:
            iterator = tqdm(
                iterator,
                desc=f"  self-play (iter {self.iteration})",
                leave=False,
            )
        for _ in iterator:
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

    def _make_argmax_mcts_agent(self, net: AlphaZeroNet):
        """Build an MCTS-with-NN agent that plays τ=0, no Dirichlet noise."""
        from .mcts import MCTS

        eval_fn = self._make_eval_fn(net)
        def agent(game, state):
            mcts = MCTS(game, eval_fn, c_puct=self.config.c_puct)
            pi = mcts.search(state, num_simulations=self.config.num_simulations,
                             add_root_noise=False)
            return int(np.argmax(pi))
        return agent

    def _arena_win_rate(self) -> float:
        """Play candidate vs best; return candidate's win rate."""
        from .arena import play_match
        candidate_agent = self._make_argmax_mcts_agent(self.candidate_net)
        best_agent = self._make_argmax_mcts_agent(self.best_net)
        result = play_match(
            self.game, candidate_agent, best_agent,
            num_games=self.config.arena_games,
        )
        return result.win_rate

    def _maybe_accept_candidate(self) -> bool:
        """Run arena gate. If candidate wins >= threshold, accept; else revert."""
        win_rate = self._arena_win_rate()
        if win_rate >= self.config.arena_threshold:
            self.best_net = copy.deepcopy(self.candidate_net)
            self.best_net.eval()
            return True
        self.candidate_net.load_state_dict(self.best_net.state_dict())
        return False

    def _eval_vs_solver(self) -> dict:
        """Play current best_net vs the perfect TTT solver."""
        from .arena import play_match
        from .solvers.tictactoe_solver import solve_tictactoe_action

        def solver_agent(game, state):
            return solve_tictactoe_action(state)

        net_agent = self._make_argmax_mcts_agent(self.best_net)
        result = play_match(
            self.game, net_agent, solver_agent,
            num_games=self.config.eval_games,
        )
        return {
            "wins": result.wins_a,
            "draws": result.draws,
            "losses": result.losses_a,
        }

    def _save_checkpoint(self) -> Path:
        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"iter_{self.iteration:04d}.pt"
        torch.save({
            "iteration": self.iteration,
            "best_net": self.best_net.state_dict(),
            "candidate_net": self.candidate_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)
        return path

    def run(self, verbose: bool = True) -> None:
        if verbose:
            print(
                f"Starting training: {self.config.num_iterations} iterations | "
                f"{self.config.games_per_iteration} games/iter, "
                f"{self.config.num_simulations} MCTS sims, "
                f"net {self.config.n_blocks} blocks × {self.config.n_channels} ch | "
                f"device={self.device}",
                flush=True,
            )
        for _ in range(self.config.num_iterations):
            self.iteration += 1
            iter_start = time.time()
            if verbose:
                print(f"\n[iter {self.iteration}/{self.config.num_iterations}]", flush=True)

            self._run_self_play_iteration(verbose=verbose)
            if verbose:
                print(f"  buffer size: {len(self.replay_buffer)}", flush=True)

            if len(self.replay_buffer) >= self.config.min_buffer_size:
                loss_sum, loss_count = 0.0, 0
                step_iter = range(self.config.training_steps_per_iteration)
                if verbose:
                    step_iter = tqdm(step_iter, desc="  training", leave=False)
                for _step in step_iter:
                    loss_sum += self._train_step()
                    loss_count += 1
                if verbose and loss_count > 0:
                    print(f"  avg loss: {loss_sum / loss_count:.4f}", flush=True)
            elif verbose:
                print(f"  (skipping training — buffer < {self.config.min_buffer_size})", flush=True)

            if self.iteration % self.config.arena_interval == 0:
                if verbose:
                    print(f"  arena: candidate vs best ({self.config.arena_games} games)...",
                          flush=True)
                accepted = self._maybe_accept_candidate()
                if verbose:
                    print(f"  arena: {'ACCEPTED — new best_net' if accepted else 'rejected — reverted'}",
                          flush=True)

            if self.iteration % self.config.eval_interval == 0:
                if verbose:
                    print(f"  eval vs solver ({self.config.eval_games} games)...", flush=True)
                result = self._eval_vs_solver()
                if verbose:
                    print(f"  eval: wins={result['wins']} draws={result['draws']} "
                          f"losses={result['losses']}", flush=True)

            ckpt_path = self._save_checkpoint()
            if verbose:
                elapsed = time.time() - iter_start
                print(f"  checkpoint: {ckpt_path} ({elapsed:.1f}s)", flush=True)
