"""Monte Carlo Tree Search guided by a neural network.

MCTS calls a generic `eval_fn(state) -> (priors, value)`; it does not know
the network exists. This boundary makes MCTS testable in isolation with
mock evaluators.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .games.base import Game, State

EvalFn = Callable[[np.ndarray], tuple[np.ndarray, float]]


@dataclass
class Node:
    prior: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, "Node"] = field(default_factory=dict)
    is_expanded: bool = False

    @property
    def Q(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTS:
    """PUCT-guided search."""

    def __init__(
        self,
        game: Game,
        eval_fn: EvalFn,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 1.0,
        dirichlet_weight: float = 0.25,
    ):
        self.game = game
        self.eval_fn = eval_fn
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_weight = dirichlet_weight

    def search(
        self, root_state: State, num_simulations: int, add_root_noise: bool
    ) -> np.ndarray:
        """Run `num_simulations` simulations from root_state.

        Returns the visit distribution at the root as a normalized array of
        shape (action_size,) — this is the "improved policy" π.
        """
        root = Node()
        self._expand(root, root_state, add_root_noise=add_root_noise)

        for _ in range(num_simulations):
            self._simulate(root, root_state)

        visits = np.zeros(self.game.action_size, dtype=np.float32)
        for action, child in root.children.items():
            visits[action] = child.visit_count
        total = visits.sum()
        if total == 0:
            return visits
        return visits / total

    def _simulate(self, root: "Node", root_state: State) -> None:
        """One simulation: select → expand → backup."""
        path: list[Node] = [root]
        state = root_state
        node = root

        # Select
        while node.is_expanded and self.game.terminal_value(state) is None:
            action, child = self._select_child(node)
            state = self.game.apply(state, action)
            node = child
            path.append(node)

        # Evaluate / expand
        terminal = self.game.terminal_value(state)
        if terminal is not None:
            leaf_value = terminal
        else:
            leaf_value = self._expand(node, state, add_root_noise=False)

        # Backup
        self._backup(path, leaf_value)

    def _select_child(self, node: "Node") -> tuple[int, "Node"]:
        """PUCT: argmax over Q + c_puct · P · sqrt(ΣN) / (1 + N)."""
        total_visits = max(1, sum(c.visit_count for c in node.children.values()))
        sqrt_total = np.sqrt(total_visits)

        best_score = -float("inf")
        best_action = -1
        best_child: Node | None = None
        for action, child in node.children.items():
            u = self.c_puct * child.prior * sqrt_total / (1 + child.visit_count)
            score = child.Q + u
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        assert best_child is not None
        return best_action, best_child

    def _expand(self, node: "Node", state: State, add_root_noise: bool) -> float:
        """Call eval_fn, create children for legal actions, return leaf value."""
        canon = self.game.canonical_state(state)
        encoded = self.game.encode(canon)
        priors, leaf_value = self.eval_fn(encoded)

        legal = self.game.legal_actions_mask(state)
        priors = priors * legal
        s = priors.sum()
        if s > 0:
            priors = priors / s
        else:
            priors = legal.astype(np.float32) / legal.sum()

        if add_root_noise:
            priors = self._add_dirichlet_noise(priors, legal)

        for action in np.where(legal)[0]:
            node.children[int(action)] = Node(prior=float(priors[action]))

        node.is_expanded = True
        return float(leaf_value)

    def _backup(self, path: list["Node"], leaf_value: float) -> None:
        """Walk path in reverse, updating N and W, flipping sign per ply."""
        value = leaf_value
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value

    def _add_dirichlet_noise(
        self, priors: np.ndarray, legal_mask: np.ndarray
    ) -> np.ndarray:
        """Mix Dirichlet noise into priors at the root (self-play exploration).

        priors are already legal-masked and renormalized.
        Only legal actions get noise; illegal positions stay at zero.
        """
        legal_indices = np.where(legal_mask)[0]
        if len(legal_indices) == 0:
            return priors
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_indices))
        new_priors = priors.copy()
        for idx, n in zip(legal_indices, noise):
            new_priors[idx] = (
                (1 - self.dirichlet_weight) * priors[idx]
                + self.dirichlet_weight * n
            )
        return new_priors.astype(np.float32)
