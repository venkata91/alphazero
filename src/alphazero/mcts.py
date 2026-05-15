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
        raise NotImplementedError  # Task 12
