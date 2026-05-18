"""Game registry — single source of truth for game-name → Game-class.

Used by:
    - parallel_selfplay._make_game(name) → instantiates a Game from a string
    - trainer._run_parallel_self_play_iteration → maps class name → registry name

Adding a new game: append one entry here and the parallel-self-play path
picks it up automatically. The registry stores **callables** (not class
references) so we can defer the import — heavyweight game modules like
chess_game (which imports python-chess) don't get loaded until needed.
"""
from __future__ import annotations

from typing import Callable

from .base import Game


def _tictactoe() -> Game:
    from .tictactoe import TicTacToe

    return TicTacToe()


def _connect4() -> Game:
    from .connect4 import Connect4

    return Connect4()


def _chess() -> Game:
    from .chess_game import Chess

    return Chess()


# Public registry. Keys are the string names used in configs/CLI.
GAME_REGISTRY: dict[str, Callable[[], Game]] = {
    "tictactoe": _tictactoe,
    "connect4": _connect4,
    "chess": _chess,
}


def make_game(name: str) -> Game:
    """Construct a Game instance by registry name. Raises ValueError on unknown."""
    if name not in GAME_REGISTRY:
        raise ValueError(
            f"Unknown game: {name!r}. Known games: {sorted(GAME_REGISTRY)}"
        )
    return GAME_REGISTRY[name]()


def name_for_instance(game: Game) -> str:
    """Reverse lookup: given a Game instance, return its registry name.

    Raises ValueError if the instance's class isn't registered. Used by
    Trainer to derive the string name that parallel_selfplay needs.
    """
    cls = type(game)
    for name, factory in GAME_REGISTRY.items():
        # Compare via factory output class — avoids importing every game
        # just to compare class identity. We DO eagerly instantiate the
        # candidate; cheap for all known games.
        try:
            candidate_cls = type(factory())
        except Exception:
            continue
        if candidate_cls is cls:
            return name
    raise ValueError(
        f"Game class {cls.__name__} not in GAME_REGISTRY. "
        f"Register it in src/alphazero/games/__init__.py."
    )
