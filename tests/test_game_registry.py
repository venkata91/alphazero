"""Tests for the shared game registry in alphazero.games.__init__.

The registry is the single source of truth for game-name ↔ Game class.
parallel_selfplay._make_game and trainer's parallel-dispatch path both
go through it — these tests prevent silent drift.
"""
from __future__ import annotations

import pytest

from alphazero.games import GAME_REGISTRY, make_game, name_for_instance
from alphazero.games.chess_game import Chess
from alphazero.games.connect4 import Connect4
from alphazero.games.tictactoe import TicTacToe


def test_registry_contains_all_known_games():
    """Adding a new game without registering it should fail this test."""
    assert set(GAME_REGISTRY) == {"tictactoe", "connect4", "chess"}


def test_make_game_returns_correct_class_per_name():
    assert isinstance(make_game("tictactoe"), TicTacToe)
    assert isinstance(make_game("connect4"), Connect4)
    assert isinstance(make_game("chess"), Chess)


def test_make_game_raises_on_unknown_name():
    with pytest.raises(ValueError, match="Unknown game"):
        make_game("nonexistent")


def test_name_for_instance_round_trips():
    """name_for_instance(make_game(name)) == name for every registered game."""
    for name in GAME_REGISTRY:
        instance = make_game(name)
        assert name_for_instance(instance) == name, (
            f"Round-trip broke for {name}: registry name != instance reverse-lookup"
        )


def test_parallel_selfplay_and_trainer_agree_on_game_names():
    """Both parallel_selfplay._make_game and Trainer (via name_for_instance)
    must produce coherent results for every registered game.

    Regression for the previous two-registries-drift footgun: parallel_selfplay
    had its own if/elif registry; trainer had a separate dict. Adding a game
    required editing both. Now both delegate to GAME_REGISTRY.
    """
    from alphazero.games import name_for_instance
    from alphazero.parallel_selfplay import _make_game

    for name in GAME_REGISTRY:
        instance = _make_game(name)
        assert name_for_instance(instance) == name, (
            f"_make_game({name}) returned an instance whose name_for_instance "
            f"resolves to {name_for_instance(instance)!r} — registries are out of sync."
        )
