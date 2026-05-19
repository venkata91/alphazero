"""Self-play: play one game using MCTS, emit training tuples.

Each ply produces (encoded_canonical_state, mcts_policy, current_player).
After the game terminates, z is filled in per ply from THAT PLY'S mover's POV
and (optionally) symmetry augmentation expands each tuple into 8 for TTT.
"""
from __future__ import annotations

import inspect
from typing import Callable

import numpy as np

from .games.base import Game
from .mcts import MCTS

EvalFn = Callable[[np.ndarray], tuple[np.ndarray, float]]


def play_one_selfplay_game(
    game: Game,
    mcts: MCTS,
    num_simulations: int,
    temperature_threshold: int,
    augment: bool = True,
    rng: np.random.Generator | None = None,
    on_step: Callable[[], None] | None = None,
    on_search_progress: Callable[[int, int, int], None] | None = None,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Play one self-play game with the given MCTS.

    Drives the per-game loop shared by serial (`run_one_game`) and parallel
    (`worker_play_one_game`) self-play: terminal check, MCTS.search,
    temperature-vs-argmax action selection, history collection, _assign_z,
    and optional symmetry augmentation. Construction of the MCTS (with a
    local or remote eval_fn) is the caller's responsibility.

    Returns (encoded_canonical_state, mcts_policy_π, value_z) training tuples.
    """
    state = game.initial_state()
    history: list[tuple[np.ndarray, np.ndarray, int]] = []
    move_idx = 0

    while game.terminal_value(state) is None:
        # Pass RAW state to MCTS — MCTS canonicalizes internally in _expand.
        if on_search_progress is not None and _search_accepts_progress_callback(mcts.search):
            pi = mcts.search(
                state,
                num_simulations=num_simulations,
                add_root_noise=True,
                progress_callback=lambda done, total: on_search_progress(
                    move_idx, done, total
                ),
            )
        else:
            pi = mcts.search(state, num_simulations=num_simulations, add_root_noise=True)

        if move_idx < temperature_threshold:
            # Renormalize: pi = visits/sum is float32 and can drift past
            # numpy's 1e-8 tolerance for `np.random.choice`, especially
            # for chess (4672 actions). Cast to float64 and rescale.
            p = pi.astype(np.float64)
            p /= p.sum()
            if rng is None:
                action = int(np.random.choice(len(p), p=p))
            else:
                action = int(rng.choice(len(p), p=p))
        else:
            action = int(np.argmax(pi))

        canon = game.canonical_state(state)
        encoded = game.encode(canon)
        history.append((encoded, pi.astype(np.float32), game.current_player(state)))

        state = game.apply(state, action)
        move_idx += 1

        if on_step is not None:
            on_step()

    z_per_ply = _assign_z(history, game.terminal_value(state), state, game)

    examples: list[tuple[np.ndarray, np.ndarray, float]] = []
    for (encoded, pi, _player), z in zip(history, z_per_ply):
        if augment:
            for sym_enc, sym_pi in game.symmetries(encoded, pi):
                examples.append((sym_enc.astype(np.float32), sym_pi.astype(np.float32), z))
        else:
            examples.append((encoded, pi, z))
    return examples


def _search_accepts_progress_callback(search_fn) -> bool:
    """Return whether an MCTS-like search method supports progress_callback.

    Tests use lightweight stub MCTS objects with the old 3-argument search
    signature. Introspection keeps those stubs working while real MCTS can
    emit intra-search heartbeats.
    """
    try:
        return "progress_callback" in inspect.signature(search_fn).parameters
    except (TypeError, ValueError):
        return False


def run_one_game(
    game: Game,
    eval_fn: EvalFn,
    num_simulations: int,
    temperature_threshold: int = 6,
    c_puct: float = 1.5,
    dirichlet_alpha: float = 1.0,
    dirichlet_weight: float = 0.25,
    augment: bool = True,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Play one game using MCTS guided by `eval_fn`.

    Returns (encoded_state, mcts_policy_π, value_z) training tuples.
    """
    mcts = MCTS(
        game, eval_fn,
        c_puct=c_puct,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_weight=dirichlet_weight,
    )
    return play_one_selfplay_game(
        game,
        mcts,
        num_simulations=num_simulations,
        temperature_threshold=temperature_threshold,
        augment=augment,
    )


def _assign_z(history, final_value, final_state, game) -> list[float]:
    """Compute z for each ply from THAT PLY'S MOVER'S POV.

    final_value is terminal_value(final_state), from current_player(final_state)'s POV.
    For each ply k with mover P_k, z_k = final_value if P_k == final_current_player
    else -final_value (zero-sum).
    """
    final_current = game.current_player(final_state)
    return [
        final_value if player == final_current else -final_value
        for (_s, _pi, player) in history
    ]
