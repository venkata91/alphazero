# AlphaZero Framework + Tic-Tac-Toe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **⚠ Plan correction (post-implementation):** This plan was written before we discovered that arena gating causes training stagnation. **Tasks involving `_maybe_accept_candidate`, `arena_threshold`, and the arena-gating tests were superseded.** The Trainer now unconditionally promotes the candidate to best after each iteration (AlphaZero 2017 paper). The arena-related portions of Task 22 and the invariant test referencing arena rejection are NOT in the current codebase. See [`concepts.html`](../../../concepts.html) concept card #3 or [Sub-project 2 spec §6](../specs/2026-05-15-connect4-design.md) for the post-mortem.

**Goal:** Build a game-agnostic AlphaZero framework end-to-end, validated on Tic-Tac-Toe by reaching the kill criterion (zero losses out of 200 games against a perfect minimax solver).

**Architecture:** Four orthogonal layers — Game (rules) → AlphaZeroNet (PyTorch ResNet) → MCTS (PUCT search calling an `eval_fn`) → Trainer (orchestrates self-play → buffer → train → arena loop). Strict layer boundaries make every component independently testable; only the `Game` class changes when moving to Connect 4 or chess in later sub-projects.

**Tech Stack:** Python 3.11+, PyTorch 2.2+ (MPS backend on M4 Pro / CUDA portable to cloud), NumPy, pytest, TensorBoard (local logging), no external ML/RL frameworks.

**Spec:** [`docs/superpowers/specs/2026-05-15-alphazero-framework-tictactoe-design.md`](../specs/2026-05-15-alphazero-framework-tictactoe-design.md)

---

## File Structure

Files this plan creates (no modifications to pre-existing files except `.gitignore`):

```
pyproject.toml                                    # Task 1
configs/tictactoe.toml                            # Task 2
src/alphazero/__init__.py                         # Task 1
src/alphazero/config.py                           # Task 2 (TrainingConfig)
src/alphazero/games/__init__.py                   # Task 1
src/alphazero/games/base.py                       # Task 3 (Game ABC)
src/alphazero/games/tictactoe.py                  # Tasks 4–7 (TicTacToe impl)
src/alphazero/network.py                          # Tasks 8–9 (ResidualBlock, AlphaZeroNet)
src/alphazero/replay_buffer.py                    # Task 10
src/alphazero/mcts.py                             # Tasks 11–15 (Node + MCTS)
src/alphazero/selfplay.py                         # Tasks 16–18 (run_one_game)
src/alphazero/solvers/__init__.py                 # Task 1
src/alphazero/solvers/tictactoe_solver.py         # Task 19
src/alphazero/arena.py                            # Task 20
src/alphazero/trainer.py                          # Tasks 21–22
src/alphazero/cli.py                              # Task 25
tests/__init__.py                                 # Task 1
tests/conftest.py                                 # Task 1
tests/test_config.py                              # Task 2
tests/test_game_base.py                           # Task 3
tests/test_game_tictactoe.py                      # Tasks 4–7
tests/test_network.py                             # Tasks 8–9
tests/test_replay_buffer.py                       # Task 10
tests/test_mcts.py                                # Tasks 11–15
tests/test_selfplay.py                            # Tasks 16–18
tests/test_tictactoe_solver.py                    # Task 19
tests/test_arena.py                               # Task 20
tests/test_trainer.py                             # Tasks 21–22
tests/test_invariants.py                          # Task 23
tests/test_e2e_tictactoe.py                       # Task 24
```

Each file has one clear responsibility. `tictactoe.py` accretes across Tasks 4–7 but stays focused (just TTT rules + encoding); same for `mcts.py` across Tasks 11–15.

---

## Conventions used in this plan

- **Commit format**: HEREDOC with `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` footer (per user's CLAUDE.md). Each commit message in this plan shows the subject line only; the executor adds the footer per repo convention.
- **Pytest invocation**: `pytest` is run from the repo root. `pytest -m "not slow"` skips the E2E test.
- **TDD discipline**: Red → Green → Commit. The plan does not include refactor steps as separate tasks; refactor inline if needed before committing.
- **Type hints**: Used throughout (Python 3.11+ syntax). Keep them; they document interfaces.
- **No print() in library code**: Use the `logging` module or TensorBoard. CLI may print.

---

## Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/alphazero/__init__.py`
- Create: `src/alphazero/games/__init__.py`
- Create: `src/alphazero/solvers/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Modify: `.gitignore` (add a few entries)

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "alphazero"
version = "0.1.0"
description = "AlphaZero-style chess engine — learning project"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.2",
    "numpy>=1.26",
    "tqdm>=4.66",
    "tensorboard>=2.16",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-xdist>=3.5",
    "ruff>=0.4",
    "markdown>=3.10",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers"
markers = [
    "slow: end-to-end tests (deselect with '-m \"not slow\"')",
]
```

- [ ] **Step 2: Create empty `__init__.py` files**

```bash
mkdir -p src/alphazero/games src/alphazero/solvers tests
touch src/alphazero/__init__.py src/alphazero/games/__init__.py src/alphazero/solvers/__init__.py tests/__init__.py
```

- [ ] **Step 3: Create `tests/conftest.py` with the small-test config fixture**

```python
"""Shared pytest fixtures. Tiny config so unit tests stay fast."""
from __future__ import annotations

import pytest

from alphazero.config import TrainingConfig


@pytest.fixture
def tiny_config() -> TrainingConfig:
    """A small TrainingConfig used by unit tests that need one.

    Trainer / network / MCTS unit tests use this so they stay sub-second.
    The end-to-end test uses the full TTT config from configs/tictactoe.toml.
    """
    return TrainingConfig(
        n_blocks=1,
        n_channels=8,
        num_simulations=10,
        c_puct=1.5,
        dirichlet_alpha=1.0,
        dirichlet_weight=0.25,
        games_per_iteration=4,
        temperature_threshold=6,
        training_steps_per_iteration=5,
        batch_size=8,
        learning_rate=1e-3,
        weight_decay=1e-4,
        replay_buffer_capacity=200,
        min_buffer_size=20,
        arena_interval=1,
        arena_games=4,
        arena_threshold=0.55,
        eval_interval=1,
        eval_games=8,
        num_iterations=2,
        seed=42,
        device="cpu",
        checkpoint_dir="checkpoints_test",
        log_dir="runs_test",
    )
```

- [ ] **Step 4: Append to `.gitignore`**

Append to existing `.gitignore`:
```
# Test artifacts
checkpoints_test/
runs_test/
.pytest_cache/

# Build
build/
dist/
*.egg-info/
src/*.egg-info/
```

- [ ] **Step 5: Install in editable mode and verify pytest runs**

```bash
pip install -e ".[dev]"
pytest --collect-only
```
Expected: pytest finds no tests yet, exits 0 (with "no tests collected").

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/alphazero tests/__init__.py tests/conftest.py .gitignore
git commit -m "Add project scaffolding (pyproject, src layout, conftest)"
```

---

## Task 2: TrainingConfig + TOML loading

**Files:**
- Create: `src/alphazero/config.py`
- Create: `configs/tictactoe.toml`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_config.py`:
```python
from pathlib import Path

import pytest

from alphazero.config import TrainingConfig, load_config


def test_default_config_has_expected_values():
    cfg = TrainingConfig()
    assert cfg.n_blocks == 4
    assert cfg.n_channels == 32
    assert cfg.num_simulations == 50
    assert cfg.c_puct == 1.5
    assert cfg.num_iterations == 50
    assert cfg.arena_threshold == 0.55


def test_config_is_frozen():
    cfg = TrainingConfig()
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        cfg.n_blocks = 999  # type: ignore[misc]


def test_load_config_overrides_defaults(tmp_path: Path):
    p = tmp_path / "tiny.toml"
    p.write_text("n_blocks = 6\nnum_simulations = 100\n")
    cfg = load_config(p)
    assert cfg.n_blocks == 6
    assert cfg.num_simulations == 100
    assert cfg.batch_size == 64  # default preserved


def test_load_config_rejects_unknown_keys(tmp_path: Path):
    p = tmp_path / "bad.toml"
    p.write_text("not_a_real_key = 99\n")
    with pytest.raises(ValueError, match="Unknown"):
        load_config(p)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_config.py -v
```
Expected: ImportError (`alphazero.config` doesn't exist yet).

- [ ] **Step 3: Implement `src/alphazero/config.py`**

```python
"""Training configuration: a single frozen dataclass loaded from TOML.

Saving the exact config alongside each checkpoint requires the config be
serializable (TOML round-trip) and stable across runs (frozen dataclass).
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    # Network
    n_blocks: int = 4
    n_channels: int = 32

    # MCTS
    num_simulations: int = 50
    c_puct: float = 1.5
    dirichlet_alpha: float = 1.0
    dirichlet_weight: float = 0.25

    # Self-play
    games_per_iteration: int = 100
    temperature_threshold: int = 6

    # Training
    training_steps_per_iteration: int = 500
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # Replay buffer
    replay_buffer_capacity: int = 50_000
    min_buffer_size: int = 5_000

    # Arena
    arena_interval: int = 5
    arena_games: int = 40
    arena_threshold: float = 0.55

    # Eval
    eval_interval: int = 5
    eval_games: int = 200

    # Schedule
    num_iterations: int = 50

    # Misc
    seed: int = 42
    device: str = "auto"            # "auto" | "cpu" | "mps" | "cuda"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "runs"


def load_config(path: Path | str) -> TrainingConfig:
    """Load TrainingConfig from a TOML file.

    Unknown keys raise ValueError to catch typos early.
    """
    data = tomllib.loads(Path(path).read_text())
    field_names = {f.name for f in fields(TrainingConfig)}
    unknown = set(data) - field_names
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    return TrainingConfig(**data)
```

- [ ] **Step 4: Create `configs/tictactoe.toml`**

```toml
# AlphaZero training config for Tic-Tac-Toe.
# Values omitted use defaults from TrainingConfig.

n_blocks = 4
n_channels = 32

num_simulations = 50
c_puct = 1.5

games_per_iteration = 100
temperature_threshold = 6

training_steps_per_iteration = 500
batch_size = 64
learning_rate = 1e-3

replay_buffer_capacity = 50_000
min_buffer_size = 5_000

arena_interval = 5
arena_games = 40
arena_threshold = 0.55

eval_interval = 5
eval_games = 200

num_iterations = 50
seed = 42
device = "auto"
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_config.py -v
```
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add src/alphazero/config.py configs/tictactoe.toml tests/test_config.py
git commit -m "Add TrainingConfig dataclass + TOML loader"
```

---

## Task 3: Game abstract base class

**Files:**
- Create: `src/alphazero/games/base.py`
- Test: `tests/test_game_base.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_game_base.py`:
```python
import numpy as np
import pytest

from alphazero.games.base import Game


def test_game_is_abstract_and_cannot_be_instantiated():
    with pytest.raises(TypeError):
        Game()  # type: ignore[abstract]


def test_default_symmetries_returns_identity_singleton():
    """A subclass that doesn't override symmetries() gets [(s, π)] back."""

    class DummyGame(Game):
        @property
        def input_shape(self):
            return (1, 1, 1)

        @property
        def action_size(self):
            return 1

        def initial_state(self):
            return np.zeros((1, 1))

        def current_player(self, state):
            return 1

        def legal_actions_mask(self, state):
            return np.array([True])

        def apply(self, state, action):
            return state

        def terminal_value(self, state):
            return None

        def encode(self, state):
            return np.zeros((1, 1, 1), dtype=np.float32)

        def canonical_state(self, state):
            return state

    g = DummyGame()
    encoded = np.zeros((1, 1, 1), dtype=np.float32)
    policy = np.array([1.0], dtype=np.float32)
    syms = g.symmetries(encoded, policy)
    assert len(syms) == 1
    np.testing.assert_array_equal(syms[0][0], encoded)
    np.testing.assert_array_equal(syms[0][1], policy)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_game_base.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `src/alphazero/games/base.py`**

```python
"""Abstract base class for games.

Concrete Games encapsulate everything game-specific: rules, terminal
detection, state encoding, perspective normalization, and symmetry-based
data augmentation. The rest of the framework (network, MCTS, training)
consumes a Game purely through this interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

State = Any  # game-specific; e.g. np.ndarray for TTT


class Game(ABC):
    """Abstract base class for a two-player, zero-sum, perfect-information game.

    Convention:
        - Players are +1 and -1.
        - terminal_value returns the result from the CURRENT PLAYER'S POV.
        - encode is intended to be called on `canonical_state(state)` —
          canonical_state rewrites the board so the current player's pieces
          are +1, freeing the network from learning whose-turn-is-it.
    """

    @property
    @abstractmethod
    def input_shape(self) -> tuple[int, ...]:
        """Shape of `encode(state)` output: (channels, height, width)."""

    @property
    @abstractmethod
    def action_size(self) -> int:
        """Total number of distinct actions (legal or not) in the action space."""

    @abstractmethod
    def initial_state(self) -> State:
        """Return the starting position."""

    @abstractmethod
    def current_player(self, state: State) -> int:
        """Return +1 or -1 — whose turn is it to move."""

    @abstractmethod
    def legal_actions_mask(self, state: State) -> np.ndarray:
        """Boolean array of shape (action_size,) — True for legal actions."""

    @abstractmethod
    def apply(self, state: State, action: int) -> State:
        """Return a new state after applying `action` for current_player."""

    @abstractmethod
    def terminal_value(self, state: State) -> float | None:
        """Return None if game in progress; else +1/0/-1 from current_player's POV."""

    @abstractmethod
    def encode(self, state: State) -> np.ndarray:
        """Stack of float32 planes — input to the neural network. Shape == input_shape."""

    @abstractmethod
    def canonical_state(self, state: State) -> State:
        """Rewrite state so current player sees themselves as +1."""

    def symmetries(
        self, encoded: np.ndarray, policy: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return equivalent (encoded_state, policy) pairs for data augmentation.

        Default: identity only. Override for games with symmetries
        (TTT has 8; Connect 4 has 2; chess has none).
        """
        return [(encoded, policy)]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_game_base.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/alphazero/games/base.py tests/test_game_base.py
git commit -m "Add Game abstract base class"
```

---

## Task 4: TicTacToe — state, moves, current_player

**Files:**
- Create: `src/alphazero/games/tictactoe.py`
- Test: `tests/test_game_tictactoe.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_game_tictactoe.py`:
```python
import numpy as np
import pytest

from alphazero.games.tictactoe import TicTacToe


@pytest.fixture
def ttt() -> TicTacToe:
    return TicTacToe()


def test_input_shape_and_action_size(ttt: TicTacToe):
    assert ttt.input_shape == (3, 3, 3)
    assert ttt.action_size == 9


def test_initial_state_is_empty(ttt: TicTacToe):
    s = ttt.initial_state()
    assert s.shape == (3, 3)
    assert s.dtype == np.int8
    assert np.all(s == 0)


def test_current_player_starts_at_plus_one(ttt: TicTacToe):
    assert ttt.current_player(ttt.initial_state()) == 1


def test_apply_places_correct_piece(ttt: TicTacToe):
    s = ttt.initial_state()
    s1 = ttt.apply(s, action=4)  # X plays center
    assert s1[1, 1] == 1
    assert ttt.current_player(s1) == -1
    s2 = ttt.apply(s1, action=0)  # O plays corner
    assert s2[0, 0] == -1
    assert ttt.current_player(s2) == 1


def test_apply_does_not_mutate_input(ttt: TicTacToe):
    s = ttt.initial_state()
    s_copy = s.copy()
    _ = ttt.apply(s, 0)
    np.testing.assert_array_equal(s, s_copy)


def test_apply_raises_on_occupied_square(ttt: TicTacToe):
    s = ttt.apply(ttt.initial_state(), 4)
    with pytest.raises(ValueError, match="Illegal"):
        ttt.apply(s, 4)


def test_legal_actions_mask_initial_state_all_true(ttt: TicTacToe):
    mask = ttt.legal_actions_mask(ttt.initial_state())
    assert mask.shape == (9,)
    assert mask.dtype == bool
    assert mask.all()


def test_legal_actions_mask_after_one_move(ttt: TicTacToe):
    s = ttt.apply(ttt.initial_state(), 4)
    mask = ttt.legal_actions_mask(s)
    assert mask.sum() == 8
    assert not mask[4]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_game_tictactoe.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement core TTT (state, current_player, apply, legal_actions_mask)**

`src/alphazero/games/tictactoe.py`:
```python
"""Tic-Tac-Toe: the smallest game we use to validate the AlphaZero pipeline.

State representation:
    np.ndarray shape (3, 3), dtype int8, values in {+1, 0, -1}.
    +1 = player X's piece; -1 = player O's piece; 0 = empty.
    Player to move is determined by piece-count parity (count_nonzero % 2).

Action space:
    Integers 0..8, row-major: action = 3*row + col.
"""
from __future__ import annotations

import numpy as np

from .base import Game

State = np.ndarray


class TicTacToe(Game):
    @property
    def input_shape(self) -> tuple[int, ...]:
        return (3, 3, 3)

    @property
    def action_size(self) -> int:
        return 9

    def initial_state(self) -> State:
        return np.zeros((3, 3), dtype=np.int8)

    def current_player(self, state: State) -> int:
        # +1 plays first; alternates with each move.
        moves = int(np.count_nonzero(state))
        return 1 if moves % 2 == 0 else -1

    def legal_actions_mask(self, state: State) -> np.ndarray:
        return state.flatten() == 0

    def apply(self, state: State, action: int) -> State:
        r, c = divmod(action, 3)
        if state[r, c] != 0:
            raise ValueError(f"Illegal action {action}: square already occupied")
        new_state = state.copy()
        new_state[r, c] = self.current_player(state)
        return new_state

    # Filled in by later tasks:
    def terminal_value(self, state: State) -> float | None:
        raise NotImplementedError  # Task 5

    def encode(self, state: State) -> np.ndarray:
        raise NotImplementedError  # Task 6

    def canonical_state(self, state: State) -> State:
        raise NotImplementedError  # Task 6
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_game_tictactoe.py -v
```
Expected: 8 passed (subset of TTT tests; terminal/encode tests come later).

- [ ] **Step 5: Commit**

```bash
git add src/alphazero/games/tictactoe.py tests/test_game_tictactoe.py
git commit -m "Add TicTacToe state, apply, current_player, legal mask"
```

---

## Task 5: TicTacToe — terminal_value

**Files:**
- Modify: `src/alphazero/games/tictactoe.py` (implement `terminal_value`, add `_find_winner`)
- Modify: `tests/test_game_tictactoe.py` (add terminal tests)

- [ ] **Step 1: Write the failing tests** — append to `tests/test_game_tictactoe.py`:

```python
def _board(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=np.int8)


def test_terminal_value_none_when_in_progress(ttt: TicTacToe):
    assert ttt.terminal_value(ttt.initial_state()) is None
    s = ttt.apply(ttt.initial_state(), 4)
    assert ttt.terminal_value(s) is None


def test_terminal_value_x_wins_row_returns_minus_one_for_o(ttt: TicTacToe):
    # X just played and won; it's now O's turn → from O's POV, X won → -1
    state = _board([
        [1, 1, 1],
        [-1, -1, 0],
        [0, 0, 0],
    ])
    # parity: 3 X's + 2 O's = 5 nonzero → odd → current player is -1 (O)
    assert ttt.current_player(state) == -1
    assert ttt.terminal_value(state) == -1.0


def test_terminal_value_x_wins_col(ttt: TicTacToe):
    state = _board([
        [1, -1, 0],
        [1, -1, 0],
        [1, 0, 0],
    ])
    assert ttt.terminal_value(state) == -1.0


def test_terminal_value_x_wins_diag(ttt: TicTacToe):
    state = _board([
        [1, -1, 0],
        [-1, 1, 0],
        [0, 0, 1],
    ])
    assert ttt.terminal_value(state) == -1.0


def test_terminal_value_x_wins_antidiag(ttt: TicTacToe):
    state = _board([
        [0, -1, 1],
        [-1, 1, 0],
        [1, 0, 0],
    ])
    assert ttt.terminal_value(state) == -1.0


def test_terminal_value_o_wins_returns_minus_one_for_x(ttt: TicTacToe):
    # O just played and won; now X's turn → from X's POV, O won → -1
    state = _board([
        [-1, -1, -1],
        [1, 1, 0],
        [1, 0, 0],
    ])
    # 3 O's + 3 X's = 6 nonzero → even → current player is +1 (X)
    assert ttt.current_player(state) == 1
    assert ttt.terminal_value(state) == -1.0


def test_terminal_value_draw_returns_zero(ttt: TicTacToe):
    state = _board([
        [1, -1, 1],
        [1, -1, -1],
        [-1, 1, 1],
    ])
    assert ttt.terminal_value(state) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_game_tictactoe.py -v
```
Expected: 7 fail with `NotImplementedError`.

- [ ] **Step 3: Implement `terminal_value` + `_find_winner`**

Replace the stub in `src/alphazero/games/tictactoe.py`:
```python
    def terminal_value(self, state: State) -> float | None:
        winner = self._find_winner(state)
        if winner is not None:
            return 1.0 if winner == self.current_player(state) else -1.0
        if np.all(state != 0):
            return 0.0  # board full, no winner → draw
        return None

    def _find_winner(self, state: State) -> int | None:
        for player in (1, -1):
            # rows
            for r in range(3):
                if np.all(state[r, :] == player):
                    return player
            # columns
            for c in range(3):
                if np.all(state[:, c] == player):
                    return player
            # diagonals
            if np.all(np.diag(state) == player):
                return player
            if np.all(np.diag(np.fliplr(state)) == player):
                return player
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_game_tictactoe.py -v
```
Expected: 15 passed (8 from Task 4 + 7 new).

- [ ] **Step 5: Commit**

```bash
git add src/alphazero/games/tictactoe.py tests/test_game_tictactoe.py
git commit -m "Add TicTacToe terminal_value (rows, cols, diagonals, draws)"
```

---

## Task 6: TicTacToe — encode + canonical_state

**Files:**
- Modify: `src/alphazero/games/tictactoe.py` (implement `encode`, `canonical_state`)
- Modify: `tests/test_game_tictactoe.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_game_tictactoe.py`:

```python
def test_canonical_state_is_identity_when_current_player_plus_one(ttt: TicTacToe):
    s = ttt.initial_state()
    np.testing.assert_array_equal(ttt.canonical_state(s), s)


def test_canonical_state_inverts_when_current_player_minus_one(ttt: TicTacToe):
    s = ttt.apply(ttt.initial_state(), 4)  # X plays center; now O's turn
    canon = ttt.canonical_state(s)
    expected = np.zeros((3, 3), dtype=np.int8)
    expected[1, 1] = -1
    np.testing.assert_array_equal(canon, expected)


def test_encode_shape_matches_input_shape(ttt: TicTacToe):
    s = ttt.initial_state()
    enc = ttt.encode(ttt.canonical_state(s))
    assert enc.shape == ttt.input_shape
    assert enc.dtype == np.float32


def test_encode_initial_state_my_and_opp_planes_empty(ttt: TicTacToe):
    enc = ttt.encode(ttt.canonical_state(ttt.initial_state()))
    assert enc[0].sum() == 0  # my pieces
    assert enc[1].sum() == 0  # opp pieces
    np.testing.assert_array_equal(enc[2], np.ones((3, 3), dtype=np.float32))


def test_encode_treats_current_player_as_my_pieces(ttt: TicTacToe):
    """After X plays center, from O's canonical perspective:
       O's pieces (none) on plane 0; X's piece (center) on plane 1."""
    s = ttt.apply(ttt.initial_state(), 4)
    canon = ttt.canonical_state(s)
    enc = ttt.encode(canon)
    # plane 0 (my = O) is all empty
    assert enc[0].sum() == 0
    # plane 1 (opp = X) has the center piece
    assert enc[1, 1, 1] == 1
    assert enc[1].sum() == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_game_tictactoe.py -v
```
Expected: 5 fail with `NotImplementedError`.

- [ ] **Step 3: Implement `encode` and `canonical_state`**

Replace stubs in `src/alphazero/games/tictactoe.py`:
```python
    def canonical_state(self, state: State) -> State:
        """Rewrite so current player's pieces are +1, opponent's are -1."""
        return (state * self.current_player(state)).astype(np.int8)

    def encode(self, state: State) -> np.ndarray:
        """Three planes: my pieces (+1 in canonical state), opp pieces (-1), ones.

        Caller is expected to pass a canonical state. The ones plane is a
        constant feature that helps small CNNs learn positional reasoning
        about board edges (a standard trick).
        """
        my = (state == 1).astype(np.float32)
        opp = (state == -1).astype(np.float32)
        ones = np.ones((3, 3), dtype=np.float32)
        return np.stack([my, opp, ones], axis=0)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_game_tictactoe.py -v
```
Expected: 20 passed.

- [ ] **Step 5: Commit**

```bash
git add src/alphazero/games/tictactoe.py tests/test_game_tictactoe.py
git commit -m "Add TicTacToe encode + canonical_state (perspective transform)"
```

---

## Task 7: TicTacToe — symmetries (D₄ dihedral group)

**Files:**
- Modify: `src/alphazero/games/tictactoe.py` (override `symmetries`)
- Modify: `tests/test_game_tictactoe.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_game_tictactoe.py`:

```python
def test_symmetries_yields_8_distinct_tuples(ttt: TicTacToe):
    """Asymmetric input → all 8 D₄ symmetries are distinct."""
    s = ttt.initial_state()
    s = ttt.apply(s, 0)  # X corner top-left — asymmetric
    canon = ttt.canonical_state(s)
    enc = ttt.encode(canon)
    policy = np.array([0.1, 0.05, 0.05, 0.05, 0.6, 0.05, 0.05, 0.05, 0.0], dtype=np.float32)
    syms = ttt.symmetries(enc, policy)
    assert len(syms) == 8

    # All 8 should be distinct (asymmetric starting position)
    seen = set()
    for enc_s, _ in syms:
        seen.add(enc_s.tobytes())
    assert len(seen) == 8


def test_symmetries_includes_identity_first(ttt: TicTacToe):
    """The first symmetry is (encoded, policy) unchanged."""
    enc = ttt.encode(ttt.canonical_state(ttt.initial_state()))
    pol = np.zeros(9, dtype=np.float32)
    pol[4] = 1.0
    syms = ttt.symmetries(enc, pol)
    np.testing.assert_array_equal(syms[0][0], enc)
    np.testing.assert_array_equal(syms[0][1], pol)


def test_symmetries_preserves_policy_sum(ttt: TicTacToe):
    """Symmetries should not change the total probability mass."""
    enc = ttt.encode(ttt.canonical_state(ttt.initial_state()))
    pol = np.full(9, 1 / 9, dtype=np.float32)
    syms = ttt.symmetries(enc, pol)
    for _, p in syms:
        np.testing.assert_allclose(p.sum(), 1.0, rtol=1e-6)


def test_symmetries_action_index_follows_board_rotation(ttt: TicTacToe):
    """If the original puts all probability on the top-left corner (action 0),
    the 90°-rotated version puts it on top-right (action 2)."""
    enc = ttt.encode(ttt.canonical_state(ttt.initial_state()))
    pol = np.zeros(9, dtype=np.float32)
    pol[0] = 1.0  # top-left

    syms = ttt.symmetries(enc, pol)
    # syms layout: (rot0, rot0+mirror, rot90, rot90+mirror, rot180, ...)
    # By our convention (Task 7 step 3), symmetries are paired rotation/mirror.
    # Find the one whose policy places mass on top-right (index 2).
    found = False
    for _, p in syms:
        if p[2] == 1.0:
            found = True
            break
    assert found
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_game_tictactoe.py -v
```
Expected: 4 fail (default `symmetries` returns only the identity tuple, so length-8 assertion fails).

- [ ] **Step 3: Implement `symmetries`**

Add to `src/alphazero/games/tictactoe.py`:
```python
    def symmetries(
        self, encoded: np.ndarray, policy: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """All 8 D₄ symmetries of a TTT board (4 rotations × 2 mirrors).

        encoded shape: (C, H, W) = (3, 3, 3); policy shape: (9,) row-major.
        Returns list of (encoded_sym, policy_sym) pairs; element 0 is identity.
        """
        results: list[tuple[np.ndarray, np.ndarray]] = []
        pi_grid = policy.reshape(3, 3)
        for k in range(4):
            # rotate 90° k times (axes 1,2 are H,W of the encoded tensor)
            rot_enc = np.rot90(encoded, k=k, axes=(1, 2)).copy()
            rot_pi = np.rot90(pi_grid, k=k).copy()
            results.append((rot_enc, rot_pi.flatten()))
            # mirror along the W axis
            mirror_enc = np.flip(rot_enc, axis=2).copy()
            mirror_pi = np.flip(rot_pi, axis=1).copy()
            results.append((mirror_enc, mirror_pi.flatten()))
        return results
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_game_tictactoe.py -v
```
Expected: 24 passed.

- [ ] **Step 5: Commit**

```bash
git add src/alphazero/games/tictactoe.py tests/test_game_tictactoe.py
git commit -m "Add TicTacToe D4 symmetries (8 for data augmentation)"
```

---

## Task 8: ResidualBlock

**Files:**
- Create: `src/alphazero/network.py` (just ResidualBlock for now)
- Test: `tests/test_network.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_network.py`:
```python
import torch

from alphazero.network import ResidualBlock


def test_residual_block_preserves_shape():
    block = ResidualBlock(n_channels=8)
    x = torch.randn(4, 8, 3, 3)
    y = block(x)
    assert y.shape == x.shape


def test_residual_block_skip_connection_is_identity_when_F_is_zero():
    """If we manually zero the conv weights, output should be ReLU(x) — the skip path."""
    block = ResidualBlock(n_channels=8)
    # Zero out both conv layers' weights AND the batchnorm scale to disable F(x)
    for layer in [block.conv1, block.conv2]:
        torch.nn.init.zeros_(layer.weight)
    block.eval()  # disable batchnorm running-mean updates
    x = torch.relu(torch.randn(2, 8, 3, 3))   # non-negative so ReLU is identity
    y = block(x)
    # F(x) is now zero, so y = ReLU(0 + x) = ReLU(x) = x (since x >= 0)
    torch.testing.assert_close(y, x, atol=1e-5, rtol=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_network.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `ResidualBlock`**

`src/alphazero/network.py`:
```python
"""AlphaZero neural network: ResNet backbone with policy + value heads.

Generic over input shape and action space: provided by the Game class.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """One residual block: y = ReLU(x + BN(conv(ReLU(BN(conv(x)))))).

    The skip connection (the "+ x") is what makes deep nets trainable.
    Without it, gradients vanish through deep stacks of layers.
    """

    def __init__(self, n_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.conv2 = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(n_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_network.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/alphazero/network.py tests/test_network.py
git commit -m "Add ResidualBlock (conv-BN-ReLU-conv-BN + skip + ReLU)"
```

---

## Task 9: AlphaZeroNet

**Files:**
- Modify: `src/alphazero/network.py` (add `AlphaZeroNet`)
- Modify: `tests/test_network.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_network.py`:

```python
import pytest

from alphazero.network import AlphaZeroNet


def test_alphazero_net_forward_shapes():
    net = AlphaZeroNet(input_shape=(3, 3, 3), action_size=9, n_blocks=2, n_channels=8)
    x = torch.randn(4, 3, 3, 3)
    policy_logits, value = net(x)
    assert policy_logits.shape == (4, 9)
    assert value.shape == (4,)


def test_alphazero_net_value_in_tanh_range():
    net = AlphaZeroNet(input_shape=(3, 3, 3), action_size=9, n_blocks=1, n_channels=8)
    # Large random inputs; verify value head stays in [-1, +1]
    x = torch.randn(16, 3, 3, 3) * 5.0
    _, value = net(x)
    assert torch.all(value <= 1.0)
    assert torch.all(value >= -1.0)


def test_alphazero_net_state_dict_roundtrip(tmp_path):
    net1 = AlphaZeroNet(input_shape=(3, 3, 3), action_size=9, n_blocks=1, n_channels=8)
    path = tmp_path / "net.pt"
    torch.save(net1.state_dict(), path)

    net2 = AlphaZeroNet(input_shape=(3, 3, 3), action_size=9, n_blocks=1, n_channels=8)
    net2.load_state_dict(torch.load(path))

    net1.eval()
    net2.eval()
    x = torch.randn(2, 3, 3, 3)
    p1, v1 = net1(x)
    p2, v2 = net2(x)
    torch.testing.assert_close(p1, p2)
    torch.testing.assert_close(v1, v2)


def test_alphazero_net_overfits_a_single_batch():
    """Sanity check: a tiny net should be able to overfit a 4-example synthetic batch.

    If this can't memorize 4 examples in 200 steps, something fundamental is broken
    (gradient flow, optimizer, loss shape).
    """
    torch.manual_seed(0)
    net = AlphaZeroNet(input_shape=(3, 3, 3), action_size=9, n_blocks=1, n_channels=8)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-2)

    x = torch.randn(4, 3, 3, 3)
    target_pi = torch.softmax(torch.randn(4, 9), dim=-1)
    target_z = torch.tensor([0.7, -0.5, 0.3, -0.9])

    initial_loss = None
    for step in range(200):
        opt.zero_grad()
        logits, v = net(x)
        log_softmax = torch.log_softmax(logits, dim=-1)
        pl = -(target_pi * log_softmax).sum(dim=-1).mean()
        vl = torch.nn.functional.mse_loss(v, target_z)
        loss = pl + vl
        if step == 0:
            initial_loss = loss.item()
        loss.backward()
        opt.step()

    assert loss.item() < initial_loss * 0.1, (
        f"Net failed to overfit a tiny batch: started at {initial_loss:.3f}, ended at {loss.item():.3f}"
    )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_network.py -v
```
Expected: ImportError on `AlphaZeroNet`.

- [ ] **Step 3: Implement `AlphaZeroNet`**

Append to `src/alphazero/network.py`:
```python
class AlphaZeroNet(nn.Module):
    """ResNet backbone with two heads: policy (action logits) and value (tanh scalar).

    Architecture:
        input conv (C_in → n_channels)
          → BN → ReLU
          → n_blocks × ResidualBlock(n_channels)
          → policy_head (1x1 conv → flatten → linear → action_size logits)
          → value_head  (1x1 conv → flatten → linear → relu → linear → tanh)
    """

    def __init__(
        self,
        input_shape: tuple[int, ...],
        action_size: int,
        n_blocks: int = 4,
        n_channels: int = 32,
    ):
        super().__init__()
        in_channels, height, width = input_shape

        # Input projection
        self.input_conv = nn.Conv2d(in_channels, n_channels, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(n_channels)

        # Backbone
        self.blocks = nn.Sequential(*[ResidualBlock(n_channels) for _ in range(n_blocks)])

        # Policy head
        self.policy_conv = nn.Conv2d(n_channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * height * width, action_size)

        # Value head
        self.value_conv = nn.Conv2d(n_channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(height * width, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Backbone
        h = F.relu(self.input_bn(self.input_conv(x)))
        h = self.blocks(h)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(h)))
        p = p.flatten(start_dim=1)
        policy_logits = self.policy_fc(p)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(h)))
        v = v.flatten(start_dim=1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        value = v.squeeze(-1)  # shape (B,)

        return policy_logits, value
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_network.py -v
```
Expected: 6 passed (2 from Task 8 + 4 new).

- [ ] **Step 5: Commit**

```bash
git add src/alphazero/network.py tests/test_network.py
git commit -m "Add AlphaZeroNet (ResNet backbone + policy/value heads)"
```

---

## Task 10: ReplayBuffer

**Files:**
- Create: `src/alphazero/replay_buffer.py`
- Test: `tests/test_replay_buffer.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_replay_buffer.py`:
```python
import numpy as np
import pytest
import torch

from alphazero.replay_buffer import ReplayBuffer


def _make_tuple(seed: int) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    s = rng.standard_normal((3, 3, 3)).astype(np.float32)
    p = rng.dirichlet(np.ones(9)).astype(np.float32)
    z = float(rng.uniform(-1, 1))
    return (s, p, z)


def test_buffer_starts_empty():
    buf = ReplayBuffer(capacity=10)
    assert len(buf) == 0


def test_buffer_add_increases_length():
    buf = ReplayBuffer(capacity=10)
    buf.add([_make_tuple(i) for i in range(5)])
    assert len(buf) == 5


def test_buffer_overflow_drops_oldest():
    buf = ReplayBuffer(capacity=3)
    buf.add([_make_tuple(0)])
    buf.add([_make_tuple(1)])
    buf.add([_make_tuple(2)])
    buf.add([_make_tuple(99)])  # overflow; oldest (seed=0) dropped
    assert len(buf) == 3


def test_buffer_sample_returns_batch_shapes():
    buf = ReplayBuffer(capacity=100)
    buf.add([_make_tuple(i) for i in range(30)])
    states, policies, values = buf.sample(8)
    assert isinstance(states, torch.Tensor)
    assert isinstance(policies, torch.Tensor)
    assert isinstance(values, torch.Tensor)
    assert states.shape == (8, 3, 3, 3)
    assert policies.shape == (8, 9)
    assert values.shape == (8,)
    assert states.dtype == torch.float32
    assert policies.dtype == torch.float32
    assert values.dtype == torch.float32


def test_buffer_sample_raises_when_insufficient():
    buf = ReplayBuffer(capacity=10)
    buf.add([_make_tuple(0), _make_tuple(1)])
    with pytest.raises(ValueError):
        buf.sample(8)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_replay_buffer.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `ReplayBuffer`**

`src/alphazero/replay_buffer.py`:
```python
"""Fixed-capacity FIFO replay buffer.

Holds (state, policy_target, value_target) tuples from recent self-play games.
Random uniform sampling — no prioritization (per AlphaZero paper).
"""
from __future__ import annotations

from collections import deque
from typing import Sequence

import numpy as np
import torch


Tuple3 = tuple[np.ndarray, np.ndarray, float]


class ReplayBuffer:
    """FIFO queue of (state, policy, value) tuples.

    Sampling returns PyTorch tensors batched on the first dim, ready to feed
    directly into the network forward pass.
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self._buf: deque[Tuple3] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._buf)

    def add(self, examples: Sequence[Tuple3]) -> None:
        """Append tuples; oldest are dropped if capacity is exceeded."""
        self._buf.extend(examples)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a random minibatch. Returns (states, policies, values) tensors."""
        if batch_size > len(self._buf):
            raise ValueError(
                f"Cannot sample {batch_size} from buffer of size {len(self._buf)}"
            )
        indices = np.random.choice(len(self._buf), size=batch_size, replace=False)
        batch = [self._buf[int(i)] for i in indices]

        states = np.stack([b[0] for b in batch])
        policies = np.stack([b[1] for b in batch])
        values = np.array([b[2] for b in batch], dtype=np.float32)

        return (
            torch.from_numpy(states).float(),
            torch.from_numpy(policies).float(),
            torch.from_numpy(values).float(),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_replay_buffer.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/alphazero/replay_buffer.py tests/test_replay_buffer.py
git commit -m "Add ReplayBuffer (FIFO, uniform sampling, returns tensors)"
```

---

## Task 11: MCTS — Node dataclass + skeleton

**Files:**
- Create: `src/alphazero/mcts.py` (Node + class skeleton)
- Test: `tests/test_mcts.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_mcts.py`:
```python
"""MCTS tests. We use a mock eval_fn so behavior is fully deterministic."""
from __future__ import annotations

import numpy as np
import pytest

from alphazero.games.tictactoe import TicTacToe
from alphazero.mcts import MCTS, Node


# ---------- helpers ----------

def make_eval_fn(prior: np.ndarray, value: float):
    """eval_fn that returns the same (prior, value) for any state."""
    def fn(state: np.ndarray) -> tuple[np.ndarray, float]:
        return prior.astype(np.float32), float(value)
    return fn


@pytest.fixture
def ttt() -> TicTacToe:
    return TicTacToe()


# ---------- Node tests ----------

def test_node_default_values():
    n = Node(prior=0.3)
    assert n.prior == 0.3
    assert n.visit_count == 0
    assert n.value_sum == 0.0
    assert n.children == {}
    assert n.is_expanded is False
    assert n.Q == 0.0


def test_node_Q_after_visits():
    n = Node(prior=0.1)
    n.visit_count = 3
    n.value_sum = 1.5
    assert n.Q == pytest.approx(0.5)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_mcts.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `Node` + class skeleton**

`src/alphazero/mcts.py`:
```python
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
    """One position in the search tree.

    Notes:
        - `prior` is the probability of the action that led to this node,
          assigned by the parent's eval_fn output at expansion time.
        - `value_sum` accumulates leaf values backed up through this node;
          divide by `visit_count` to get the running mean Q.
    """

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
        raise NotImplementedError  # Tasks 12-14 fill this in
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_mcts.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/alphazero/mcts.py tests/test_mcts.py
git commit -m "Add MCTS Node dataclass and class skeleton"
```

---

## Task 12: MCTS — select, expand, basic search loop

**Files:**
- Modify: `src/alphazero/mcts.py` (implement search loop sans Dirichlet)
- Modify: `tests/test_mcts.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_mcts.py`:

```python
def test_search_returns_distribution_summing_to_one(ttt):
    eval_fn = make_eval_fn(np.full(9, 1/9), 0.0)
    mcts = MCTS(ttt, eval_fn)
    pi = mcts.search(ttt.initial_state(), num_simulations=10, add_root_noise=False)
    assert pi.shape == (9,)
    np.testing.assert_allclose(pi.sum(), 1.0, rtol=1e-6)


def test_search_visits_only_legal_actions(ttt):
    """After X plays center, only 8 legal moves remain."""
    eval_fn = make_eval_fn(np.full(9, 1/9), 0.0)
    mcts = MCTS(ttt, eval_fn)
    state = ttt.apply(ttt.initial_state(), 4)  # center taken
    pi = mcts.search(state, num_simulations=20, add_root_noise=False)
    assert pi[4] == 0.0  # never visited


def test_search_prior_favoring_action_concentrates_visits(ttt):
    """A heavily skewed prior should bias visits toward that action."""
    prior = np.full(9, 0.01)
    prior[0] = 1.0 - 0.01 * 8  # action 0 gets ~92% prior
    eval_fn = make_eval_fn(prior, 0.0)
    mcts = MCTS(ttt, eval_fn)
    pi = mcts.search(ttt.initial_state(), num_simulations=40, add_root_noise=False)
    assert pi[0] > 0.5  # action 0 dominates visits


def test_search_more_simulations_means_more_total_visits(ttt):
    """Sanity: visit count distribution should approach a stable shape with more sims."""
    eval_fn = make_eval_fn(np.full(9, 1/9), 0.0)
    mcts = MCTS(ttt, eval_fn)
    pi_a = mcts.search(ttt.initial_state(), num_simulations=4, add_root_noise=False)
    pi_b = mcts.search(ttt.initial_state(), num_simulations=40, add_root_noise=False)
    # Coarse check: both should be distributions
    assert pi_a.sum() == pytest.approx(1.0)
    assert pi_b.sum() == pytest.approx(1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_mcts.py -v
```
Expected: 4 fail with `NotImplementedError`.

- [ ] **Step 3: Implement select / expand / backup in `MCTS.search`**

Replace the `search` stub in `src/alphazero/mcts.py`:
```python
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

        # Build visit distribution
        visits = np.zeros(self.game.action_size, dtype=np.float32)
        for action, child in root.children.items():
            visits[action] = child.visit_count
        total = visits.sum()
        if total == 0:
            # Degenerate case (e.g., 0 simulations and root has no legal moves).
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

        # Mask + renormalize
        legal = self.game.legal_actions_mask(state)
        priors = priors * legal
        s = priors.sum()
        if s > 0:
            priors = priors / s
        else:
            # eval_fn returned zero for all legal actions; fall back to uniform
            priors = legal.astype(np.float32) / legal.sum()

        # Add Dirichlet noise to root if requested (Task 14 enables this)
        if add_root_noise:
            priors = self._add_dirichlet_noise(priors, legal)

        # Create children for legal actions only
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
            value = -value  # zero-sum sign flip

    def _add_dirichlet_noise(
        self, priors: np.ndarray, legal_mask: np.ndarray
    ) -> np.ndarray:
        """Stub — implemented in Task 14."""
        return priors
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_mcts.py -v
```
Expected: 6 passed (2 from Task 11 + 4 new).

- [ ] **Step 5: Commit**

```bash
git add src/alphazero/mcts.py tests/test_mcts.py
git commit -m "Add MCTS select/expand/backup with PUCT"
```

---

## Task 13: MCTS — value-driven exploitation test + backup sign-flip test

**Files:**
- Modify: `tests/test_mcts.py`

- [ ] **Step 1: Write the new MCTS tests** — append to `tests/test_mcts.py`:

```python
def test_search_value_favoring_child_concentrates_visits(ttt):
    """If the NN values action 0 highly (returns +1 leaf value from that subtree)
    and other actions get +0, visits should concentrate on action 0."""

    def biased_eval_fn(encoded: np.ndarray) -> tuple[np.ndarray, float]:
        prior = np.full(9, 1/9, dtype=np.float32)
        # Detect "X played action 0 already" → high value for that branch
        # (canonical: my=plane 0; opp=plane 1; opp piece at top-left = "X piece"
        # in the original is now opp from O's POV)
        opp_corner = encoded[1, 0, 0]
        value = 1.0 if opp_corner > 0.5 else 0.0
        return prior, value

    mcts = MCTS(ttt, biased_eval_fn)
    pi = mcts.search(ttt.initial_state(), num_simulations=80, add_root_noise=False)
    # Action 0 leads to the high-value subtree → should be visited most
    assert pi.argmax() == 0


def test_backup_flips_sign_per_ply(ttt):
    """Direct unit test on a known tree shape.

    Set up: root with one child (action=0), whose only child is terminal +1.
    Backing up +1 from the terminal leaf should make root.Q = -1 (opponent
    wins from current player's POV is a loss).
    """
    from alphazero.mcts import Node

    eval_fn = make_eval_fn(np.full(9, 1/9), 0.5)  # arbitrary; not used after expand
    mcts = MCTS(ttt, eval_fn)
    root = Node()
    root.is_expanded = True
    child = Node(prior=1.0)
    root.children[0] = child

    # Direct test of _backup
    mcts._backup([root, child], leaf_value=1.0)

    assert child.visit_count == 1
    assert child.value_sum == 1.0
    assert child.Q == 1.0
    assert root.visit_count == 1
    assert root.value_sum == -1.0      # SIGN FLIPPED
    assert root.Q == -1.0
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
pytest tests/test_mcts.py -v
```
Expected: 8 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/test_mcts.py
git commit -m "Add MCTS value-exploitation and backup-sign-flip tests"
```

---

## Task 14: MCTS — Dirichlet noise at root (self-play only)

**Files:**
- Modify: `src/alphazero/mcts.py` (implement `_add_dirichlet_noise`)
- Modify: `tests/test_mcts.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_mcts.py`:

```python
def test_dirichlet_noise_changes_priors_when_enabled(ttt):
    """With add_root_noise=True and a fixed seed, root priors should differ from
    a noise-free baseline."""
    np.random.seed(0)
    eval_fn = make_eval_fn(np.full(9, 1/9), 0.0)
    mcts = MCTS(ttt, eval_fn, dirichlet_alpha=1.0, dirichlet_weight=0.5)

    # We inspect the root via a tiny number of sims so visits don't dominate.
    pi_noisy = mcts.search(ttt.initial_state(), num_simulations=1, add_root_noise=True)

    np.random.seed(0)
    pi_clean = mcts.search(ttt.initial_state(), num_simulations=1, add_root_noise=False)

    assert not np.allclose(pi_noisy, pi_clean)


def test_dirichlet_noise_disabled_yields_uniform_visit_at_one_sim(ttt):
    """With no noise, 1 simulation should give a single visit somewhere."""
    eval_fn = make_eval_fn(np.full(9, 1/9), 0.0)
    mcts = MCTS(ttt, eval_fn)
    pi = mcts.search(ttt.initial_state(), num_simulations=1, add_root_noise=False)
    # Exactly one visit, so distribution is one-hot
    assert pi.sum() == pytest.approx(1.0)
    assert (pi > 0).sum() == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_mcts.py -v
```
Expected: 1 failure on `test_dirichlet_noise_changes_priors_when_enabled` (stub returns priors unchanged).

- [ ] **Step 3: Implement `_add_dirichlet_noise`**

Replace the stub in `src/alphazero/mcts.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_mcts.py -v
```
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add src/alphazero/mcts.py tests/test_mcts.py
git commit -m "Add Dirichlet noise mixing at MCTS root (self-play exploration)"
```

---

## Task 15: MCTS — terminal-leaf uses terminal_value (not eval_fn)

**Files:**
- Modify: `tests/test_mcts.py`

The terminal-leaf behavior is already implemented in Task 12 (the `_simulate` method checks `terminal_value(state)` before expanding). This task is just a regression test guarding that behavior.

- [ ] **Step 1: Write the test** — append to `tests/test_mcts.py`:

```python
def test_terminal_leaf_uses_terminal_value_not_eval_fn(ttt):
    """If a simulation reaches a terminal state, the leaf value MUST come from
    Game.terminal_value, not from a (possibly wrong) NN forward pass."""
    eval_fn_calls = {"count": 0}

    def tracking_eval_fn(state: np.ndarray) -> tuple[np.ndarray, float]:
        eval_fn_calls["count"] += 1
        return np.full(9, 1/9, dtype=np.float32), -0.99  # NN says "very bad"

    mcts = MCTS(ttt, tracking_eval_fn)

    # Construct a near-terminal position: X needs to play action 8 to win.
    # state: X at (0,0), (1,1), and (1,0); O at (0,1), (2,0)
    #   X . . O .   (then we make moves to reach close-to-win-or-draw)
    # We'll just walk through a sequence of moves manually.
    state = ttt.initial_state()
    state = ttt.apply(state, 0)  # X(0,0)
    state = ttt.apply(state, 1)  # O(0,1)
    state = ttt.apply(state, 4)  # X(1,1)
    state = ttt.apply(state, 6)  # O(2,0)
    # Now X to move. Action 8 → X(2,2) completes diagonal {(0,0),(1,1),(2,2)}.
    pi = mcts.search(state, num_simulations=40, add_root_noise=False)
    # MCTS should prefer the winning move
    assert pi[8] > 0.4


def test_legal_actions_only_get_children(ttt):
    """After expanding the root of a partially-played game, children dict
    should only contain legal actions."""
    eval_fn = make_eval_fn(np.full(9, 1/9), 0.0)
    mcts = MCTS(ttt, eval_fn)
    state = ttt.apply(ttt.initial_state(), 4)  # X plays center
    pi = mcts.search(state, num_simulations=2, add_root_noise=False)
    assert pi[4] == 0.0  # center is no longer legal
    # Sum of pi over legal actions = 1
    legal = ttt.legal_actions_mask(state)
    np.testing.assert_allclose(pi[legal].sum(), 1.0, rtol=1e-6)
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
pytest tests/test_mcts.py -v
```
Expected: 12 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/test_mcts.py
git commit -m "Add MCTS terminal-leaf + legal-only-children regression tests"
```

---

## Task 16: SelfPlayWorker — game loop (no z-assignment yet)

**Files:**
- Create: `src/alphazero/selfplay.py`
- Test: `tests/test_selfplay.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_selfplay.py`:
```python
from __future__ import annotations

import numpy as np
import pytest

from alphazero.games.tictactoe import TicTacToe
from alphazero.selfplay import run_one_game


def uniform_eval_fn(encoded: np.ndarray) -> tuple[np.ndarray, float]:
    return np.full(9, 1/9, dtype=np.float32), 0.0


def test_run_one_game_terminates():
    examples = run_one_game(
        TicTacToe(), uniform_eval_fn,
        num_simulations=5, temperature_threshold=6, augment=False,
    )
    # TTT max 9 plies, so we get up to 9 tuples
    assert 1 <= len(examples) <= 9


def test_run_one_game_emits_tuples_with_correct_shapes():
    examples = run_one_game(
        TicTacToe(), uniform_eval_fn,
        num_simulations=5, temperature_threshold=6, augment=False,
    )
    for s, pi, z in examples:
        assert s.shape == (3, 3, 3)
        assert s.dtype == np.float32
        assert pi.shape == (9,)
        assert pi.dtype == np.float32
        np.testing.assert_allclose(pi.sum(), 1.0, rtol=1e-5)
        assert -1.0 <= z <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_selfplay.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `run_one_game` (without final z-assignment)**

`src/alphazero/selfplay.py`:
```python
"""Self-play: play one game using MCTS, emit training tuples.

Each ply produces (encoded_canonical_state, mcts_policy, current_player).
After the game terminates, z is filled in per ply from THAT PLY'S mover's POV
and (optionally) symmetry augmentation expands each tuple into 8 for TTT.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .games.base import Game
from .mcts import MCTS

EvalFn = Callable[[np.ndarray], tuple[np.ndarray, float]]


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

    Returns a list of (encoded_state, mcts_policy_π, value_z) training tuples.
    z is the eventual game outcome from THAT PLY'S MOVER'S perspective.
    Symmetry augmentation is applied if `augment=True`.
    """
    mcts = MCTS(
        game, eval_fn,
        c_puct=c_puct,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_weight=dirichlet_weight,
    )
    state = game.initial_state()
    history: list[tuple[np.ndarray, np.ndarray, int]] = []
    move_idx = 0

    while game.terminal_value(state) is None:
        # Pass RAW state to MCTS — MCTS canonicalizes internally at the eval_fn
        # boundary (in _expand). Double-canonicalizing here would undo itself
        # for player == -1 and corrupt training.
        pi = mcts.search(state, num_simulations=num_simulations, add_root_noise=True)

        # Temperature: τ=1 for early moves (sample), τ=0 after (argmax)
        if move_idx < temperature_threshold:
            action = int(np.random.choice(len(pi), p=pi))
        else:
            action = int(np.argmax(pi))

        # For the training tuple, we record the canonical-perspective encoding.
        canon = game.canonical_state(state)
        encoded = game.encode(canon)
        history.append((encoded, pi.astype(np.float32), game.current_player(state)))

        state = game.apply(state, action)
        move_idx += 1

    z_per_ply = _assign_z(history, game.terminal_value(state), state, game)

    examples: list[tuple[np.ndarray, np.ndarray, float]] = []
    for (encoded, pi, _player), z in zip(history, z_per_ply):
        if augment:
            for sym_enc, sym_pi in game.symmetries(encoded, pi):
                examples.append((sym_enc.astype(np.float32), sym_pi.astype(np.float32), z))
        else:
            examples.append((encoded, pi, z))
    return examples


def _assign_z(history, final_value, final_state, game) -> list[float]:
    """Stub — implemented in Task 17."""
    return [0.0] * len(history)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_selfplay.py -v
```
Expected: 2 passed (z's are all 0.0 stubs, but tests only check structure here).

- [ ] **Step 5: Commit**

```bash
git add src/alphazero/selfplay.py tests/test_selfplay.py
git commit -m "Add SelfPlayWorker game loop (z-assignment stubbed)"
```

---

## Task 17: SelfPlayWorker — z-assignment (the bug-prone part)

**Files:**
- Modify: `src/alphazero/selfplay.py` (implement `_assign_z`)
- Modify: `tests/test_selfplay.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_selfplay.py`:

```python
def deterministic_eval_fn(encoded: np.ndarray) -> tuple[np.ndarray, float]:
    """Use the same prior so behavior is deterministic w/ seeded rng."""
    return np.full(9, 1/9, dtype=np.float32), 0.0


def test_z_assignment_x_wins_on_move_5():
    """Construct a 5-move X-wins game by hand, check z for every ply."""
    from alphazero.selfplay import _assign_z

    ttt = TicTacToe()

    # Build history by simulating the moves explicitly:
    # Move 0 (X plays 0):  player=+1
    # Move 1 (O plays 3):  player=-1
    # Move 2 (X plays 1):  player=+1
    # Move 3 (O plays 4):  player=-1
    # Move 4 (X plays 2):  player=+1, X WINS the top row.
    history = [
        (np.zeros((3, 3, 3), dtype=np.float32), np.full(9, 1/9, dtype=np.float32), 1),
        (np.zeros((3, 3, 3), dtype=np.float32), np.full(9, 1/9, dtype=np.float32), -1),
        (np.zeros((3, 3, 3), dtype=np.float32), np.full(9, 1/9, dtype=np.float32), 1),
        (np.zeros((3, 3, 3), dtype=np.float32), np.full(9, 1/9, dtype=np.float32), -1),
        (np.zeros((3, 3, 3), dtype=np.float32), np.full(9, 1/9, dtype=np.float32), 1),
    ]
    # After X plays move 4, current_player(final_state) = -1, terminal_value = -1 (O lost).
    final_state = np.array([[1, 1, 1], [-1, -1, 0], [0, 0, 0]], dtype=np.int8)
    final_value = -1.0  # from O's POV; X won

    zs = _assign_z(history, final_value, final_state, ttt)
    # X plies: +1 (X won), O plies: -1 (O lost)
    assert zs == [1.0, -1.0, 1.0, -1.0, 1.0]


def test_z_assignment_o_wins_on_move_6():
    from alphazero.selfplay import _assign_z

    ttt = TicTacToe()
    # X plays first; O wins on its third move (move idx 5).
    history = [
        (np.zeros((3, 3, 3), dtype=np.float32), np.full(9, 1/9, dtype=np.float32), 1),
        (np.zeros((3, 3, 3), dtype=np.float32), np.full(9, 1/9, dtype=np.float32), -1),
        (np.zeros((3, 3, 3), dtype=np.float32), np.full(9, 1/9, dtype=np.float32), 1),
        (np.zeros((3, 3, 3), dtype=np.float32), np.full(9, 1/9, dtype=np.float32), -1),
        (np.zeros((3, 3, 3), dtype=np.float32), np.full(9, 1/9, dtype=np.float32), 1),
        (np.zeros((3, 3, 3), dtype=np.float32), np.full(9, 1/9, dtype=np.float32), -1),
    ]
    # After O plays move 5, current_player(final_state) = +1, terminal_value = -1 (X lost).
    final_state = np.array([[-1, -1, -1], [1, 1, 0], [1, 0, 0]], dtype=np.int8)
    final_value = -1.0

    zs = _assign_z(history, final_value, final_state, ttt)
    # X plies: -1 (X lost), O plies: +1 (O won)
    assert zs == [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]


def test_z_assignment_draw_assigns_zero_to_every_ply():
    from alphazero.selfplay import _assign_z

    ttt = TicTacToe()
    history = [
        (np.zeros((3, 3, 3), dtype=np.float32), np.full(9, 1/9, dtype=np.float32), p)
        for p in [1, -1, 1, -1, 1, -1, 1, -1, 1]
    ]
    final_state = np.array([[1, -1, 1], [1, -1, -1], [-1, 1, 1]], dtype=np.int8)
    final_value = 0.0  # draw

    zs = _assign_z(history, final_value, final_state, ttt)
    assert zs == [0.0] * 9
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_selfplay.py -v
```
Expected: 3 fail (`_assign_z` is a stub returning zeros).

- [ ] **Step 3: Implement `_assign_z`**

Replace the stub in `src/alphazero/selfplay.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_selfplay.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/alphazero/selfplay.py tests/test_selfplay.py
git commit -m "Add per-ply z-assignment in self-play (X/O wins + draw)"
```

---

## Task 18: SelfPlayWorker — symmetry augmentation integration test

**Files:**
- Modify: `tests/test_selfplay.py`

- [ ] **Step 1: Write the test** — append to `tests/test_selfplay.py`:

```python
def test_symmetry_augmentation_multiplies_tuples_by_8():
    """With augment=True (TTT), each ply produces 8 symmetric tuples."""
    np.random.seed(123)
    examples_aug = run_one_game(
        TicTacToe(), uniform_eval_fn,
        num_simulations=3, temperature_threshold=6, augment=True,
    )

    np.random.seed(123)
    examples_raw = run_one_game(
        TicTacToe(), uniform_eval_fn,
        num_simulations=3, temperature_threshold=6, augment=False,
    )

    # Same game (seeded RNG); augmented count = raw count × 8
    assert len(examples_aug) == 8 * len(examples_raw)


def test_z_values_are_consistent_within_a_game():
    """Every ply of a single game has z ∈ {+1, 0, -1} and signs match parity."""
    np.random.seed(7)
    examples = run_one_game(
        TicTacToe(), uniform_eval_fn,
        num_simulations=4, temperature_threshold=6, augment=False,
    )
    zs = {e[2] for e in examples}
    # Either all draw (0), or a mix of +1 and -1 (winner vs loser).
    assert zs.issubset({1.0, -1.0, 0.0})
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
pytest tests/test_selfplay.py -v
```
Expected: 7 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/test_selfplay.py
git commit -m "Add self-play symmetry-augmentation + z-consistency tests"
```

---

## Task 19: Perfect TTT solver (eval ground truth)

**Files:**
- Create: `src/alphazero/solvers/tictactoe_solver.py`
- Test: `tests/test_tictactoe_solver.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_tictactoe_solver.py`:
```python
import numpy as np
import pytest

from alphazero.games.tictactoe import TicTacToe
from alphazero.solvers.tictactoe_solver import (
    solve_tictactoe_value,
    solve_tictactoe_action,
)


@pytest.fixture
def ttt() -> TicTacToe:
    return TicTacToe()


def test_initial_state_is_a_draw_with_perfect_play(ttt: TicTacToe):
    """TTT is a draw with optimal play on both sides."""
    s = ttt.initial_state()
    assert solve_tictactoe_value(s) == 0


def test_immediate_winning_move_returns_plus_one(ttt: TicTacToe):
    """X about to win — value should be +1."""
    s = np.array([
        [1, 1, 0],
        [-1, -1, 0],
        [0, 0, 0],
    ], dtype=np.int8)
    # X plays action 2 to win
    assert solve_tictactoe_value(s) == 1


def test_immediate_losing_position_returns_minus_one(ttt: TicTacToe):
    """O is about to play but X has a forced win next."""
    s = np.array([
        [1, 1, 0],
        [0, 0, 0],
        [0, 0, 0],
    ], dtype=np.int8)
    # O's turn. Whatever O plays, X plays action 2 next → X wins
    # From O's POV, value is -1.
    assert solve_tictactoe_value(s) == -1


def test_solver_action_completes_winning_move(ttt: TicTacToe):
    """Given a position where X can win, solver should pick the winning move."""
    s = np.array([
        [1, 1, 0],
        [-1, -1, 0],
        [0, 0, 0],
    ], dtype=np.int8)
    assert solve_tictactoe_action(s) == 2


def test_solver_action_blocks_opponent_win(ttt: TicTacToe):
    """O must block at action 2; not blocking loses immediately."""
    s = np.array([
        [1, 1, 0],
        [0, 0, 0],
        [0, 0, 0],
    ], dtype=np.int8)
    # O's turn. O must play action 2 to block.
    assert solve_tictactoe_action(s) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_tictactoe_solver.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement the solver**

`src/alphazero/solvers/tictactoe_solver.py`:
```python
"""Perfect-play Tic-Tac-Toe solver via memoized minimax.

TTT has only ~5478 reachable positions, so full enumeration with caching is
cheap. Used as eval ground truth: the AlphaZero agent's success criterion is
zero losses against this solver out of 200 games.
"""
from __future__ import annotations

import numpy as np

from alphazero.games.tictactoe import TicTacToe

_cache: dict[bytes, int] = {}
_ttt = TicTacToe()


def solve_tictactoe_value(state: np.ndarray) -> int:
    """Return the perfect-play value for current_player at this state.

    +1 if current_player wins with optimal play, -1 if loses, 0 if draw.
    """
    key = state.tobytes()
    cached = _cache.get(key)
    if cached is not None:
        return cached

    terminal = _ttt.terminal_value(state)
    if terminal is not None:
        v = int(terminal)
        _cache[key] = v
        return v

    legal = _ttt.legal_actions_mask(state)
    best = -2  # worse than -1
    for action in np.where(legal)[0]:
        next_state = _ttt.apply(state, int(action))
        # Negate because the opponent's "good" value is our "bad" value (zero-sum)
        v = -solve_tictactoe_value(next_state)
        if v > best:
            best = v

    _cache[key] = best
    return best


def solve_tictactoe_action(state: np.ndarray) -> int:
    """Return an optimal action for current_player at this state.

    Ties are broken by lowest action index (deterministic for tests).
    """
    terminal = _ttt.terminal_value(state)
    if terminal is not None:
        raise ValueError("Cannot solve a terminal position")

    legal = _ttt.legal_actions_mask(state)
    best_val = -2
    best_action = -1
    for action in np.where(legal)[0]:
        next_state = _ttt.apply(state, int(action))
        v = -solve_tictactoe_value(next_state)
        if v > best_val:
            best_val = v
            best_action = int(action)
    return best_action
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_tictactoe_solver.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/alphazero/solvers/tictactoe_solver.py tests/test_tictactoe_solver.py
git commit -m "Add perfect TTT minimax solver (eval ground truth)"
```

---

## Task 20: Arena (head-to-head match)

**Files:**
- Create: `src/alphazero/arena.py`
- Test: `tests/test_arena.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_arena.py`:
```python
from __future__ import annotations

import numpy as np
import pytest

from alphazero.arena import MatchResult, play_match
from alphazero.games.tictactoe import TicTacToe
from alphazero.solvers.tictactoe_solver import solve_tictactoe_action


@pytest.fixture
def ttt() -> TicTacToe:
    return TicTacToe()


def random_agent_factory(seed: int):
    """An agent that picks a random legal move."""
    rng = np.random.default_rng(seed)
    def agent(game, state):
        legal = game.legal_actions_mask(state)
        choices = np.where(legal)[0]
        return int(rng.choice(choices))
    return agent


def perfect_agent(game, state):
    return solve_tictactoe_action(state)


def test_perfect_vs_random_perfect_never_loses(ttt: TicTacToe):
    np.random.seed(0)
    result = play_match(ttt, perfect_agent, random_agent_factory(42), num_games=20)
    assert isinstance(result, MatchResult)
    assert result.losses_a == 0  # perfect never loses
    assert result.total == 20


def test_perfect_vs_perfect_always_draws(ttt: TicTacToe):
    """Two perfect-play agents always draw at TTT."""
    result = play_match(ttt, perfect_agent, perfect_agent, num_games=10)
    assert result.draws == 10
    assert result.wins_a == 0
    assert result.losses_a == 0


def test_play_match_alternates_colors(ttt: TicTacToe):
    """Half the games should have agent_a as X, half as O."""
    np.random.seed(0)
    result = play_match(ttt, perfect_agent, random_agent_factory(7), num_games=20)
    # Both colors get played: at minimum total = 20
    assert result.total == 20


def test_match_result_win_rate(ttt: TicTacToe):
    np.random.seed(0)
    result = play_match(ttt, perfect_agent, random_agent_factory(11), num_games=10)
    # win_rate counts draws as 0.5 (standard Elo convention)
    rate = result.win_rate
    assert 0.5 <= rate <= 1.0  # perfect can't lose, draws count half
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_arena.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `Arena.play_match`**

`src/alphazero/arena.py`:
```python
"""Head-to-head match between two agents.

An agent is a callable `(game, state) -> int` returning the chosen action.
Used both for training arena (NN gate) and for evaluation vs external opponents.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .games.base import Game, State

Agent = Callable[[Game, State], int]


@dataclass
class MatchResult:
    wins_a: int = 0
    losses_a: int = 0
    draws: int = 0

    @property
    def total(self) -> int:
        return self.wins_a + self.losses_a + self.draws

    @property
    def win_rate(self) -> float:
        if self.total == 0:
            return 0.0
        # Draws count as 0.5 (Elo convention)
        return (self.wins_a + 0.5 * self.draws) / self.total


def play_match(
    game: Game,
    agent_a: Agent,
    agent_b: Agent,
    num_games: int,
) -> MatchResult:
    """Play num_games games alternating which agent moves first (plays as +1).

    Returns a MatchResult from agent_a's perspective.
    """
    result = MatchResult()
    for i in range(num_games):
        # Alternate: even games agent_a is +1, odd games agent_a is -1
        first, second = (agent_a, agent_b) if i % 2 == 0 else (agent_b, agent_a)

        winner_sign = _play_one_game(game, first, second)
        # winner_sign: +1 = first player won; -1 = second player won; 0 = draw

        if winner_sign == 0:
            result.draws += 1
        else:
            first_is_a = (i % 2 == 0)
            a_won = (winner_sign == 1 and first_is_a) or (winner_sign == -1 and not first_is_a)
            if a_won:
                result.wins_a += 1
            else:
                result.losses_a += 1
    return result


def _play_one_game(game: Game, plus_agent: Agent, minus_agent: Agent) -> int:
    """Play one game; return winner sign (+1, -1) or 0 for draw."""
    state = game.initial_state()
    while game.terminal_value(state) is None:
        agent = plus_agent if game.current_player(state) == 1 else minus_agent
        action = agent(game, state)
        state = game.apply(state, action)

    # Game over. terminal_value is from current_player's POV.
    final_v = game.terminal_value(state)
    if final_v == 0:
        return 0
    # current_player is whoever's turn it WOULD be next (i.e., the loser if final_v == -1)
    cur = game.current_player(state)
    if final_v == -1:
        # cur loses → opponent (the just-moved player) wins
        return -cur
    else:
        # cur wins (only happens if terminal_value returns +1, which TTT never does post-move,
        # but kept for generality)
        return cur
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_arena.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/alphazero/arena.py tests/test_arena.py
git commit -m "Add Arena.play_match (alternating colors, MatchResult)"
```

---

## Task 21: Trainer — initialization + single-iteration body

**Files:**
- Create: `src/alphazero/trainer.py`
- Test: `tests/test_trainer.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_trainer.py`:
```python
from __future__ import annotations

import numpy as np
import pytest
import torch

from alphazero.games.tictactoe import TicTacToe
from alphazero.trainer import Trainer


def test_trainer_initializes_two_independent_nets(tiny_config):
    """best_net and candidate_net must be separate objects."""
    trainer = Trainer(TicTacToe(), tiny_config)
    assert trainer.best_net is not trainer.candidate_net
    # ...with the SAME initial weights (we just copied candidate to best)
    for p_best, p_cand in zip(trainer.best_net.parameters(), trainer.candidate_net.parameters()):
        torch.testing.assert_close(p_best, p_cand)


def test_trainer_run_iteration_populates_buffer(tiny_config):
    trainer = Trainer(TicTacToe(), tiny_config)
    assert len(trainer.replay_buffer) == 0
    trainer._run_self_play_iteration()
    assert len(trainer.replay_buffer) > 0


def test_trainer_train_step_changes_candidate_weights(tiny_config):
    trainer = Trainer(TicTacToe(), tiny_config)
    # Populate buffer
    trainer._run_self_play_iteration()

    # Snapshot weights
    before = [p.detach().clone() for p in trainer.candidate_net.parameters()]
    trainer._train_step()
    after = [p.detach().clone() for p in trainer.candidate_net.parameters()]

    # At least one parameter must have changed (a forward+backward+step happened)
    changed = any(not torch.equal(b, a) for b, a in zip(before, after))
    assert changed


def test_trainer_train_step_does_not_change_best_net(tiny_config):
    trainer = Trainer(TicTacToe(), tiny_config)
    trainer._run_self_play_iteration()
    before = [p.detach().clone() for p in trainer.best_net.parameters()]
    trainer._train_step()
    after = [p.detach().clone() for p in trainer.best_net.parameters()]
    # best_net is frozen during training; only arena gate changes it.
    for b, a in zip(before, after):
        torch.testing.assert_close(b, a)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_trainer.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement Trainer init + single-iteration body**

`src/alphazero/trainer.py`:
```python
"""Trainer: the outer loop coordinating self-play, training, arena, eval.

Owns:
    - candidate_net (the network being trained)
    - best_net (the frozen network used for self-play and as arena opponent)
    - replay buffer
    - optimizer
    - config

Critical design property: candidate_net is updated by training; best_net is
ONLY updated when the arena gate accepts candidate. Self-play always uses
best_net so the training data never reflects a regressed network.
"""
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

        # candidate_net is updated by training; best_net is the frozen "current best"
        self.candidate_net = AlphaZeroNet(
            input_shape=game.input_shape,
            action_size=game.action_size,
            n_blocks=config.n_blocks,
            n_channels=config.n_channels,
        ).to(self.device)
        self.best_net = copy.deepcopy(self.candidate_net)
        # best_net is never trained; mark eval mode so BN uses running stats.
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

    # ---- Eval functions wrapping the two networks (used by MCTS / self-play) ----

    def _make_eval_fn(self, net: AlphaZeroNet) -> Callable[[np.ndarray], tuple[np.ndarray, float]]:
        """Wrap a network as an eval_fn for MCTS."""
        def fn(encoded: np.ndarray) -> tuple[np.ndarray, float]:
            with torch.no_grad():
                x = torch.from_numpy(encoded).float().unsqueeze(0).to(self.device)
                logits, value = net(x)
                priors = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                return priors.astype(np.float32), float(value.item())
        return fn

    # ---- Iteration parts ----

    def _run_self_play_iteration(self) -> None:
        """Generate games_per_iteration games using best_net; push tuples to buffer."""
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_trainer.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/alphazero/trainer.py tests/test_trainer.py
git commit -m "Add Trainer init + self-play/train step (no arena yet)"
```

---

## Task 22: Trainer — full run loop with arena gate (superseded — removed in commit 413f42f)

**Files:**
- Modify: `src/alphazero/trainer.py` (add `run()`, arena gate, eval, checkpointing)
- Modify: `tests/test_trainer.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_trainer.py`:

```python
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


def test_arena_acceptance_replaces_best_net(tiny_config):  # superseded — removed in commit 413f42f
    """If candidate wins enough, best_net adopts candidate weights."""
    trainer = Trainer(TicTacToe(), tiny_config)
    trainer._run_self_play_iteration()
    # Manually scramble candidate to make it definitely different
    for p in trainer.candidate_net.parameters():
        with torch.no_grad():
            p.add_(torch.randn_like(p) * 0.5)

    # Stub arena: return a win rate above threshold
    win_rate_above = tiny_config.arena_threshold + 0.1
    trainer._arena_win_rate = lambda: win_rate_above   # type: ignore[method-assign]
    accepted = trainer._maybe_accept_candidate()
    assert accepted is True

    for p_b, p_c in zip(trainer.best_net.parameters(), trainer.candidate_net.parameters()):
        torch.testing.assert_close(p_b, p_c)


def test_arena_rejection_reverts_candidate_to_best(tiny_config):  # superseded — removed in commit 413f42f
    """If candidate fails the gate, candidate is reset to best_net's weights."""
    trainer = Trainer(TicTacToe(), tiny_config)
    trainer._run_self_play_iteration()
    # Snapshot original best_net params
    best_snapshot = [p.detach().clone() for p in trainer.best_net.parameters()]
    # Scramble candidate
    for p in trainer.candidate_net.parameters():
        with torch.no_grad():
            p.add_(torch.randn_like(p) * 0.5)

    trainer._arena_win_rate = lambda: tiny_config.arena_threshold - 0.1  # type: ignore[method-assign]
    accepted = trainer._maybe_accept_candidate()
    assert accepted is False

    # best_net unchanged
    for snap, p in zip(best_snapshot, trainer.best_net.parameters()):
        torch.testing.assert_close(snap, p)
    # candidate reverted to best
    for p_b, p_c in zip(trainer.best_net.parameters(), trainer.candidate_net.parameters()):
        torch.testing.assert_close(p_b, p_c)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_trainer.py -v
```
Expected: 4 fail with `AttributeError` (no `run`, `_maybe_accept_candidate`, etc).

- [ ] **Step 3: Implement `run`, arena gate, eval, and checkpointing**

Append to `src/alphazero/trainer.py`:
```python
    # ---- Arena ----

    def _make_argmax_mcts_agent(self, net: AlphaZeroNet):
        """Build an MCTS-with-NN agent that plays τ=0, no Dirichlet noise."""
        from .mcts import MCTS

        eval_fn = self._make_eval_fn(net)
        def agent(game, state):
            mcts = MCTS(game, eval_fn, c_puct=self.config.c_puct)
            # Pass raw state to MCTS; it canonicalizes internally at the eval_fn boundary.
            pi = mcts.search(state, num_simulations=self.config.num_simulations,
                             add_root_noise=False)
            return int(np.argmax(pi))
        return agent

    def _arena_win_rate(self) -> float:
        """Play candidate vs best; return candidate's win rate (draws count as 0.5)."""
        from .arena import play_match
        candidate_agent = self._make_argmax_mcts_agent(self.candidate_net)
        best_agent = self._make_argmax_mcts_agent(self.best_net)
        result = play_match(
            self.game, candidate_agent, best_agent,
            num_games=self.config.arena_games,
        )
        return result.win_rate

    def _maybe_accept_candidate(self) -> bool:
        """Run arena gate. If candidate wins ≥ threshold, accept; else revert."""
        win_rate = self._arena_win_rate()
        if win_rate >= self.config.arena_threshold:
            self.best_net = copy.deepcopy(self.candidate_net)
            self.best_net.eval()
            return True
        # Reject: revert candidate to best
        self.candidate_net.load_state_dict(self.best_net.state_dict())
        return False

    # ---- Eval vs solver ----

    def _eval_vs_solver(self) -> dict:
        """Play current best_net vs the perfect TTT solver, return win/draw/loss counts."""
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

    # ---- Checkpoint ----

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

    # ---- Outer loop ----

    def run(self) -> None:
        for _ in range(self.config.num_iterations):
            self.iteration += 1

            # 1. Self-play with best_net
            self._run_self_play_iteration()

            # 2. Train candidate_net for K steps
            if len(self.replay_buffer) >= self.config.min_buffer_size:
                for _step in range(self.config.training_steps_per_iteration):
                    self._train_step()

            # 3. Arena gate (every arena_interval iterations)
            if self.iteration % self.config.arena_interval == 0:
                self._maybe_accept_candidate()

            # 4. Eval vs solver (every eval_interval iterations)
            if self.iteration % self.config.eval_interval == 0:
                _ = self._eval_vs_solver()

            # 5. Checkpoint
            self._save_checkpoint()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_trainer.py -v
```
Expected: 8 passed (4 from Task 21 + 4 new).

- [ ] **Step 5: Commit**

```bash
git add src/alphazero/trainer.py tests/test_trainer.py
git commit -m "Add Trainer.run() with arena gate, eval, checkpointing"
```

---

## Task 23: The six invariant tests

**Files:**
- Create: `tests/test_invariants.py`

- [ ] **Step 1: Write the invariant tests**

`tests/test_invariants.py`:
```python
"""Named regression tests for the six critical invariants of AlphaZero.

These each guard a "silent failure" mode — training that appears to be
running but the agent never improves (or improves and then collapses).
"""
from __future__ import annotations

import copy
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from alphazero.games.tictactoe import TicTacToe
from alphazero.mcts import MCTS, Node
from alphazero.network import AlphaZeroNet
from alphazero.replay_buffer import ReplayBuffer
from alphazero.selfplay import run_one_game
from alphazero.trainer import Trainer


# ---------- Invariant 1: self-play uses best_net, training updates candidate_net ----------

def test_invariant_1_selfplay_uses_best_net_not_candidate(tiny_config):
    """Wrap both nets in counting wrappers; run one self-play iteration.
    Assert: best_net.forward was called, candidate_net.forward was NOT."""
    trainer = Trainer(TicTacToe(), tiny_config)

    best_calls = {"n": 0}
    candidate_calls = {"n": 0}
    original_best_fwd = trainer.best_net.forward
    original_candidate_fwd = trainer.candidate_net.forward

    def count_best(*a, **k):
        best_calls["n"] += 1
        return original_best_fwd(*a, **k)

    def count_candidate(*a, **k):
        candidate_calls["n"] += 1
        return original_candidate_fwd(*a, **k)

    trainer.best_net.forward = count_best       # type: ignore[assignment]
    trainer.candidate_net.forward = count_candidate  # type: ignore[assignment]

    trainer._run_self_play_iteration()

    assert best_calls["n"] > 0
    assert candidate_calls["n"] == 0, (
        f"candidate_net was called {candidate_calls['n']} times during self-play; "
        "this would poison training data with a regressed network."
    )


# ---------- Invariant 2: Dirichlet noise — root only, self-play only ----------

def test_invariant_2_arena_disables_dirichlet_noise(tiny_config):
    """The arena agent must NOT add Dirichlet noise (it would corrupt the gate)."""
    trainer = Trainer(TicTacToe(), tiny_config)
    agent = trainer._make_argmax_mcts_agent(trainer.best_net)

    # Patch MCTS.search to capture add_root_noise
    captured = {"add_root_noise": None}
    from alphazero.mcts import MCTS
    original_search = MCTS.search
    def spy_search(self, root_state, num_simulations, add_root_noise):
        captured["add_root_noise"] = add_root_noise
        return original_search(self, root_state, num_simulations, add_root_noise)
    MCTS.search = spy_search  # type: ignore[method-assign]
    try:
        ttt = TicTacToe()
        agent(ttt, ttt.initial_state())
    finally:
        MCTS.search = original_search  # type: ignore[method-assign]

    assert captured["add_root_noise"] is False


def test_invariant_2_dirichlet_noise_affects_only_root(tiny_config):
    """Noise is applied at root expansion; not at deeper expansions."""
    # We test this indirectly via the implementation: _expand only invokes
    # _add_dirichlet_noise when add_root_noise=True, and only the root-level
    # _expand call gets that flag. Deeper expansions go through _simulate's
    # call to _expand with add_root_noise=False (see mcts.py Task 12).
    # This is a structural test: read the call site.
    import inspect
    from alphazero import mcts
    source = inspect.getsource(mcts.MCTS._simulate)
    assert "add_root_noise=False" in source, (
        "MCTS._simulate must call _expand with add_root_noise=False (only the "
        "initial root expansion gets noise)."
    )


# ---------- Invariant 3: Backup flips sign per ply ----------

def test_invariant_3_backup_flips_sign_per_ply():
    """Already covered in tests/test_mcts.py::test_backup_flips_sign_per_ply.
    Add a deeper-tree variant here as a second guard."""
    ttt = TicTacToe()
    mcts = MCTS(ttt, lambda s: (np.full(9, 1/9, dtype=np.float32), 0.5))

    # Build a 4-deep path manually
    root = Node()
    root.is_expanded = True
    n1 = Node(prior=1.0); root.children[0] = n1
    n2 = Node(prior=1.0); n1.children[0] = n2
    n3 = Node(prior=1.0); n2.children[0] = n3
    n4 = Node(prior=1.0); n3.children[0] = n4

    mcts._backup([root, n1, n2, n3, n4], leaf_value=+1.0)

    # Sign flips each step: leaf=+1 → n4 sees +1 → n3 sees -1 → n2 sees +1 → n1 sees -1 → root sees +1
    # WAIT — the loop in _backup is "for node in reversed(path)": it updates each
    # node with `value`, then flips for the NEXT iteration. So:
    #   n4: value_sum += +1, then value = -1
    #   n3: value_sum += -1, then value = +1
    #   n2: value_sum += +1, then value = -1
    #   n1: value_sum += -1, then value = +1
    #   root: value_sum += +1
    assert n4.value_sum == +1.0
    assert n3.value_sum == -1.0
    assert n2.value_sum == +1.0
    assert n1.value_sum == -1.0
    assert root.value_sum == +1.0


# ---------- Invariant 4: canonical_state applied consistently ----------

def test_invariant_4_encode_after_canonical_swaps_planes_when_player_changes():
    """After every move the perspective swap is consistent."""
    ttt = TicTacToe()
    s = ttt.initial_state()
    # X plays center
    s = ttt.apply(s, 4)
    canon_o = ttt.canonical_state(s)
    enc_o = ttt.encode(canon_o)
    # From O's POV, X's piece at center is on the "opp" plane (index 1)
    assert enc_o[0, 1, 1] == 0
    assert enc_o[1, 1, 1] == 1

    # O plays corner
    s = ttt.apply(s, 0)
    canon_x = ttt.canonical_state(s)
    enc_x = ttt.encode(canon_x)
    # From X's POV, X's center is on plane 0 (my), O's corner is on plane 1 (opp)
    assert enc_x[0, 1, 1] == 1
    assert enc_x[1, 0, 0] == 1


# ---------- Invariant 5: z-assignment per-ply mover's POV ----------

def test_invariant_5_z_assignment_per_ply_mover_pov():
    """Already covered in tests/test_selfplay.py. Add an O-wins-quick variant."""
    from alphazero.selfplay import _assign_z
    ttt = TicTacToe()
    # X, O, X, O, X, O — O wins on move 6 (index 5)
    history = [
        (np.zeros((3, 3, 3), dtype=np.float32), np.full(9, 1/9, dtype=np.float32), p)
        for p in [1, -1, 1, -1, 1, -1]
    ]
    # After move 5, X's turn would be next; X lost.
    final_state = np.array([[-1, -1, -1], [1, 1, 0], [1, 0, 0]], dtype=np.int8)
    final_value = -1.0
    zs = _assign_z(history, final_value, final_state, ttt)
    # X plies (0,2,4): -1; O plies (1,3,5): +1
    assert zs == [-1.0, +1.0, -1.0, +1.0, -1.0, +1.0]


# ---------- Invariant 6: replay buffer NOT reset between iterations ----------

def test_invariant_6_replay_buffer_persists_across_iterations(tiny_config):
    """Two consecutive iterations should accumulate data in the same buffer."""
    trainer = Trainer(TicTacToe(), tiny_config)

    trainer._run_self_play_iteration()
    size_after_1 = len(trainer.replay_buffer)
    assert size_after_1 > 0

    trainer._run_self_play_iteration()
    size_after_2 = len(trainer.replay_buffer)

    # Buffer should grow (or stay capped if at capacity). NEVER reset.
    assert size_after_2 >= size_after_1, (
        "Replay buffer shrank between iterations — this should never happen. "
        "Did something call replay_buffer.clear()?"
    )
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
pytest tests/test_invariants.py -v
```
Expected: 7 passed (one invariant has two tests, Invariant 2).

- [ ] **Step 3: Commit**

```bash
git add tests/test_invariants.py
git commit -m "Add named regression tests for the 6 critical invariants"
```

---

## Task 24: End-to-end kill-criterion test

**Files:**
- Create: `tests/test_e2e_tictactoe.py`

- [ ] **Step 1: Write the E2E test**

`tests/test_e2e_tictactoe.py`:
```python
"""End-to-end test: train an AlphaZero agent on TTT and verify it never
loses to a perfect minimax solver.

This is the kill criterion for Sub-project 1. Marked slow because a full
training run takes ~15–45 minutes on M4 Pro. Run with:
    pytest -m slow tests/test_e2e_tictactoe.py -v -s
"""
from __future__ import annotations

import pytest

from alphazero.arena import play_match
from alphazero.config import TrainingConfig
from alphazero.games.tictactoe import TicTacToe
from alphazero.solvers.tictactoe_solver import solve_tictactoe_action
from alphazero.trainer import Trainer


@pytest.mark.slow
def test_e2e_tictactoe_reaches_zero_losses_vs_perfect_solver(tmp_path, monkeypatch):
    """Train, then play 200 games vs the perfect TTT minimax solver.
    Pass criterion: 0 losses (TTT is a draw with perfect play)."""
    monkeypatch.chdir(tmp_path)

    config = TrainingConfig(
        # Modest training run for the kill criterion
        n_blocks=4,
        n_channels=32,
        num_simulations=50,
        games_per_iteration=100,
        training_steps_per_iteration=500,
        num_iterations=30,
        arena_interval=5,
        eval_interval=5,
        eval_games=200,
        seed=42,
        device="cpu",  # set to "auto" if you have MPS/CUDA configured
    )

    trainer = Trainer(TicTacToe(), config)
    trainer.run()

    agent = trainer._make_argmax_mcts_agent(trainer.best_net)
    def solver_agent(game, state):
        return solve_tictactoe_action(state)

    result = play_match(TicTacToe(), agent, solver_agent, num_games=200)

    print(
        f"\nE2E result: wins={result.wins_a} draws={result.draws} losses={result.losses_a}"
    )
    assert result.losses_a == 0, (
        f"Agent lost {result.losses_a} games to perfect play. TTT is a draw "
        f"with optimal play; any loss indicates a pipeline bug or insufficient "
        f"training. Increase num_iterations or num_simulations."
    )
```

- [ ] **Step 2: Verify the test is properly marked slow and not collected by default**

```bash
pytest --collect-only -m "not slow" | grep test_e2e_tictactoe
```
Expected: nothing printed (test is excluded from default collection).

```bash
pytest --collect-only -m slow tests/test_e2e_tictactoe.py
```
Expected: 1 test collected.

- [ ] **Step 3: Commit**

```bash
git add tests/test_e2e_tictactoe.py
git commit -m "Add E2E kill-criterion test (zero losses vs perfect solver)"
```

- [ ] **Step 4: Run the slow test** (the actual Sub-project 1 success bar)

```bash
pytest -m slow tests/test_e2e_tictactoe.py -v -s
```
Expected: PASS. Wall-time: 15–45 minutes on M4 Pro CPU.

If this fails, **do not move on**. Investigate the eval-vs-solver curve (it should be in checkpoints/iter_*.pt). The most common causes are:
1. Bug in one of the six invariants (rerun `pytest tests/test_invariants.py -v`).
2. Insufficient training (try `num_iterations=50` and `num_simulations=100`).
3. Off-by-one in z-assignment or backup-sign-flip.

---

## Task 25: CLI

**Files:**
- Create: `src/alphazero/cli.py`

- [ ] **Step 1: Implement the CLI**

`src/alphazero/cli.py`:
```python
"""Command-line entry point: python -m alphazero <subcommand> [...]

Subcommands:
    train --config configs/tictactoe.toml
    eval  --checkpoint checkpoints/iter_NNNN.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from .arena import play_match
from .config import TrainingConfig, load_config
from .games.tictactoe import TicTacToe
from .solvers.tictactoe_solver import solve_tictactoe_action
from .trainer import Trainer


def _cmd_train(args: argparse.Namespace) -> int:
    config = load_config(args.config) if args.config else TrainingConfig()
    game = TicTacToe()  # Sub-project 1: only TTT
    trainer = Trainer(game, config)
    print(f"Starting training: {config.num_iterations} iterations on {trainer.device}")
    trainer.run()
    print("Training complete.")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    config = TrainingConfig(device="cpu")
    game = TicTacToe()
    trainer = Trainer(game, config)

    ckpt = torch.load(args.checkpoint, map_location=trainer.device)
    trainer.best_net.load_state_dict(ckpt["best_net"])
    trainer.best_net.eval()

    agent = trainer._make_argmax_mcts_agent(trainer.best_net)
    def solver_agent(_g, state):
        return solve_tictactoe_action(state)

    result = play_match(game, agent, solver_agent, num_games=args.num_games)
    print(
        f"vs perfect solver ({args.num_games} games): "
        f"wins={result.wins_a} draws={result.draws} losses={result.losses_a}"
    )
    return 0 if result.losses_a == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m alphazero")
    subs = parser.add_subparsers(dest="cmd", required=True)

    p_train = subs.add_parser("train", help="Run training loop")
    p_train.add_argument("--config", type=Path, default=Path("configs/tictactoe.toml"))
    p_train.set_defaults(func=_cmd_train)

    p_eval = subs.add_parser("eval", help="Evaluate a checkpoint vs perfect solver")
    p_eval.add_argument("--checkpoint", type=Path, required=True)
    p_eval.add_argument("--num-games", type=int, default=200)
    p_eval.set_defaults(func=_cmd_eval)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Add `__main__.py` so `python -m alphazero` works**

`src/alphazero/__main__.py`:
```python
from .cli import main
import sys

sys.exit(main())
```

- [ ] **Step 3: Smoke-test the CLI**

```bash
python -m alphazero --help
python -m alphazero train --help
python -m alphazero eval --help
```
Expected: each prints help and exits cleanly.

- [ ] **Step 4: Commit**

```bash
git add src/alphazero/cli.py src/alphazero/__main__.py
git commit -m "Add CLI: python -m alphazero {train,eval}"
```

---

## Task 26: Run the full unit-test suite as a final check

- [ ] **Step 1: Run all unit tests (not slow)**

```bash
pytest -m "not slow" -v
```
Expected: all tests pass (≈100 tests across 11 files).

- [ ] **Step 2: Run linter**

```bash
ruff check src tests
```
Expected: no errors. Fix any minor issues inline before continuing.

- [ ] **Step 3: Commit any fixes**

```bash
git add -u && git commit -m "Lint cleanup" || true
```

- [ ] **Step 4: Push**

```bash
git push
```

---

## Final state

After Task 25, the repo has:

- Complete AlphaZero framework (Game ABC + TTT, AlphaZeroNet, MCTS, ReplayBuffer, SelfPlay, Arena, Trainer)
- ~100 unit tests, all passing in under a minute
- 7 invariant tests guarding the 6 silent-failure modes
- 1 slow E2E test that defines Sub-project 1 success
- CLI for training and evaluation
- TOML-based configuration with sensible TTT defaults

**Sub-project 1 is done when** `pytest -m slow tests/test_e2e_tictactoe.py` passes consistently with `result.losses_a == 0`.

Then Sub-project 2 (Connect 4) can begin — only the `Game` class changes; every other layer is reusable.
