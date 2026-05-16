---
title: Connect 4 on the AlphaZero Framework (Sub-project 2)
date: 2026-05-15
status: Approved (brainstorming → implementation)
sub_project: 2
target: Validate the framework on Connect 4 — beat minimax-depth-8 in ≥80% of 200 games
---

# Connect 4 on the AlphaZero Framework (Sub-project 2)

## 1. Context

Sub-project 1 built and validated the game-agnostic AlphaZero framework on Tic-Tac-Toe. Sub-project 2's job is to demonstrate that the framework **scales** to a meaningfully harder game **without changing any framework code** — only a new `Game` subclass and a new evaluation opponent.

Connect 4 is the ideal stepping stone between TTT and chess:

| Property | TTT | Connect 4 | Chess |
|---|---|---|---|
| Board | 3×3 = 9 cells | 6×7 = 42 cells | 8×8 = 64 cells |
| Action space | 9 | 7 | ~4672 |
| Reachable states | ~5,478 | ~10¹³ | ~10⁴³ |
| Max game length | 9 plies | 42 plies | unbounded (we cap) |
| Branching factor | ~9 → 1 | ~7 throughout | ~35 avg |
| Solved? | Yes (draw) | Yes (1st player wins) | No |

Connect 4 is 10⁹× larger than TTT in state space but still tractable for from-scratch self-play training on a consumer machine. If the framework works on Connect 4, it'll work on chess (modulo compute scale).

## 2. Goals

- Implement `Connect4(Game)` as a new file in `src/alphazero/games/`.
- Implement a from-scratch minimax-depth-8 opponent with α-β pruning and a center-bias heuristic, as the eval ground truth.
- Reuse the entire AlphaZero framework from Sub-project 1 **unchanged** (Game ABC, AlphaZeroNet, MCTS, SelfPlayWorker, ReplayBuffer, Arena, Trainer).
- Reach the kill criterion: trained agent **beats the minimax-depth-8 opponent in ≥80% of 200 games** (alternating colors).

## 3. Non-goals

- Matching perfect play (deferred to Sub-project 4 if pursued at all).
- Distillation from existing Connect 4 solvers (e.g., Pascal Pons's C++ solver).
- Parallel self-play / multi-process training (deferred to Sub-project 3 for chess).
- Hyperparameter sweeps. We pick reasonable starting values, tune only if convergence fails.
- New framework features. If a framework limitation surfaces, we document it and revisit in Sub-project 3.

---

## 4. What's new vs what's reused

The architecture is unchanged. Only three new source files and three new tests.

| Component | Status | What changes |
|---|---|---|
| `Game` ABC | reuse as-is | nothing |
| `TicTacToe` | leave in place | nothing |
| **`Connect4`** | new | rules, encoding, terminal detection, symmetries |
| `AlphaZeroNet` | reuse as-is | only `input_shape` and `action_size` differ (parameterized by Game) |
| `MCTS` | reuse as-is | nothing |
| `SelfPlayWorker` | reuse as-is | nothing |
| `ReplayBuffer` | reuse as-is | nothing |
| `Arena` | reuse as-is | nothing |
| **`Connect4MinimaxOpponent`** | new | α-β pruning at depth 8 + heuristic eval |
| `Trainer` | small extension | replace TTT-specific `_eval_vs_solver` with a generic `_eval_vs_opponent(opponent)` callable injection |
| `cli.py` | small generalization | `train`/`eval`/`play` dispatch on `--game {tictactoe, connect4}` |

This is the payoff of the layered architecture in Sub-project 1.

---

## 5. Components

### 5.1 `Connect4` (new Game subclass)

**State representation**: `np.ndarray` of shape `(6, 7)`, dtype `int8`, values `+1 / −1 / 0`.

- `+1` = first player (red); `−1` = second player (yellow); `0` = empty.
- Row 0 is the top of the board; pieces fall to the lowest empty cell of their column.
- Player to move is determined by piece-count parity (consistent with the TTT convention).

**Action space**: integers `0..6` corresponding to the 7 columns.

**Interface**:

```python
class Connect4(Game):
    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (3, 6, 7)              # (channels, H, W) — my, opp, ones planes

    @property
    def action_size(self) -> int:
        return 7

    def initial_state(self) -> State:
        return np.zeros((6, 7), dtype=np.int8)

    def current_player(self, state: State) -> int:
        moves = int(np.count_nonzero(state))
        return 1 if moves % 2 == 0 else -1

    def legal_actions_mask(self, state: State) -> np.ndarray:
        # A column is legal iff its top cell (row 0) is empty.
        return state[0, :] == 0

    def apply(self, state: State, action: int) -> State:
        # Drop piece into lowest empty cell of column `action`.
        # Raise ValueError if column is full.
        ...

    def terminal_value(self, state: State) -> float | None:
        # Check 4-in-a-row in 4 directions: horizontal, vertical, both diagonals.
        # Return -1.0 (opponent of current player just won), 0.0 (draw), or None (in progress).
        ...

    def encode(self, state: State) -> np.ndarray:
        # 3 planes: my pieces, opp pieces, ones. Assumes canonical state.
        ...

    def canonical_state(self, state: State) -> State:
        return (state * self.current_player(state)).astype(np.int8)

    def symmetries(
        self, encoded: np.ndarray, policy: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        # Identity + horizontal mirror. The mirror flips columns left↔right
        # in both the encoded board planes and the policy.
        identity = (encoded, policy)
        mirrored = (
            np.flip(encoded, axis=2).copy(),
            np.flip(policy, axis=0).copy(),
        )
        return [identity, mirrored]
```

**Key decisions**:

- **Row 0 is the top** (gravity pulls pieces toward row 5). This is the natural visual convention; tests will assert this orientation.
- **Drop mechanics in `apply`**: iterate from row 5 upward, place piece at the first empty cell.
- **4-in-a-row detection** uses sliding windows of length 4 across all rows, columns, and both diagonals. ~24 row windows, 28 column windows, 24 diagonal windows = 76 total checks. Cheap.
- **Symmetries**: only 2 (identity + horizontal mirror). Unlike TTT's 8, Connect 4 has no vertical symmetry because gravity is asymmetric (pieces fall down, not up). Data augmentation is 2× per ply instead of 8×.

### 5.2 `Connect4MinimaxOpponent` (new agent)

Standard α-β minimax with a position-evaluation heuristic for non-terminal leaves.

**Interface**:

```python
class Connect4MinimaxOpponent:
    """Callable agent: takes (game, state) → action. Uses α-β minimax to depth N."""

    def __init__(self, depth: int = 8):
        self.depth = depth

    def __call__(self, game: Game, state: State) -> int:
        # Search with α-β pruning to self.depth; return best action.
        ...

    def _evaluate(self, state: State) -> float:
        """Static evaluation of a non-terminal position from current player's POV.

        Score = sum over 4-cell windows of:
            +∞  if window contains 4 of mine
            -∞  if window contains 4 of opponent
            +w[k]  if window contains k of mine + (4-k) empties (no opp)
            -w[k]  if window contains k of opp + (4-k) empties (no mine)
            0   otherwise (window has both)

        Plus a small center-bias bonus: each piece in the center column adds
        a constant for the owning side.
        """
        ...
```

**Heuristic weights** (starting values, may tune for opponent strength):

| Window content | Weight |
|---|---|
| 3 of mine + 1 empty | +5 |
| 2 of mine + 2 empties | +2 |
| Center column piece | +3 each |
| Symmetric for opponent | mirrored negative |

**Why this works**: At depth 8 with α-β pruning over a 7-action branching factor, the search visits roughly 7⁴-7⁵ ≈ 16K-100K nodes per move (after pruning). Pure Python with numpy primitives evaluates ~1-2ms per leaf, giving ~10-200ms per move — fast enough for 200-game evaluation matches that finish in minutes.

**Test target**: Depth-8 should beat depth-2 in ≥90% of games (confirms the search is correctly exploring).

### 5.3 Generalizing the Trainer's eval hook

Current Sub-project 1 code has a hardcoded `_eval_vs_solver` that imports `solve_tictactoe_action`. To support Connect 4 cleanly, we generalize to:

```python
def _eval_vs_opponent(self, opponent: Agent, num_games: int) -> dict:
    """Play current best_net vs an arbitrary opponent. Game-agnostic."""
    from .arena import play_match
    net_agent = self._make_argmax_mcts_agent(self.best_net)
    result = play_match(self.game, net_agent, opponent, num_games=num_games)
    return {"wins": result.wins_a, "draws": result.draws, "losses": result.losses_a}
```

`Trainer.run()` then takes an optional `eval_opponent: Agent | None` parameter. For TTT, the existing `solver_agent` wrapper is passed; for Connect 4, a `Connect4MinimaxOpponent(depth=8)` instance. The old `_eval_vs_solver` becomes a thin wrapper around `_eval_vs_opponent` to preserve backward compatibility with the Sub-project 1 E2E test.

### 5.4 Minor CLI generalization

`cli.py` currently hardcodes `game = TicTacToe()`. Extend to:

```python
GAMES = {"tictactoe": TicTacToe, "connect4": Connect4}

# In each subcommand:
game_cls = GAMES[args.game]
game = game_cls()
```

Add `--game {tictactoe,connect4}` to `train`, `eval`, `play` subcommands; default `tictactoe` to preserve Sub-project 1 behavior. For `eval --game connect4`, the opponent is `Connect4MinimaxOpponent(depth=args.opponent_depth or 8)`.

---

## 6. Hyperparameters (starting config)

`configs/connect4.toml`:

```toml
# Connect 4 training config — starting values from the AlphaZero paper
# scaled down for a learning project on M4 Pro.

# Network
n_blocks = 6                       # up from TTT's 4 — deeper for 6×7 board
n_channels = 64                    # up from TTT's 32
                                   # ≈ 600K params (vs TTT's 76K)

# MCTS
num_simulations = 100              # up from TTT's 50
c_puct = 2.5                       # up from TTT's 1.5 — more exploration with 7-way branching
dirichlet_alpha = 0.5              # down from TTT's 1.0 — concentrated noise on fewer actions
dirichlet_weight = 0.25

# Self-play
games_per_iteration = 100
temperature_threshold = 15         # up from TTT's 6 — Connect 4 games ~20-30 plies

# Training
training_steps_per_iteration = 1000  # up from TTT's 500
batch_size = 64
learning_rate = 1e-3
weight_decay = 1e-4

# Replay buffer
replay_buffer_capacity = 200_000   # up from 50K — longer games, more iterations
min_buffer_size = 20_000

# Arena
arena_interval = 5
arena_games = 40
arena_threshold = 0.55

# Eval (vs minimax-depth-8)
eval_interval = 5
eval_games = 200

# Schedule
num_iterations = 50

# Misc
seed = 42
device = "auto"
checkpoint_dir = "checkpoints"
log_dir = "runs"
```

**Wall-time estimates on M4 Pro**:

| Setup | Per iter | 50 iter total |
|---|---|---|
| CPU | ~15-25 min | 12-20 hours |
| MPS | ~5-10 min | 3-6 hours |

If too slow, the first knob to drop is `games_per_iteration` (50 instead of 100) — halves training time at modest quality cost.

**The handful of decisions that actually matter:**

- `num_simulations=100`: most impactful. Lower it and self-play data quality drops fast for Connect 4.
- `c_puct=2.5`: small branching factor (7) but long-horizon planning needs broad exploration early. The AlphaZero paper used 2.5-4.0 for chess; 2.5 is a reasonable starting point.
- `temperature_threshold=15`: covers the opening phase. Beyond ~15 plies, deterministic argmax accelerates convergence to optimal play.
- `dirichlet_alpha=0.5`: smaller α = more concentrated noise. With only 7 actions, we want noise to genuinely change the action distribution; α=1.0 (TTT's value) would be nearly uniform.

---

## 7. Testing strategy

### 7.1 `Connect4` game tests (`tests/test_game_connect4.py`)

About 20 tests, mirroring `tests/test_game_tictactoe.py`:

- shapes: `input_shape == (3, 6, 7)`, `action_size == 7`
- `initial_state` is empty (3, 6, 7)
- `current_player` starts at +1, alternates by parity
- `apply` drops piece in lowest empty cell of column
- `apply` does not mutate input state
- `apply` raises `ValueError` on full column
- `legal_actions_mask` reflects column-top availability
- `terminal_value`: horizontal 4-in-a-row, vertical, both diagonals (↗ and ↘), draw, in-progress
- `terminal_value` returns from current-player POV
- `encode` shape and dtype, plane assignments
- `canonical_state` identity for player +1, inverts for player -1
- `symmetries` returns 2 tuples, identity first, mirror correct

### 7.2 `Connect4MinimaxOpponent` tests (`tests/test_connect4_minimax.py`)

- only plays legal moves
- takes immediate winning move (depth ≥ 1)
- blocks immediate threat (depth ≥ 2)
- prefers center column on empty board (heuristic check)
- finds 2-move tactic at depth ≥ 4
- depth-8 beats depth-2 in ≥9 of 10 games (sanity check on search depth)

### 7.3 Kill-criterion E2E (`tests/test_e2e_connect4.py`, `@pytest.mark.slow`)

```python
@pytest.mark.slow
def test_e2e_connect4_beats_minimax_depth_8(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = load_config(REPO_ROOT / "configs/connect4.toml")
    trainer = Trainer(Connect4(), config)
    trainer.run(eval_opponent=Connect4MinimaxOpponent(depth=8))

    agent = trainer._make_argmax_mcts_agent(trainer.best_net)
    opponent = Connect4MinimaxOpponent(depth=8)
    result = play_match(Connect4(), agent, opponent, num_games=200)

    print(f"\nFinal: wins={result.wins_a} draws={result.draws} losses={result.losses_a} "
          f"win_rate={result.win_rate:.2%}")
    assert result.win_rate >= 0.80, (
        f"Agent win rate {result.win_rate:.2%} below 0.80 target. "
        "Increase num_iterations or num_simulations."
    )
```

### 7.4 Invariant tests

The existing `tests/test_invariants.py` already guards the 6 framework invariants. No duplication needed — those tests operate on TicTacToe via fixtures, but they prove the framework's invariants hold, which is what matters. If we want, we can later parameterize them across (TTT, C4) for extra safety.

---

## 8. Success criterion (recap)

**Sub-project 2 is done when:**

> `pytest -m slow tests/test_e2e_connect4.py` passes consistently with `result.win_rate >= 0.80` against the depth-8 minimax opponent over 200 games (alternating colors).

After this passes, Sub-project 3 (chess) can begin.

---

## 9. Project layout additions

```
configs/connect4.toml                              ← new
src/alphazero/games/connect4.py                    ← new (~200 lines)
src/alphazero/opponents/__init__.py                ← new package
src/alphazero/opponents/connect4_minimax.py       ← new (~200 lines)
src/alphazero/trainer.py                           ← small change: generalize _eval_vs_*
src/alphazero/cli.py                               ← small change: --game flag
tests/test_game_connect4.py                        ← new (~250 lines)
tests/test_connect4_minimax.py                     ← new (~150 lines)
tests/test_e2e_connect4.py                         ← new (~50 lines)
```

The new `opponents/` package is parallel to the existing `solvers/`. `solvers/` is for ground-truth oracles (perfect TTT minimax); `opponents/` is for *classes of agents* used as eval baselines. Same conceptual layer; different semantic role.

---

## 10. Out of scope (saved for future sub-projects)

- **Perfect-play Connect 4 evaluation** — Sub-project 4 if pursued. Requires either a precomputed lookup table or a strong solver like Pascal Pons's C++ engine.
- **Parallel self-play** — Sub-project 3 will need it for chess; revisit then.
- **Connect 4 opening book** — not necessary for the win-rate target.
- **MCTS optimizations** (batched eval, virtual loss for parallel rollouts) — Sub-project 3.
- **Bitboard representation** — Sub-project 3 if chess move-gen becomes the bottleneck.

---

## 11. References

- Allis, V. *A Knowledge-based Approach of Connect-Four* (1988) — original Connect 4 solver thesis; established that first player wins with perfect play.
- Pascal Pons, [Connect 4 solver](http://blog.gamesolver.org/) — modern strong solver; reference if we ever pursue perfect-play matching.
- Silver et al., *Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm* (AlphaZero), 2017 — primary architecture reference (unchanged from Sub-project 1).
- Sub-project 1 spec: [`2026-05-15-alphazero-framework-tictactoe-design.md`](2026-05-15-alphazero-framework-tictactoe-design.md) — defines the framework Sub-project 2 reuses.
