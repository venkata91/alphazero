---
title: Chess on the AlphaZero Framework (Sub-project 3)
date: 2026-05-16
status: Approved (brainstorming → implementation)
sub_project: 3
target: Trained chess agent achieves ≥50% win rate against Stockfish 1500 ELO over 100 games
---

# Chess on the AlphaZero Framework (Sub-project 3)

## 1. Context

This is the third sub-project in the AlphaZero learning project. Sub-projects 1 (Tic-Tac-Toe) and 2 (Connect 4) validated the **framework**: the game-agnostic AlphaZero pipeline reaches perfect TTT play and ≥80% win rate vs Connect 4 minimax-depth-8. Sub-project 3 takes the same framework and applies it to **chess** — a game ~10⁹× larger than Connect 4 in state-space — to prove the framework scales beyond toy games.

The user's primary goal remains **learning**: build the full AlphaZero pipeline from scratch, understand each component, let the achieved ELO emerge from the work. The original five-sub-project decomposition:

| # | Scope | Target | Status |
|---|---|---|---|
| 1 | Framework + Tic-Tac-Toe | 0 losses vs perfect solver | ✅ Done |
| 2 | Connect 4 plugged into framework | ≥80% vs minimax-depth-8 | ✅ Implemented (training pending) |
| **3** | **Chess + parallel self-play** | **≥50% vs Stockfish 1500 ELO** | **This spec** |
| 4 | Productionize: UCI engine, scale to 1800-2300 ELO | ≥50% vs Stockfish 2000 ELO | Future |
| 5 | Hybrid: supervised pre-training + distillation toward 2500+ ELO | ≥50% vs Stockfish 2500 ELO | Future |

Sub-project 3 explicitly does NOT chase grandmaster-level play. The target is a clearly-competent agent (1500 ELO is intermediate club level) trained from random init via pure self-play. The infrastructure built here — **parallel self-play with batched NN inference** — is the load-bearing thing.

## 2. Goals

- **Implement `Chess(Game)`** wrapping `python-chess`, with the AlphaZero 8×8×73 move encoding and a ~20-plane input representation.
- **Build a parallel self-play module** with N CPU worker processes + a central NN-server that batches inference requests. This is required infrastructure for chess training and a deliverable in its own right.
- **Integrate Stockfish** as the evaluation opponent via UCI, configured to a specific ELO (1500 initially) for an unambiguous strength measurement.
- **Reach the kill criterion**: trained agent wins ≥50% of 100 games (alternating colors) against `StockfishOpponent(elo=1500, time_per_move=0.5)`.
- **Architect for scale-invariance**: the codebase must support pushing toward 2000+ ELO via configuration changes alone (network size, iteration count, simulation depth) — no refactoring.

## 3. Non-goals

- **Matching paper-AlphaZero strength** (~3500 ELO with 19×256 ResNet, 800 sims, 700K self-play games on 5000+ TPUs). Out of reach for a learning project.
- **History planes** in the input encoding. Paper uses 8 previous positions × 14 planes = 112 history planes; we use 0. Threefold repetition is handled at terminal detection (via `python-chess`'s `is_repetition`). Adding history is a Sub-project 4 enhancement.
- **UCI engine protocol** for playing on Lichess / in chess GUIs. Sub-project 4.
- **Distillation from existing solvers** or supervised pre-training on master games. Sub-project 5.
- **Hyperparameter sweeps**. Pick reasonable starting values; tune only if convergence fails.

---

## 4. What's new vs what's reused

The framework is unchanged. We add three new components and extend two existing ones.

| Component | Status | What changes |
|---|---|---|
| `Game` ABC | reuse | nothing |
| `TicTacToe`, `Connect4` | leave in place | nothing |
| **`Chess`** | new | Wraps `python-chess`, ~20 input planes, 4672 action space |
| **Move encoding** | new | AlphaZero 8×8×73 ↔ `python-chess` Move conversion (bijective) |
| `AlphaZeroNet` | reuse | only the architectural knobs change (`n_blocks=10, n_channels=128`) |
| `MCTS` | reuse | nothing |
| `SelfPlayWorker` (`run_one_game`) | reuse for serial path | nothing |
| **`parallel_selfplay`** | new | Process pool of N workers + central NN-server with batched inference |
| `ReplayBuffer` | reuse | capacity grows to 500K-1M (chess games are longer) |
| `Arena` (`play_match`) | reuse | nothing |
| `Trainer` | small extend | When `num_workers > 1`, dispatch parallel self-play instead of serial loop |
| **`StockfishOpponent`** | new | UCI wrapper with ELO configuration |
| `cli.py` | small extend | `--game chess` routing |
| `Dockerfile` / `bin/setup.sh` | new | Reproducible env across Colab/Mac/Lambda |

This is the architectural payoff: even at chess scale, ~80% of the framework is reused unchanged.

---

## 5. Components

### 5.1 `Chess(Game)` — wraps `python-chess`

**Role**: encapsulate everything chess-specific. State is a `chess.Board` object (mutable; we copy + push to advance).

```python
class Chess(Game):
    input_shape  = (20, 8, 8)
    action_size  = 4672

    def initial_state(self) -> chess.Board:
        return chess.Board()

    def current_player(self, state) -> int:
        return 1 if state.turn == chess.WHITE else -1

    def legal_actions_mask(self, state) -> np.ndarray:
        mask = np.zeros(4672, dtype=bool)
        for move in state.legal_moves:
            mask[move_to_index(state, move)] = True
        return mask

    def apply(self, state, action: int) -> chess.Board:
        move = index_to_move(state, action)
        new_state = state.copy()
        new_state.push(move)
        return new_state

    def terminal_value(self, state) -> float | None:
        if not state.is_game_over(claim_draw=True):
            return None
        outcome = state.outcome(claim_draw=True)
        if outcome.winner is None:
            return 0.0                              # draw (incl. 3-fold rep, 50-move)
        winner_is_current = (outcome.winner == state.turn)
        return 1.0 if winner_is_current else -1.0

    def encode(self, state) -> np.ndarray:
        return encode_board(state)                  # see 5.3

    def canonical_state(self, state) -> chess.Board:
        return state.mirror() if state.turn == chess.BLACK else state

    def symmetries(self, encoded, policy) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(encoded, policy)]                  # chess has no symmetries
```

**Key decisions:**

- **State is a `python-chess` Board, not a numpy array.** `python-chess` is battle-tested for castling, en passant, threefold repetition, 50-move rule, and insufficient material. Reimplementing chess rules would be ~2-4 weeks of work and bug-prone.
- **`is_game_over(claim_draw=True)`** explicitly opts into automatic draw detection. Without it, games could loop forever in repetition cycles.
- **`canonical_state` uses `Board.mirror()`** when Black is to move — flips the board so the NN always sees "White at the bottom". `python-chess` handles castling rights and en passant correctly under mirroring.
- **`symmetries()` returns identity only.** Chess has no exploitable symmetries: castling rights and pawn direction break left-right mirror; player asymmetry (White moves first) breaks rotation. This means ~8× lower data efficiency than TTT and ~2× lower than Connect 4 — partially compensated by parallel self-play volume.

### 5.2 Move encoding — the AlphaZero 8×8×73 scheme

**Why this encoding**: every chess move needs a unique integer index in `[0, 4672)`. We use `(source_square, move_type)` where `move_type ∈ [0, 73)`:

| Category | Indices | Count | Covers |
|---|---|---|---|
| Queen-like moves (8 directions × 7 distances) | 0–55 | 56 | Sliding moves; pawn pushes & captures; king moves (dist=1); queen promotions encoded as queen-direction moves to last rank |
| Knight moves (8 L-shapes) | 56–63 | 8 | All 8 knight destinations from a source |
| Underpromotions (3 pieces × 3 file directions) | 64–72 | 9 | Promotion to knight/bishop/rook (queen promo already covered as queen move) |

`64 squares × 73 move types = 4672` actions total. Each legal move maps bijectively to exactly one action index.

**Helper functions** (`src/alphazero/games/chess_move_encoding.py`):

```python
def move_to_index(state: chess.Board, move: chess.Move) -> int:
    """Convert a python-chess Move to a flat action index in [0, 4672)."""

def index_to_move(state: chess.Board, index: int) -> chess.Move:
    """Convert a flat action index back to a python-chess Move.
    Requires state because the same index can decode to different moves
    depending on what piece is on the source square."""
```

**Cross-references**: alpha-zero-general's chess fork and Lc0's encoding are well-documented references; we'll cross-check our implementation against them via property-based tests in `tests/test_move_encoding.py`.

### 5.3 Input encoding — start simple (~20 planes)

The board state is encoded as 20 channels of an 8×8 tensor (shape `(20, 8, 8)`):

| Plane | Content |
|---|---|
| 0–5 | My pieces by type: pawn, knight, bishop, rook, queen, king |
| 6–11 | Opp pieces by type (same order) |
| 12 | Ones plane (boundary detection — convs with padding mix in zeros from outside the board; ones distinguishes "edge" from "empty middle") |
| 13–16 | Castling rights: my-K, my-Q, opp-K, opp-Q (each plane all-ones if right exists, all-zeros otherwise) |
| 17 | En passant target square (1-hot at the square if any, all-zeros otherwise) |
| 18 | Halfmove clock for 50-move rule (constant plane = halfmove_clock / 100) |
| 19 | Fullmove number / 100 (rough "how late in the game" signal) |

**What we skip (vs paper)**: 8 previous positions × 14 history planes = 112 planes. Threefold repetition is handled by `python-chess`'s detection at terminal time (`is_game_over(claim_draw=True)`). The NN doesn't get a direct "this position has been repeated" signal — it learns from outcomes.

**Tradeoff**:
- Pro: ~6× fewer input planes; smaller input conv; less memory.
- Pro: simpler implementation; less surface area for bugs.
- Con: NN can't strategically anticipate repetition draws; sees them only post-hoc. Mild disadvantage for endgame play. **If this hurts training, add history planes — a 10-line addition to `encode_board()`.** Deferred to Sub-project 4.

### 5.4 Parallel self-play — the major new piece

The current `selfplay.run_one_game()` runs one game at a time serially. For chess this is wall-time impossible — MCTS is Python-bound, GPU sits 95% idle, and a single training run takes days even on A100. **Parallel self-play is required infrastructure for chess.**

**Architecture**:

```
                    ┌─────────────────────────────────────────────┐
                    │ NN-server process (owns the GPU)            │
                    │                                              │
                    │  1. poll request queue                       │
                    │  2. collect ≥B requests OR wait ≤5ms         │
                    │  3. stack → batched forward pass             │
                    │  4. route results back to each worker's      │
                    │     reply queue (preserving order)           │
                    └────────────────┬────────────────────────────┘
                                     ▲ ▼ via mp.Queue
                                     │
        ┌────────────────────────────┼─────────────────────────────┐
        ▼                            ▼                              ▼
   ┌─────────┐                  ┌─────────┐                   ┌─────────┐
   │Worker 0 │                  │Worker 1 │ ... ... ...       │Worker N │
   │         │                  │         │                   │         │
   │ MCTS    │                  │ MCTS    │                   │ MCTS    │
   │ Chess() │                  │ Chess() │                   │ Chess() │
   │         │                  │         │                   │         │
   │ eval_fn │ ── submits state │ eval_fn │                   │ eval_fn │
   │ blocks  │     to server    │ blocks  │                   │ blocks  │
   │ waits   │ ◀── gets back    │ waits   │                   │ waits   │
   │         │     (priors, v)  │         │                   │         │
   └────┬────┘                  └────┬────┘                   └────┬────┘
        │                            │                              │
        ▼                            ▼                              ▼
   completed games → result queue → Trainer collects → ReplayBuffer
```

**Module**: `src/alphazero/parallel_selfplay.py`. Public interface:

```python
def run_parallel_self_play(
    game_factory: Callable[[], Game],
    best_net_state_dict: dict,
    config: TrainingConfig,
    num_games: int,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Run `num_games` self-play games in parallel.

    Spawns config.num_workers worker processes + 1 NN-server.
    Returns a flat list of training tuples (state, π, z), already
    symmetry-augmented if the game has symmetries (chess doesn't).
    """
```

**Implementation notes**:

- **`torch.multiprocessing`** as the concurrency primitive (not plain `multiprocessing`). Handles PyTorch tensor sharing correctly.
- **`mp.Queue`** for request/reply between workers and NN-server.
- **`best_net_state_dict`** is broadcast to workers at startup; each worker loads it into its own NN copy. Cheaper than per-call serialization.
- **Workers do NOT touch CUDA**: only the NN-server interacts with the GPU. Avoids the "CUDA-in-fork" issue (`fork()` after CUDA init is unsafe; we use `spawn()` start method).
- **Clean shutdown**: trainer sends sentinel values down the queues; workers + server check for sentinel each iteration and exit cleanly.
- **Backpressure**: if workers produce requests faster than the GPU can process, the request queue can grow unbounded. Bound the queue and let workers block — natural rate limiting.

**Risk assessment**: this is the highest-risk component in Sub-project 3. Common failure modes:
- Pickling errors when passing objects through queues
- Deadlocks if shutdown ordering is wrong
- Memory leaks from unclosed file descriptors / queues
- "GPU OOM" if `inference_batch_size` is too large for the model size

We budget **~6-10 hours of agent work** for the parallel module alone (vs ~30-60 min per typical task), plus dedicated stress tests.

### 5.5 `StockfishOpponent` — eval ground truth

Uses `python-chess`'s built-in UCI engine wrapper.

```python
class StockfishOpponent:
    def __init__(self, elo: int = 1500, time_per_move: float = 0.5,
                 stockfish_path: str = "stockfish"):
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self.engine.configure({
            "UCI_LimitStrength": True,
            "UCI_Elo": elo,
        })
        self.time_per_move = time_per_move

    def __call__(self, game: Chess, state: chess.Board) -> int:
        result = self.engine.play(state, chess.engine.Limit(time=self.time_per_move))
        return move_to_index(state, result.move)

    def reset(self): pass
    def close(self): self.engine.quit()
    def __del__(self):
        try: self.close()
        except Exception: pass
```

**Requires** `stockfish` binary on PATH:
- macOS: `brew install stockfish`
- Linux/Debian: `apt install stockfish`
- Colab: `!apt install stockfish -y`
- Docker image: `apt-get install -y stockfish` in the Dockerfile

**Stockfish UCI ELO support**: Stockfish 14+ supports `UCI_LimitStrength` + `UCI_Elo` (range 1320–3190). Some older versions don't; the test suite verifies this at import time.

---

## 6. Hyperparameters + compute estimates

### Starting `configs/chess.toml`

```toml
# Network — ~5M params. Scale knobs for Sub-project 4 (target 2000+ ELO).
n_blocks = 10
n_channels = 128

# MCTS
num_simulations = 200          # paper used 800; this is enough for 1500-1800 ELO
c_puct = 2.5
dirichlet_alpha = 0.3          # paper's chess value (lower than TTT/C4)
dirichlet_weight = 0.25

# Self-play
games_per_iteration = 100      # scales linearly with parallel workers
temperature_threshold = 30     # cover full opening repertoire

# Training
training_steps_per_iteration = 2000
batch_size = 256
learning_rate = 1e-3
weight_decay = 1e-4

# Replay buffer
replay_buffer_capacity = 500_000
min_buffer_size = 50_000

# Eval (vs Stockfish 1500 ELO)
eval_interval = 5
eval_games = 100

# Schedule
num_iterations = 80

# Parallel self-play (NEW)
num_workers = 8                # 1 = serial fallback; raise to use more CPU cores
inference_batch_size = 64      # NN-server batches up to N states per forward

# Misc
seed = 42
device = "auto"
checkpoint_dir = "checkpoints"
log_dir = "runs"
```

### Wall-time estimates by hardware

| Setup | Workers | Per iter | Eval cycle | 80 iters total | Cost |
|---|---|---|---|---|---|
| M4 Pro MPS (parallel) | 8 | ~3–5 min | ~3 min | **~5–8 hr** | $0 |
| Colab T4 free (parallel) | 2–3 | ~6–10 min | ~5 min | ~10–15 hr (multiple sessions) | $0 |
| Colab Pro L4 (parallel) | 6–8 | ~2–4 min | ~3 min | **~3–5 hr** | $10/mo flat |
| Lambda A6000 (parallel) | 12 | ~2–3 min | ~2 min | ~3–4 hr | ~$2–3 total |
| Lambda A100 (parallel) | 16 | ~1–2 min | ~1.5 min | **~2–3 hr** | ~$4–7 total |
| **For comparison: any GPU, serial (no parallel self-play)** | 1 | ~30–60 min | ~5 min | ~40–80 hr | prohibitive |

Parallel self-play is the difference between "tractable" and "infeasible."

### Eval-vs-Stockfish wall-time

Stockfish at `time_per_move=0.5` over 100 games × ~50 plies average = ~50 sec × 100 games = ~80 min per eval cycle, **but** this is sequential. With eval-time parallelization (deferred to Sub-project 4), this could drop to ~20 min on 8 CPUs. For SP3, we accept the serial eval cost; it dominates only a small fraction of total wall-time.

### Scaling path to Sub-project 4 (2000+ ELO)

All config changes, **no code changes**:

```toml
n_blocks = 19
n_channels = 256                  # ~46M params (matches AlphaZero paper architecture)
num_simulations = 800             # paper's setting
games_per_iteration = 500
training_steps_per_iteration = 5000
num_iterations = 200
batch_size = 512
num_workers = 16
inference_batch_size = 256
eval_stockfish_elo = 2000
```

Wall-time on Lambda A100: ~50–150 hr. Cost: ~$60–200. We'd also want history planes (10-line addition to `encode_board`) at this scale.

---

## 7. Testing strategy + critical invariants

### Inherited invariants (all 6 still apply)

The same six "silent failure" invariants from Sub-projects 1 and 2 must hold:

1. Self-play uses `best_net`; training updates `candidate_net`.
2. Dirichlet noise: root only, self-play only.
3. Backup flips sign per ply (zero-sum).
4. `canonical_state` applied consistently at every encode site.
5. `z` is assigned per-ply from THAT ply's mover's POV.
6. Replay buffer is NOT reset between iterations.

These are guarded by the existing `tests/test_invariants.py` and don't need re-implementing.

### New chess-specific invariants

| # | Invariant | Failure mode if broken |
|---|---|---|
| 7 | Move encoding is bijective: `index_to_move(state, move_to_index(state, m)) == m` for every legal move in every reachable position | Training data has wrong action targets; the agent learns to play illegal/incorrect moves |
| 8 | Workers don't share `chess.Board` objects | One worker's `push()` corrupts another worker's game; data poisoning |
| 9 | NN-server batched output preserves input ordering | Worker A gets B's value/policy; complete training corruption |
| 10 | Stockfish subprocesses are closed cleanly | Zombie processes accumulate; eventually exhausts process limits |

### Test pyramid

| File | Tests | Coverage |
|---|---|---|
| `tests/test_game_chess.py` | ~25 | Interface (initial_state, current_player, legal_actions_mask, apply, terminal_value, encode, canonical_state); terminal detection for all draw types (3-fold, 50-move, stalemate, insufficient material); canonical_state mirror is self-inverse |
| `tests/test_move_encoding.py` | ~15 | Exhaustive round-trip: every queen direction × distance from every source square; all 8 knight moves; all 9 underpromotions; castling encoded correctly (kingside = king moves 2 east); en passant captures encoded correctly |
| `tests/test_chess_minimax.py` | ~5 | StockfishOpponent plays only legal moves; UCI_Elo configuration takes effect; clean shutdown on `close()` and `__del__` |
| `tests/test_parallel_selfplay.py` | ~10 | Single-worker matches serial behavior; N workers produce N independent games; NN-server preserves batch ordering; clean shutdown with no zombie processes; stress test with 100 games |
| `tests/test_invariants_chess.py` | 4 | The 4 new invariants (#7–#10) |
| `tests/test_e2e_chess.py` (`@pytest.mark.slow`) | 1 | The kill criterion |

### The kill criterion

```python
@pytest.mark.slow
def test_e2e_chess_beats_stockfish_1500(tmp_path, monkeypatch):
    """Train + ≥50% win rate vs Stockfish ELO=1500 over 100 games."""
    monkeypatch.chdir(tmp_path)
    config = load_config(REPO_ROOT / "configs/chess.toml")
    opponent = StockfishOpponent(elo=1500, time_per_move=0.5)

    trainer = Trainer(Chess(), config)
    trainer.run(eval_opponent=opponent)

    agent = trainer._make_argmax_mcts_agent(trainer.best_net)
    result = play_match(Chess(), agent, opponent, num_games=100)
    opponent.close()

    print(f"\nFinal: wins={result.wins_a} draws={result.draws} "
          f"losses={result.losses_a} win_rate={result.win_rate:.2%}")
    assert result.win_rate >= 0.50, (
        f"Agent only reached {result.win_rate:.1%} win rate vs Stockfish 1500 ELO. "
        f"Increase num_iterations or num_simulations in configs/chess.toml."
    )
```

This test is **slow** — full training + 100-game eval against Stockfish takes 3-8 hours depending on hardware. Run via `pytest -m slow tests/test_e2e_chess.py -v -s`.

**Pass criterion: `result.win_rate ≥ 0.50`** against Stockfish at UCI_Elo=1500. 50% means matching 1500 ELO play; >50% means slightly stronger.

---

## 8. Success criterion (recap)

**Sub-project 3 is done when:**

> `pytest -m slow tests/test_e2e_chess.py` passes consistently with `result.win_rate >= 0.50` against `StockfishOpponent(elo=1500, time_per_move=0.5)` over 100 games (alternating colors).

After this passes, Sub-project 4 (productionize: UCI engine, scale to 2000+ ELO) can begin.

---

## 9. Project layout

```
src/alphazero/
├── games/
│   ├── chess_game.py                ← new — Chess(Game) class
│   └── chess_move_encoding.py       ← new — 8×8×73 encoding helpers
├── opponents/
│   └── stockfish.py                 ← new — StockfishOpponent
├── parallel_selfplay.py             ← new — worker pool + NN-server
├── trainer.py                       ← extend — dispatch parallel path when num_workers > 1
└── cli.py                           ← extend — chess dispatch in --game

configs/
└── chess.toml                       ← new — starting hyperparameters

bin/
├── setup.sh                         ← new — portable install with Stockfish detection
└── train_chess.sh                   ← new — convenience wrapper

tests/
├── test_game_chess.py               ← new
├── test_move_encoding.py            ← new
├── test_chess_minimax.py            ← new
├── test_parallel_selfplay.py        ← new
├── test_invariants_chess.py         ← new
└── test_e2e_chess.py                ← new (@pytest.mark.slow)

notebooks/
└── train_chess_colab.ipynb          ← new — sibling to train_connect4_colab.ipynb

Dockerfile                           ← new — reproducible env (Python + Stockfish + torch)
```

### New dependencies

`pyproject.toml`:

```toml
dependencies = [
    "torch>=2.2",
    "numpy>=1.26",
    "tomli>=2.0",
    "tqdm>=4.66",
    "tensorboard>=2.16",
    "python-chess>=1.10",       # NEW
]
```

System-level: `stockfish` binary on PATH. Documented in `bin/setup.sh`:

```bash
#!/usr/bin/env bash
# bin/setup.sh — install all dependencies. Idempotent; safe to re-run.
set -euo pipefail

# Detect platform & install Stockfish
if [[ "$(uname)" == "Darwin" ]]; then
    brew list stockfish &>/dev/null || brew install stockfish
elif [[ -f /etc/debian_version ]]; then
    dpkg -s stockfish &>/dev/null || sudo apt-get install -y stockfish
else
    echo "Unknown platform; install stockfish manually." >&2
fi

# Python deps
pip install -e ".[dev]"

# Verify
python3 -c "
import chess.engine
eng = chess.engine.SimpleEngine.popen_uci('stockfish')
eng.configure({'UCI_LimitStrength': True, 'UCI_Elo': 1500})
print('Stockfish OK; UCI_Elo configurable')
eng.quit()
"
```

---

## 10. Out of scope (saved for future sub-projects)

- **Pushing past 1800 ELO** — Sub-project 4. Requires bigger network (15×192 → 19×256), more sims (400–800), more iterations (150–300), and likely history planes. Cost: hundreds of $ on cloud GPU.
- **UCI engine protocol** — Sub-project 4. Lets the agent play in Lichess / any chess GUI / public bot leagues.
- **History planes** — 10-line addition to `encode_board()` when needed. Mainly helps endgame repetition handling.
- **Distillation from Stockfish / supervised pre-training on master games** — Sub-project 5. Trades "pure self-play" purity for much faster convergence to high ELO.
- **MCTS optimizations**: virtual loss, batched simulations within a single search (different from our worker-pool approach). Marginal gain on top of our setup; ignore for SP3.
- **Bitboard move generation**: `python-chess` is pure-Python and not the fastest. A C-accelerated alternative could be ~5× faster for move generation. Not the bottleneck for us; revisit only if profiling shows it dominates.
- **Distributed training** across multiple GPUs. Not needed at our scale; SP4 may want it.

---

## 11. References

- **AlphaZero paper** — Silver et al. 2017: https://arxiv.org/abs/1712.01815 — primary reference for the 8×8×73 encoding, 119-plane input, and overall architecture.
- **alpha-zero-general** — Surag Nair: https://github.com/suragnair/alpha-zero-general — pedagogical reference (no chess in master, but referenceable for general framework patterns we've already adopted).
- **chess-zero** — https://github.com/Zeta36/chess-alpha-zero — earlier chess-AlphaZero implementation; useful cross-reference for move encoding.
- **Lc0 (Leela Chess Zero)** — https://lczero.org/ — production-grade AlphaZero-style chess engine. Reference for move encoding details + scaling tricks (we deliberately avoid the optimizations they use; we want a clean learning implementation).
- **python-chess docs** — https://python-chess.readthedocs.io/ — primary library reference.
- **Sub-project 1 spec**: [`2026-05-15-alphazero-framework-tictactoe-design.md`](2026-05-15-alphazero-framework-tictactoe-design.md) — the framework Sub-project 3 reuses.
- **Sub-project 2 spec**: [`2026-05-15-connect4-design.md`](2026-05-15-connect4-design.md) — pattern for adding a new game.
