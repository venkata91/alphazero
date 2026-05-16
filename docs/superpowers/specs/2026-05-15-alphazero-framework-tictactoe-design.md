---
title: AlphaZero Framework + Tic-Tac-Toe (Sub-project 1)
date: 2026-05-15
status: Approved (brainstorming → implementation)
sub_project: 1
target: Game-agnostic AlphaZero framework, validated end-to-end on Tic-Tac-Toe
---

# AlphaZero Framework + Tic-Tac-Toe (Sub-project 1)

> **⚠ Design correction (post-implementation):** This spec describes the original design that included AlphaGo Zero-style arena gating. **We removed arena gating during implementation** after it caused training stagnation on TTT (`best_net` froze at iter 5; agent learned a defeatist O policy; lost 100% of O-side games against perfect play). The Trainer now matches the AlphaZero (2017) paper: `_promote_candidate_to_best` is called unconditionally after each iteration. See [`concepts.html`](../../../concepts.html) concept card #3 for the visual story, or [Sub-project 2 spec §6 "Lessons from Sub-project 1"](../specs/2026-05-15-connect4-design.md) for the technical post-mortem. The arena-related sections of this spec are preserved as historical record of the original design; the current code does not implement them.

## 1. Context

This is the first of a multi-stage project to build an AlphaZero-style chess engine from scratch.
The user's primary goal is **learning**: build the full AlphaZero pipeline (board representation → neural network → MCTS → self-play → training loop), understand each component deeply, and let the achieved ELO emerge from the work — rather than chasing a target ELO via shortcuts.

The project is decomposed into five sub-projects. This document specifies **Sub-project 1**.

| # | Scope | Target | Estimate |
|---|---|---|---|
| **1** | **AlphaZero framework + Tic-Tac-Toe (this spec)** | Pipeline validated; zero losses vs perfect TTT solver | 2–3 weeks |
| 2 | Connect 4 plugged into the same framework | Beats minimax-depth-8 reliably | 1–2 weeks |
| 3 | Chess (python-chess wrapper, scaled net, core training) | 1500–1800 ELO from scratch | 4–8 weeks |
| 4 | Productionize: UCI protocol, cloud training, longer runs | 1800–2300 ELO | open-ended |
| 5 | Hybrid (supervised pre-training, distillation) | 2500+ ELO | open-ended |

Sub-project 1 produces the reusable framework. Everything downstream depends on it being correct.

---

## 2. Goals

- Build a **game-agnostic** framework where everything except the `Game` class is reusable across TTT, Connect 4, and chess.
- Validate the full AlphaZero loop end-to-end on Tic-Tac-Toe — small enough that bugs are obvious, large enough that the pipeline is non-trivial.
- Reach the **kill criterion**: an agent trained on TTT loses **zero games** out of 200 against a perfect minimax solver. (TTT is a draw with optimal play; a correctly trained agent must match.)
- Keep boundaries between layers strict enough that each component is independently testable.

## 3. Non-goals

These are explicitly out of scope for Sub-project 1; some are picked up in later sub-projects.

- Performance optimization (single-process; we will not parallelize self-play yet).
- Cloud / distributed training.
- GPU-specific kernels or torch.compile experiments.
- UCI engine protocol (Sub-project 4).
- Connect 4 / chess game implementations (Sub-projects 2 and 3).
- Hyperparameter sweeps. We pick reasonable defaults from the AlphaZero paper and tune only if convergence fails.

---

## 4. Architecture

The framework is structured as **four orthogonal layers** with strict boundaries.
The most important design property: **MCTS does not import the neural network, and the neural network does not import MCTS.** They communicate through a function signature (`eval_fn`). This makes each layer independently testable and means moving to Connect 4 or chess in later sub-projects requires changing **only the `Game` class** — every other layer is reusable.

```
┌─────────────────────────────────────────────────────────────┐
│  L4. Training orchestration (Trainer)                        │
│      self-play → buffer → train → arena, looping             │
└──┬───────────┬───────────┬───────────┬──────────────────────┘
   │           │           │           │
   ▼           ▼           ▼           ▼
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│SelfPlay │ │ Replay  │ │ Arena   │ │ Eval vs │
│ Worker  │ │ Buffer  │ │         │ │ Solver  │
└────┬────┘ └─────────┘ └────┬────┘ └─────────┘
     │                       │
     ▼                       ▼
┌────────────────────┐ ┌────────────────────────┐
│ L3. MCTS           │ │ L2. Game (rules,       │
│   PUCT, expand,    │ │     encoding,          │
│   backup           │ │     symmetries)        │
└──────────┬─────────┘ └────────────────────────┘
           │ eval_fn(state) → (priors, value)
           ▼
┌──────────────────────────────────────────────────────────────┐
│  L1. AlphaZeroNet — PyTorch ResNet with policy + value heads │
│      Pure forward pass; knows nothing about MCTS or training │
└──────────────────────────────────────────────────────────────┘
```

| Layer | Component | Imports | Does NOT know about |
|---|---|---|---|
| L1 | `AlphaZeroNet` | torch, `Game.input_shape`, `Game.action_size` | MCTS, self-play, training loop |
| L2 | `Game` (and `TicTacToe`) | nothing | NN, MCTS, training |
| L3 | `MCTS` | `Game`, an `eval_fn` callable | NN class, training, self-play |
| L4 | `SelfPlayWorker`, `ReplayBuffer`, `Arena`, `Trainer` | everything above | (top-level) |

This table is the design's load-bearing claim. If we ever find ourselves importing MCTS inside the NN or `Game` inside the trainer's `train_step`, we've broken an abstraction.

---

## 5. Components

### 5.1 `Game` interface (+ `TicTacToe` implementation)

**Role**: encapsulates everything game-specific so MCTS / NN / training don't know which game they're operating on.

```python
class Game(ABC):
    input_shape: tuple[int, ...]   # e.g. (3, 3, 3) for TTT — (channels, H, W)
    action_size: int                # e.g. 9 for TTT

    def initial_state() -> State
    def current_player(state) -> int                     # +1 or -1
    def legal_actions_mask(state) -> np.ndarray[bool]    # shape (action_size,)
    def apply(state, action: int) -> State
    def terminal_value(state) -> Optional[float]         # None if not terminal; else ±1 / 0 from current player's POV
    def encode(state) -> np.ndarray                      # shape == input_shape
    def canonical_state(state) -> State                  # rewrite state so current player is "+1"
    def symmetries(encoded_state, policy) -> list[tuple] # data augmentation; default returns [(s, π)]
```

**Key decisions:**

- **`canonical_state` is the "always-as-+1" trick.** The NN never has to learn "whose turn is it" — `canonical_state` rewrites the board from the current player's perspective. Roughly halves the effective state space the NN must model.
- **`terminal_value` is from current-player POV**, not "white wins / black wins". Consistent with canonical encoding.
- **`symmetries`** is data augmentation. TTT has 8 (the D₄ dihedral group: 4 rotations × 2 reflections). Connect 4 has 2 (left-right mirror). **Chess has none** because castling and pawn direction break symmetry. Default returns `[(s, π)]`; TTT overrides with full 8.

**TTT-specific encoding**: state is a 3×3 numpy array with +1 / −1 / 0. Encoding into a (3, 3, 3) tensor uses 3 planes: my-pieces, opponent-pieces, all-ones (the constant plane helps small CNNs reason about board edges — a standard trick).

### 5.2 `AlphaZeroNet`

**Role**: PyTorch module. Encoded state batch → (policy logits, value).

```python
class AlphaZeroNet(nn.Module):
    def __init__(self, input_shape, action_size, n_blocks=4, n_channels=32): ...
    def forward(x) -> (policy_logits[B, action_size], value[B])
```

Structure: input conv → N residual blocks → two heads (policy + value).

A residual block is:

```python
out = relu(x + bn2(conv2(relu(bn1(conv1(x))))))
```

Loss is the AlphaZero loss:

```python
policy_loss = −sum(π · log_softmax(policy_logits))  # soft target — not one-hot
value_loss  = mse(value, z)
loss        = policy_loss + value_loss              # L2 regularization via AdamW(weight_decay)
```

**Key decisions:**

- **AdamW (not SGD+momentum).** The original AlphaZero paper used SGD+momentum, but AdamW is the modern pragmatic choice for replications: faster convergence on small games with no observable downside.
- **Soft cross-entropy.** Target π is a *probability distribution* from MCTS (not a one-hot of "best move"). We compute `−sum(π * log_softmax(logits))` manually rather than using `nn.CrossEntropyLoss` (which expects class indices).
- **Legal-action masking at inference**: set illegal action logits to −∞ before softmax. Not applied during training (loss is conditional on legal moves only, since π itself only has support on legal moves).
- **Sizing for TTT**: 4 blocks × 32 channels (~76K params) is plenty. For chess later: 10–20 blocks × 128–256 channels.

### 5.3 `MCTS`

**Role**: from a given state, run N simulations, return an improved policy (visit distribution).

```python
class MCTS:
    def __init__(self, game, eval_fn, c_puct=1.5,
                 dirichlet_alpha=1.0, dirichlet_weight=0.25): ...
    def search(self, root_state, num_simulations: int,
               add_root_noise: bool) -> np.ndarray
        # returns shape (action_size,), summing to 1.0
```

`eval_fn` is `(encoded_state) -> (priors, value)`. **This is the boundary** — MCTS does not know it is calling a neural network. In tests we mock it with hand-coded priors and verify MCTS behavior independently.

Tree node:

```python
@dataclass
class Node:
    prior: float                      # P(s, a) of the action that led HERE
    visit_count: int                  # N
    value_sum: float                  # W (cumulative)
    children: dict[int, Node]         # action_index → child
    is_expanded: bool

    @property
    def Q(self): return self.value_sum / max(1, self.visit_count)
```

**Key decisions:**

- **PUCT formula** as in the AlphaZero paper:

  ```
  U(s, a) = c_puct · P(s, a) · sqrt(sum_b N(s, b)) / (1 + N(s, a))
  score   = Q(s, a) + U(s, a)
  ```

- **Dirichlet noise at root** (only during self-play, never arena / eval):

  ```
  priors[root] = 0.75 · priors[root] + 0.25 · Dirichlet(α)
  ```

  α scales inversely with average legal moves: 1.0 for TTT, 0.3 for chess in the paper.

- **Sign flip on backup**: zero-sum games. After backing up by 1 ply, `value = −value`, so each node's Q is always from *its own* current-player POV.

- **Temperature lives outside MCTS** — in the self-play caller. MCTS returns raw visit counts; the caller chooses argmax (τ=0) or sample-by-visits (τ=1).

### 5.4 `SelfPlayWorker`

**Role**: play one full game using MCTS, emit training tuples.

```python
def run_one_game(game, eval_fn, num_simulations,
                 temperature_threshold=6) -> list[(s, π, z)]:
    state = game.initial_state()
    mcts = MCTS(game, eval_fn, ...)
    history = []                       # list of (encoded_s, π, current_player)
    move = 0
    while game.terminal_value(state) is None:
        canon = game.canonical_state(state)
        π = mcts.search(canon, num_simulations, add_root_noise=True)
        action = (sample_from(π) if move < temperature_threshold
                  else π.argmax())
        history.append((game.encode(canon), π, game.current_player(state)))
        state = game.apply(state, action)
        move += 1

    # Game terminated — assign z per ply from THAT PLY'S MOVER's POV
    z_final = game.terminal_value(state)
    examples = []
    # for each (s, π, player_at_ply) in history:
    #     z_at_ply = z_final * sign_for(player_at_ply, final_player)
    return augment_with_symmetries(examples)
```

**Key decisions:**

- **Temperature schedule**: τ=1 (sample) for first N moves to encourage diverse openings; τ=0 (argmax) after, to exploit. N=6 for TTT (basically the entire game given TTT's 9-ply maximum), N≈30 for chess.
- **Symmetry augmentation happens at the worker**, not in the buffer. TTT: 1 game → up to 8× training tuples. The augmentation logic lives next to the game-specific code where it logically belongs.
- **z-assignment** is the most bug-prone part of the entire framework. Each ply's z is the *eventual* game outcome from *that ply's mover's* POV. Worth dedicated unit tests (see §8.2 "SelfPlayWorker tests").

### 5.5 `ReplayBuffer`

**Role**: fixed-size FIFO of training tuples.

```python
class ReplayBuffer:
    def __init__(capacity: int)
    def add(examples: list[tuple]) -> None
    def sample(batch_size: int) -> (state_batch, policy_batch, value_batch)
    def __len__()
```

Backed by `collections.deque(maxlen=capacity)`. **No prioritization** — uniform random sampling is what the AlphaZero paper specifies and what we'll use. Prioritized experience replay has been tried by replications with mixed results; not worth the complexity in v1.

### 5.6 `Trainer` (orchestration)

**Role**: the outer loop. Owns the network, the replay buffer, the schedule.

```python
class Trainer:
    def run(self):
        for iteration in range(config.num_iterations):
            # 1. Self-play with current best NN
            for _ in range(config.games_per_iteration):
                examples = run_one_game(
                    game,
                    eval_fn=self.best_net,
                    num_simulations=config.num_simulations,
                )
                self.replay_buffer.add(examples)

            # 2. Train: K minibatches sampled from buffer
            if len(self.replay_buffer) >= config.min_buffer_size:
                for _ in range(config.training_steps_per_iteration):
                    batch = self.replay_buffer.sample(config.batch_size)
                    self._train_step(batch)

            # 3. Arena every N iterations
            if iteration % config.arena_interval == 0:
                win_rate = arena.play_match(
                    candidate=self.candidate_net,
                    opponent=self.best_net,
                    num_games=config.arena_games,
                )
                if win_rate >= config.arena_threshold:
                    self.best_net = copy.deepcopy(self.candidate_net)
                else:
                    self.candidate_net.load_state_dict(self.best_net.state_dict())

            # 4. Eval vs perfect solver every M iterations
            if iteration % config.eval_interval == 0:
                self._eval_vs_solver()

            # 5. Checkpoint
            torch.save({...}, f"checkpoints/iter_{iteration}.pt")
```

**Key decisions:**

- **Self-play uses `best_net`, training updates `candidate_net`.** This is the most critical and easiest-to-miss design rule. New training updates `candidate_net`; `best_net` only updates after arena acceptance. Without this distinction, you can poison self-play data with a regressed network.
- **Revert on arena rejection**: load `best_net`'s weights back into `candidate_net`. This makes the candidate try again from the same starting point next iteration.
- **Single-process** for Sub-project 1. Self-play, training, arena — all serial. Parallelism is a Sub-project 3 problem.

### 5.7 `Arena`

```python
def play_match(game, agent_a, agent_b, num_games=40) -> float:
    """Returns agent_a's win rate. Alternates colors to split the white-mover advantage."""
```

Agents are callables `(state) -> action_int`. The MCTS-with-current-NN agent is built by wrapping `(network) -> agent`: a function that for any given state runs MCTS with `eval_fn = network`, **τ=0**, **no root noise**, and returns argmax. Same `MCTS` class, different settings.

### 5.8 Perfect TTT solver (eval ground truth)

```python
def solve_tictactoe(state, player) -> int:   # returns +1 / 0 / −1
    """Standard alpha-beta minimax. TTT has ~5478 states; runs in <1ms cached."""
```

Used in evaluation:

```python
def eval_vs_solver(agent, num_games=200) -> dict:
    """Play agent (alternating colors) vs perfect-play opponent.
    Return: {wins, draws, losses}. Pass criterion: zero losses."""
```

TTT is a draw with perfect play. A correctly trained agent should always draw or win — never lose. Any loss = a bug somewhere in the pipeline.

---

## 6. Data flow

Three nested loops: the iteration (Trainer.run), one self-play game, one MCTS simulation.

### 6.1 One training iteration

```
ITERATION N
  best_net (frozen for this iteration)
        │
        ▼
  SelfPlay × 100 games  ──→  ReplayBuffer  ──→  train_step × 500  ──→  candidate_net
   (uses best_net)          (FIFO 50K)        (AdamW, lr=1e-3)
                                                                            │
                                                                            ▼
                            ┌── Arena (every K iter) ──────────────────────┐
                            │  candidate vs best, 40 games                  │
                            │  win_rate ≥ 0.55 → best_net := candidate     │
                            │  else            → candidate := best_net      │
                            └───────────────────────────────────────────────┘
                                                                            │
                            Periodic: eval_vs_solver, checkpoint            │
                                                                            ▼
                                                                  (next iteration)
```

**Concrete TTT scale**: 1 iteration ≈ 100 games × ~7 plies × 8 symmetries = ~5,600 training tuples. 500 minibatches × batch_size 64 = 32K samples seen per iteration. ~30–50 iterations to convergence. Estimated wall-time on M4 Pro: **15–45 minutes total** for the full Sub-project 1 training run.

### 6.2 One self-play game

```
state₀ = game.initial_state()
loop while game.terminal_value(state) is None:
    canon  = game.canonical_state(state)
    π      = MCTS.search(canon, num_simulations=50, add_root_noise=True)
    action = sample(π)  if move < temperature_threshold
             argmax(π)  otherwise
    record (game.encode(canon), π, current_player(state))
    state  = game.apply(state, action)
end loop

z_final = game.terminal_value(state)
for each recorded ply, assign z from THAT PLY'S mover's POV
augment with symmetries  (TTT: 1 game → up to 8× training tuples)
```

### 6.3 One MCTS simulation

```
1. SELECT
   path = [root]; node = root
   while node.is_expanded and not terminal:
       a = argmax_a [ Q(s, a)  +  c_puct · P(s, a) · √ΣN / (1 + N(s, a)) ]
       node = node.children[a];  path.append(node)

2. EXPAND
   if terminal:
       leaf_value = game.terminal_value(node.state)
   else:
       priors, leaf_value = eval_fn(node.state)
       priors = mask_illegal(priors, legal_mask); renormalize
       for a in legal_actions:
           node.children[a] = Node(prior=priors[a], ...)
       node.is_expanded = True

3. BACKUP
   value = leaf_value
   for node in reversed(path):
       node.visit_count += 1
       node.value_sum   += value
       value = −value      # zero-sum sign flip
```

After 50 such simulations, π = visit_counts / sum at the root.

Dirichlet noise is applied to the root's children's priors **exactly once, before the 50 simulations start**, and **only during self-play**.

### 6.4 One training step

```python
batch = replay_buffer.sample(64)
states, target_πs, target_zs = batch.tensors

policy_logits, values = candidate_net(states)
policy_loss = −(target_πs * log_softmax(policy_logits, dim=-1)).sum(dim=-1).mean()
value_loss  = F.mse_loss(values.squeeze(-1), target_zs)
loss = policy_loss + value_loss   # L2 reg via AdamW(weight_decay=1e-4)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

Roughly 10 lines of PyTorch. The complexity in this project lives in MCTS and self-play, **not** in training itself.

---

## 7. Critical invariants

Six rules that, if broken, cause **silent training failure** (training looks like it's running but the agent never improves, or improves and then collapses). These each become a named regression test.

| # | Invariant | Failure mode if broken |
|---|---|---|
| 1 | Self-play uses `best_net`; training updates `candidate_net`. | Self-play data is generated by a regressed network → buffer poisoned → next training step trains on bad data. |
| 2 | Dirichlet noise: root only, self-play only. | Adding it during arena breaks the head-to-head signal. Adding it deep in the tree corrupts search. |
| 3 | Backup flips sign per ply (zero-sum). | Q values don't represent current-player POV → PUCT picks the wrong moves → MCTS plays badly. |
| 4 | `canonical_state` applied consistently at every encode site. | NN sees inconsistent inputs across positions → never converges. |
| 5 | Z is assigned per-ply from *that ply's mover's POV*, not the final-mover's POV. | Value targets are systematically wrong; value head never learns. |
| 6 | Replay buffer is NOT reset between iterations. | Training signal becomes too narrow per iteration; loss oscillates. |

Some of these are reinforced by the type system / shape of the API (#3 is enforced by MCTS internals; #6 is enforced by the trainer never calling `replay_buffer.clear()`). Others require explicit tests (#1, #2, #4, #5).

---

## 8. Testing strategy

### 8.1 Philosophy

- **Test-first for the high-risk components**: MCTS, self-play z-assignment, training loop. Write the test (with hand-computed expected values) *before* the implementation.
- **Mock at the layer boundary**: MCTS tested with a mock `eval_fn`; NN tested with synthetic batches; self-play tested with a fixed-prior eval_fn. Isolates failures.
- **The 6 invariants from §7 each get a named regression test.**
- **Slow tests gated.** End-to-end "train + verify zero losses vs perfect solver" takes 15+ minutes; mark `@pytest.mark.slow` so unit-test runs stay under a minute.

### 8.2 Test pyramid (by component)

**`Game` / `TicTacToe`** — pure-function tests; coverage target near 100%:

- `test_initial_state_is_empty`
- `test_apply_places_correct_piece` + `test_apply_advances_player`
- `test_legal_actions_correct_for_known_positions`
- `test_terminal_value_x_wins_row` / `col` / `diag` / `draw` / `in_progress`
- `test_encode_shape_matches_input_shape`
- `test_canonical_state_is_identity_when_player_is_plus_one`
- `test_canonical_state_inverts_when_player_is_minus_one`
- `test_symmetries_yields_8_distinct_tuples` + `test_symmetries_includes_identity`

**`AlphaZeroNet`** — shape and convergence sanity:

- `test_forward_pass_shape`
- `test_value_in_range` (tanh → [−1, +1])
- `test_save_load_state_dict_roundtrip`
- `test_loss_decreases_on_synthetic_overfit_batch` (classic "can the net memorize a tiny batch?")

**`MCTS`** — deterministic with mocks; coverage target near 100%:

- `test_search_returns_distribution_summing_to_one`
- `test_uniform_prior_uniform_value_produces_uniform_visits` (symmetry)
- `test_prior_favoring_action_a_concentrates_visits_at_a`
- `test_value_favoring_child_a_concentrates_visits_at_a` (exploit-only)
- `test_puct_formula_at_known_state` (hand-compute, assert numeric)
- `test_dirichlet_noise_changes_root_priors_only`
- `test_dirichlet_noise_disabled_when_add_noise_is_false`
- `test_backup_flips_sign_per_ply` (zero-sum invariant)
- `test_terminal_leaf_uses_terminal_value_not_eval_fn`
- `test_legal_actions_only_get_children`

**`SelfPlayWorker`** — focus on z-assignment:

- `test_one_game_runs_to_terminal`
- `test_history_length_matches_plies_played`
- `test_z_assignment_x_wins_on_move_5` ← critical
- `test_z_assignment_o_wins_on_move_4` ← critical
- `test_z_assignment_draw_assigns_zero_to_every_ply` ← critical
- `test_symmetry_augmentation_multiplies_by_8` (for TTT)
- `test_temperature_zero_picks_argmax`
- `test_temperature_one_samples_proportionally`

**`ReplayBuffer`** — boring but covered:

- `test_capacity_respected`
- `test_overflow_drops_oldest`
- `test_sample_returns_batch_size`
- `test_sample_is_approximately_uniform`

**`Arena`**:

- `test_same_agent_self_match_approximately_fifty_percent`
- `test_stronger_agent_wins` (e.g., MCTS-200 vs MCTS-50)
- `test_color_alternation`

**`Trainer`** — light unit testing, heavy integration:

- `test_iteration_advances_state`
- `test_arena_acceptance_replaces_best_net`
- `test_arena_rejection_reverts_candidate_to_best`
- `test_checkpoint_save_and_load`

### 8.3 The 6-invariants map

| Invariant | Test |
|---|---|
| Self-play uses best_net | `test_selfplay_calls_best_net_not_candidate` (install counting wrapper, assert `best_net.forward` called and `candidate_net.forward` not called) |
| Dirichlet noise scope | `test_dirichlet_noise_changes_root_priors_only` + `test_arena_disables_dirichlet_noise` |
| Backup sign flip | `test_backup_flips_sign_per_ply` |
| Canonical state | enforced structurally by `Game` API; cross-check via `test_encode_swaps_planes_when_player_changes` |
| Z per-ply mover's POV | The three `test_z_assignment_*` tests |
| Buffer not reset | `test_replay_buffer_not_reset_between_iterations` |

### 8.4 Kill criterion (the success bar)

```python
@pytest.mark.slow
def test_e2e_tictactoe_reaches_zero_losses_vs_perfect_solver():
    """Train an AlphaZero agent on TTT. Play 200 games vs perfect minimax solver.
    Assert: 0 losses. (TTT is a draw with perfect play; agent must match.)"""

    trainer = Trainer(game=TicTacToe(), config=TTT_TRAINING_CONFIG)
    trainer.run()   # ~15 min on M4 Pro

    agent = make_mcts_agent(trainer.best_net, num_simulations=50, temperature=0.0)
    results = play_match(TicTacToe(), agent, PerfectTTTSolver(), num_games=200)
    assert results.losses == 0, f"Agent lost {results.losses} games to perfect play"
```

**This test passing is the definition of Sub-project 1 success.** Sub-project 2 (Connect 4) can begin only after this passes consistently.

### 8.5 Tooling

- `pytest` + `pytest-xdist` for parallelism
- `@pytest.mark.slow` for the kill-criterion E2E test
- `unittest.mock` (stdlib) is sufficient; no external mocking framework
- Coverage target: 90%+ on `games/`, `mcts.py`, `selfplay.py`. Don't chase coverage on the trainer loop body.

---

## 9. Hyperparameter defaults (Tic-Tac-Toe)

Frozen dataclass; values are starting points from the AlphaZero paper adapted to TTT scale.

```python
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

    # Eval vs solver
    eval_interval: int = 5
    eval_games: int = 200

    # Schedule
    num_iterations: int = 50

    # Misc
    seed: int = 42
    device: str = "auto"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "runs"
```

**The handful that actually matter:**

| Knob | Why |
|---|---|
| `num_simulations = 50` | Most impactful. Too few (5–10) → MCTS adds little over raw NN. Too many → self-play crawls. 50 is enough on TTT's tiny tree. |
| `c_puct = 1.5` | Lower bound of the typical range; appropriate for small branching factor. Chess uses 2.5–4. |
| `temperature_threshold = 6` | For TTT this is nearly the whole game. Schedule mostly matters for longer games; kept for forward compatibility. |
| `arena_threshold = 0.55` | AlphaGo Zero value. AlphaZero (2017) dropped arena; we keep it for safety during development. |
| `min_buffer_size = 5,000` | Prevents training on a tiny correlated buffer in iteration 1. Buffer fills after ~100 games × ~50 (with symmetries) in iter 1. |
| `weight_decay = 1e-4` | AlphaZero paper value. Higher would underfit; lower would risk overfitting. |

Loaded from `configs/tictactoe.toml` so the exact config is saved alongside checkpoints without code changes.

---

## 10. Project layout

```
alphazero/                            # repo root (~/git/alphazero)
├── pyproject.toml                    # PEP 621 metadata + dependencies
├── .gitignore                        # ✓ exists
├── concepts.html                     # ✓ exists — visual primer
├── tools/
│   └── render_spec.py                # md → styled html (this spec, future specs)
├── configs/
│   └── tictactoe.toml                # TrainingConfig for TTT
├── docs/superpowers/specs/
│   ├── 2026-05-15-alphazero-framework-tictactoe-design.md   ← this doc
│   └── 2026-05-15-alphazero-framework-tictactoe-design.html ← rendered
├── src/alphazero/                    # source package (src-layout)
│   ├── __init__.py
│   ├── config.py                     # TrainingConfig + load_config()
│   ├── games/
│   │   ├── __init__.py
│   │   ├── base.py                   # Game ABC
│   │   └── tictactoe.py
│   ├── solvers/
│   │   ├── __init__.py
│   │   └── tictactoe_solver.py       # perfect minimax (eval ground truth)
│   ├── network.py                    # AlphaZeroNet + ResidualBlock
│   ├── mcts.py                       # MCTS + Node
│   ├── selfplay.py                   # run_one_game + z-assignment
│   ├── replay_buffer.py
│   ├── arena.py                      # play_match
│   ├── trainer.py                    # Trainer.run — outer loop
│   ├── eval_vs_solver.py             # eval harness for TTT
│   └── cli.py                        # python -m alphazero ...
├── tests/
│   ├── __init__.py
│   ├── conftest.py                   # shared fixtures
│   ├── test_game_tictactoe.py
│   ├── test_network.py
│   ├── test_mcts.py
│   ├── test_selfplay.py
│   ├── test_replay_buffer.py
│   ├── test_arena.py
│   ├── test_trainer.py
│   ├── test_invariants.py            # the 5 named invariant tests
│   └── test_e2e_tictactoe.py         # kill criterion (@pytest.mark.slow)
├── checkpoints/                      # .gitignored
└── runs/                             # .gitignored
```

Notes:

- **`src/` layout** over flat. Prevents accidental imports from the source tree when an installed copy exists.
- **`configs/*.toml`** lets us version-control hyperparameters separately from code.
- **No README.md yet** — added when the project is presentable.

---

## 11. Dependencies & tooling

```toml
[project]
name = "alphazero"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.2",         # PyTorch — MPS backend works on M4 Pro
    "numpy>=1.26",
    "tomli>=2.0",         # config loading (3.11+ has tomllib; tomli for compat)
    "tqdm>=4.66",         # progress bars during self-play
    "tensorboard>=2.16",  # local logging — no account / cloud needed
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-xdist>=3.5",
    "ruff>=0.4",          # lint + format
    "markdown>=3.10",     # for tools/render_spec.py
]
```

**Choices:**

- **TensorBoard, not wandb.** TensorBoard is local-only and zero-config. wandb requires an account and uploads metadata to a third party. Can swap later.
- **`python-chess` not yet.** Saved for Sub-project 3. Including it here would imply we're building toward chess in Sub-project 1, which we're not.
- **`uv`** (recommended) or `pip` + venv for package management.
- **Python 3.11+**. Tested primarily on 3.11.

---

## 12. CLI surface

```
python -m alphazero train --config configs/tictactoe.toml
python -m alphazero eval  --checkpoint checkpoints/iter_30.pt --opponent perfect_solver
python -m alphazero play  --checkpoint checkpoints/iter_30.pt --as x    # nice-to-have
```

The interactive `play` subcommand is a nice-to-have for sanity checks — useful for verifying the trained agent "feels right" (blocks obvious wins, takes obvious wins) beyond the numeric eval.

---

## 13. Success criterion (recap)

Sub-project 1 is **done** when:

> `pytest -m slow tests/test_e2e_tictactoe.py` passes consistently:
> a trained agent loses **zero** games out of 200 against a perfect TTT minimax solver, alternating colors.

Only after this passes does Sub-project 2 (Connect 4) begin.

---

## 14. Out of scope (saved for future sub-projects)

- **Parallel self-play** (Sub-project 3) — necessary for chess scale, premature for TTT.
- **Cloud / distributed training** (Sub-project 4) — Colab Pro / AWS only if we hit hardware limits.
- **UCI protocol** (Sub-project 4) — for playing in chess GUIs / on Lichess.
- **Supervised pre-training** (Sub-project 5) — only if pure self-play hits an ELO ceiling.
- **Bitboard move generator** (Sub-project 3+) — only if `python-chess` becomes the bottleneck.
- **Prioritized replay** — uniform sampling works fine per the paper.
- **Network architecture experiments** — stick with the paper's ResNet shape; revisit if convergence fails.

---

## 15. References

- Silver et al., *Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm* (AlphaZero), 2017 — primary reference for architecture and hyperparameters.
- Silver et al., *Mastering the game of Go without human knowledge* (AlphaGo Zero), 2017 — the arena gate originates here.
- `concepts.html` (in this repo) — visual primer on MCTS, ResNet, PUCT, replay buffer, arena.
