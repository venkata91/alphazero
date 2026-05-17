# Sub-Project 3b: Supervised Pre-Training + AlphaZero Refinement

> **Status:** Design — pending approval to proceed to implementation plan.
> **Predecessor:** [SP3 — Chess on the AlphaZero framework](./2026-05-16-chess-design.md) (complete; framework code shipped).
> **Successor:** SP4 — Productionize (UCI engine, 2000+ ELO target).

---

## 1. Context

SP3 shipped the chess framework end-to-end: `Chess(Game)`, AlphaZero 8×8×73 move encoding, `StockfishOpponent`, parallel self-play with NN-server + workers, Trainer dispatch, CLI, configs, tests. The framework works.

However, when the user launched the SP3 training run on M4 Pro MPS:

```
[iter 1/80]
  parallel self-play (iter 1): 8 workers × 100 games
  checkpoint: checkpoints/iter_0001.pt (13838.9s)
```

That is **~3.85 hours per iteration**, projecting **~13 days** for the full 80-iteration run. Pure AlphaZero self-play from random initialization is wall-clock infeasible on consumer hardware for chess.

The root cause is the "cold start" problem: an untrained network produces near-uniform priors, MCTS exploration is uninformed, games are very long (hit the 75-move rule because random moves rarely deliver checkmate), and the resulting (state, π, z) tuples are weak training signal.

## 2. Goal

Train a chess agent that meets the SP3 kill criterion (≥50% win rate vs Stockfish ELO=1500 over 100 games) in **wall-clock days, not weeks** on M4 Pro by:

1. **Bootstrapping the network via supervised pre-training** on a corpus of strong-engine games. After pre-training, the policy head produces meaningful priors and the value head estimates positions reasonably — eliminating the cold start.
2. **Refining the pre-trained network with a short AlphaZero loop** (10 iterations instead of 80). The MCTS-improved targets push the agent past the teacher's behavior-cloning ceiling.

Wall-clock target: **~16 hours total on M4 Pro** (or ~10 hours if AZ refinement runs on Colab L4).

## 3. Non-goals

- Beating full-strength Stockfish, or reaching 3000+ ELO. Those are SP4 / SP5 targets.
- Multi-PV soft policy targets (top-K weighted by eval). The simpler hard-one-hot target is enough for v1.
- Using a Lichess / CCRL human-game corpus. Stockfish self-generation keeps the pipeline self-contained and deterministic. Lichess corpora are an SP5 enhancement.
- Distillation from a teacher network. Out of scope for SP3b.
- UCI engine wrapper (lichess-bot integration). SP4.

## 4. What's new vs what's reused

**New code (~480 lines):**

- `scripts/generate_chess_corpus.py` (~150 lines) — multi-process Stockfish-vs-Stockfish game generator
- `src/alphazero/supervised.py` (~150 lines) — `load_corpus`, `pretrain_supervised`, early stopping, lr schedule
- `src/alphazero/cli.py` — small extension (~30 lines) for new `pretrain` subcommand
- `tests/test_chess_corpus.py` (~60 lines), `tests/test_supervised.py` (~80 lines), `tests/test_cli.py` extension (~30 lines)
- `configs/chess-pretrain.toml`, `configs/chess-refine.toml` (new)

**Reused unchanged:**

- `AlphaZeroNet` — same architecture, just sized larger (n_blocks=15, n_channels=192 ≈ 15M params)
- `Chess(Game)` — state encoding, move encoding, terminal_value, canonical_state
- `chess_move_encoding.py` — `move_to_index` and `index_to_move` used during corpus generation
- `StockfishOpponent` — used by the corpus generator (with `Threads=1`)
- `parallel_selfplay.py` — used by AZ refinement phase via `Trainer.run()`
- `Trainer.run()` and the `--resume-from` flag — loads pre-trained checkpoint, continues AZ loop unchanged
- `_train_step`, `_assign_z`, replay buffer, all framework primitives

## 5. Architecture: three phases producing three checkpoints

```
Phase 1: Corpus generation
  ┌────────────────────────────────────────┐
  │  scripts/generate_chess_corpus.py      │
  │  10 parallel workers, each playing     │
  │  Stockfish-vs-Stockfish games          │
  └─────────────────┬──────────────────────┘
                    │ writes
                    ▼
        data/chess_corpus/shard_NNNN.npz  (~300 shards, ~3.8GB)

Phase 2: Supervised pre-training
  ┌────────────────────────────────────────┐
  │  python -m alphazero pretrain \         │
  │    --corpus data/chess_corpus/ \        │
  │    --config configs/chess-pretrain.toml │
  └─────────────────┬──────────────────────┘
                    │ writes
                    ▼
              pretrained.pt

Phase 3: AlphaZero refinement
  ┌────────────────────────────────────────┐
  │  python -m alphazero train \            │
  │    --game chess \                       │
  │    --config configs/chess-refine.toml \ │
  │    --resume-from pretrained.pt          │
  └─────────────────┬──────────────────────┘
                    │ writes
                    ▼
        checkpoints/iter_0001.pt ... iter_0010.pt
```

Each phase is independently restartable. If pre-training converges and AZ refinement disappoints, you can re-run just Phase 3 with different hyperparameters from the same `pretrained.pt`.

## 6. Phase 1: Corpus generation

### 6.1 Architecture: embarrassingly parallel

Unlike `parallel_selfplay.py` (which needs a central NN-server to batch GPU inference), corpus generation has no neural network involved. Each worker is fully independent — it spawns its own Stockfish subprocess, plays games, and writes its own shards. No queues, no shared state, no IPC beyond initial work assignment.

```
Main process (orchestrator)
    │
    ├─ Worker 0 ── Stockfish ── plays 5,000 games ── writes shard_0_*.npz
    ├─ Worker 1 ── Stockfish ── plays 5,000 games ── writes shard_1_*.npz
    ├─ ...
    └─ Worker 9 ── Stockfish ── plays 5,000 games ── writes shard_9_*.npz
```

### 6.2 Per-game flow

For each game:
1. Initialize `board = chess.Board()`
2. **Random first 4 plies** (diversity) — sample uniformly from `board.legal_moves`, push to board
3. **Stockfish plays both sides for the rest of the game.** At each ply:
   - Call `engine.play(board, chess.engine.Limit(time=0.05))`
   - Record `(board.fen(), move, current_player)` into game's position log
   - Push the move
4. When game ends (mate / draw / 75-move rule):
   - Compute `outcome z ∈ {-1, 0, +1}` from the final result
   - For each recorded position: encode board via `Chess.encode(board)`, encode move via `move_to_index(board, move)`, assign z from mover's POV (re-uses `_assign_z` logic)
   - Append `(encoded_state, move_idx, z)` to worker's shard buffer
5. When shard buffer reaches 10,000 positions, flush to disk as `shard_W_S.npz` (W = worker id, S = shard counter)

### 6.3 Worker count and Stockfish configuration

**10 workers** on M4 Pro (10 P-cores + 4 E-cores). All P-cores fully utilized; E-cores stay free for OS responsiveness. Each Stockfish subprocess is configured with:

```python
engine.configure({
    "Threads": 1,    # critical — prevents thread oversubscription across workers
    "Hash": 16,      # 16MB transposition table, plenty for depth ~10-12
})
```

Without `Threads=1`, each Stockfish instance would spawn multiple internal threads → 10 workers × 4-8 threads each → ~50-80 threads contending for 14 cores → severe slowdown.

### 6.4 Shard format

Each shard is a numpy `.npz` archive:

```python
np.savez_compressed(
    shard_path,
    states=np.array(states, dtype=np.int8),       # (N, 20, 8, 8) — encoded boards
    move_indices=np.array(moves, dtype=np.int32), # (N,) — flat action indices [0, 4672)
    outcomes=np.array(zs, dtype=np.int8),         # (N,) — z values in {-1, 0, +1}
)
```

`int8` for states and outcomes keeps shards small (~13MB each compressed). Cast to `float32` at training-batch-load time.

### 6.5 Targets and budget

- **50,000 total games** → ~3M positions → ~300 shards → ~3.8GB on disk
- **Wall-clock: ~4.2 hours** on M4 Pro with 10 workers
- **CLI:** `python scripts/generate_chess_corpus.py --num-games 50000 --num-workers 10 --output data/chess_corpus/`

## 7. Phase 2: Supervised pre-training

### 7.1 Loss formulation

Identical to the existing `_train_step`: cross-entropy on policy + MSE on value, equal weight.

```python
policy_loss = F.cross_entropy(predicted_logits, target_move_indices)
value_loss = F.mse_loss(predicted_values, target_outcomes)
total_loss = policy_loss + value_loss
```

The only difference vs AlphaZero training is the source of the targets:
- **AZ self-play:** `target_policy = MCTS visit distribution π`, `target_value = game outcome z`
- **Pre-training:** `target_policy = one-hot(Stockfish's move)`, `target_value = game outcome z`

Same loss function, same checkpoint format, same network class.

### 7.2 Held-out validation + early stopping

To detect overfitting, split corpus shards 95/5: train on 95%, validate on 5% (the last ~15 shards). After each epoch:

1. Run forward pass on all validation positions (no gradient updates)
2. Compute `val_loss = policy_loss + value_loss` averaged over validation set
3. If `val_loss` improved → save checkpoint, reset patience counter
4. If `val_loss` did not improve → increment patience counter
5. If patience counter ≥ 2 → stop training (early stopping)

Maximum epochs: 3. Patience: 2. Typical behavior: train for 2-3 epochs, val_loss flattens, stop.

### 7.3 Learning rate schedule: linear warmup + cosine decay

For a 15M-parameter network, naive lr=1e-3 risks first-step parameter explosion (Adam's running gradient stats are unreliable at step 0). The schedule:

```python
def lr_schedule(step, warmup_steps=500, total_steps=N, peak_lr=5e-4, end_lr=5e-5):
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return end_lr + 0.5 * (peak_lr - end_lr) * (1 + math.cos(math.pi * progress))
```

- Linear warmup over the first 500 steps: lr rises from 0 → 5e-4
- After warmup: cosine decay from 5e-4 → 5e-5 over the remaining steps
- `total_steps = num_epochs × (corpus_size // batch_size)` ≈ 3 × 12,000 = 36,000 for our setup

### 7.4 Hyperparameters (`configs/chess-pretrain.toml`)

```toml
# Network — matches what chess-refine.toml will load
n_blocks = 15
n_channels = 192

# Pre-training
num_epochs = 3                # maximum; early stopping may halt sooner
batch_size = 256
peak_lr = 5e-4
end_lr = 5e-5
warmup_steps = 500
weight_decay = 1e-4
holdout_fraction = 0.05
early_stopping_patience = 2

# Misc
seed = 42
device = "auto"
corpus_dir = "data/chess_corpus"
output_checkpoint = "pretrained.pt"
log_dir = "runs_pretrain"
```

### 7.5 New CLI subcommand

```bash
python -m alphazero pretrain --config configs/chess-pretrain.toml
```

The subcommand reads `corpus_dir` from the config, splits shards into train/val, runs the pre-training loop, writes `pretrained.pt` at the path specified by `output_checkpoint`.

`pretrained.pt` is in the **exact same format** as Trainer's `iter_NNNN.pt` checkpoints (keys: `iteration`, `config`, `best_net`, `candidate_net`, `optimizer`). This lets Phase 3 load it via the existing `--resume-from` flag with zero changes to Trainer.

### 7.6 Wall-clock estimate

- ~3M positions × 3 epochs × 1 batch of 256 / batch_size = ~36K SGD steps
- ~10ms per step on M4 Pro MPS for 15M-param net at batch 256 → ~6 minutes per epoch
- 3 epochs × 6 min = ~18 minutes of pure training, plus ~10 minutes for validation eval at each epoch
- Plus disk I/O for loading shards: ~30 seconds per epoch (3.8GB streamed)
- **Total: ~30 minutes**

Note: this is much faster than the initial ~2 hour estimate because the corpus fits in disk cache after the first epoch and SGD on MPS is fast for this network size. Will refine the estimate after the first run.

## 8. Phase 3: AlphaZero refinement

### 8.1 No code changes — just config

Phase 3 is **the existing AlphaZero loop** with two changes:

1. Start from `pretrained.pt` (via `--resume-from`) instead of random initialization
2. Use reduced hyperparameters: 10 iterations, 50 MCTS sims per move

### 8.2 Hyperparameters (`configs/chess-refine.toml`)

```toml
# Network — must match pretrained.pt
n_blocks = 15
n_channels = 192

# MCTS — reduced from chess.toml
num_simulations = 50          # down from 200; pre-trained priors compensate
c_puct = 2.5
dirichlet_alpha = 0.3
dirichlet_weight = 0.25

# Self-play
games_per_iteration = 100
temperature_threshold = 30

# Training
training_steps_per_iteration = 2000
batch_size = 256
learning_rate = 1e-3
weight_decay = 1e-4

# Replay buffer — smaller since we don't need a huge ramp from cold start
replay_buffer_capacity = 200_000
min_buffer_size = 10_000

# Eval (vs Stockfish ELO=1500)
eval_interval = 2             # eval every 2 iterations
eval_games = 100

# Schedule — reduced from 80 iters
num_iterations = 10

# Parallel self-play
num_workers = 10              # match P-core count
inference_batch_size = 64

# Misc
seed = 42
device = "auto"
checkpoint_dir = "checkpoints_refine"
log_dir = "runs_refine"
```

### 8.3 Running it

```bash
python -m alphazero train \
  --game chess \
  --config configs/chess-refine.toml \
  --resume-from pretrained.pt
```

### 8.4 Wall-clock estimate

10 iterations × ~1 hour per iter (reduced sims + 15M-param net) = ~10 hours on M4 Pro. Or ~2 hours on Colab L4.

## 9. End-to-end wall-clock

| Phase | M4 Pro local | Colab L4 (refinement only) |
|---|---|---|
| 1. Corpus generation | ~4.2 hr | — |
| 2. Supervised pre-training | ~0.5 hr | — |
| 3. AZ refinement | ~10 hr | ~2 hr |
| **Total** | **~14.7 hr** | **~6.7 hr** |

Either path fits comfortably in a day or two of wall-clock time, vs the 13-day projection for cold-start AZ.

## 10. Success criteria

1. **Corpus generation** completes 50K games without worker crashes; produces ~300 shards totaling ~3.8GB
2. **Pre-training** converges: validation loss strictly decreases for at least 1 epoch; final policy accuracy on held-out set ≥ 30% (top-1 move match with Stockfish) — a reasonable baseline given Stockfish has ~30+ plausible moves per position
3. **Pre-trained checkpoint** is loadable by Trainer via `--resume-from`; first refinement iteration completes without crashes
4. **Final agent (after 10 refinement iters)** wins ≥ 50% of 100 games vs Stockfish ELO=1500 (alternating colors)

## 11. Testing strategy

### 11.1 Unit tests (`tests/test_chess_corpus.py`, `tests/test_supervised.py`)

- Corpus shard format: load a shard, verify shapes/dtypes match spec
- Corpus generation: run a tiny corpus job (50 games, 2 workers), verify shard files appear
- `load_corpus`: returns train/val split correctly, holdout_fraction respected
- `pretrain_supervised`: runs 1 epoch on synthetic data without crashing, val_loss is computed
- Early stopping: synthetic increasing val_loss triggers stop at expected step
- Lr schedule: warmup ramp + cosine decay produce expected values at key step counts
- CLI: `pretrain` subcommand wires correctly (regression test pattern from SP3 task 16)

### 11.2 Integration tests

- E2E sanity (`tests/test_e2e_supervised.py`, marked `@pytest.mark.slow`): generate 100 games, pre-train 1 epoch, run 1 AZ iter, confirm pipeline produces a checkpoint
- The actual kill criterion is verified manually after the full ~15-hour pipeline run

### 11.3 Invariants worth checking

- **Corpus encoding invariant:** for any `(encoded_state, move_idx, z)` in a shard, `Chess.legal_actions_mask(decoded_state)[move_idx] == True` — the recorded move is always legal in the recorded state
- **Outcome invariant:** for any game's positions, the recorded z values follow the alternating sign pattern `_assign_z` produces (z of first-player-to-move ≠ z of second-player-to-move within the same game, unless draw)
- **Shard count invariant:** total positions across all shards ≈ num_games × avg_plies_per_game (sanity check that no shards were dropped)

## 12. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Stockfish at 0.05s plays at lower ELO than expected (~2000 instead of 2400), capping student strength | Medium | Medium | Test corpus quality by manually evaluating a few games before full run. Bump time_per_move to 0.1s if needed (doubles wall-clock) |
| Behavior cloning ceiling causes student to plateau before clearing 1500 ELO bar | Low-Medium | High | AZ refinement should push past cloning ceiling. If not, increase `num_iterations` to 20+ or add more corpus games |
| 15M-param net overfits 3M positions in 3 epochs | Low | Medium | Early stopping with val loss catches this. Reduce to 2 epochs if val loss diverges in epoch 2 |
| MPS out-of-memory at batch 256 with 15M params | Low | Medium | Drop batch to 128 if OOM. Will slow training ~2× but still fits in budget |
| Shard format incompatibility (int8 cast issues) | Low | Low | Unit test loads a shard and verifies round-trip integrity |
| Random-first-4-plies still produces too many duplicate game lines | Low | Medium | If observed, increase to first 6 random plies; or sample first move from a fixed opening book |
| Corpus generation worker crashes mid-run, losing partial shards | Low | Low | Each worker writes its own shards independently; restart just the failed worker(s) with adjusted game count |

## 13. Future work (deferred)

- **Multi-PV soft policy targets** — richer signal but slower corpus gen. Worth trying if v1 plateaus too low.
- **Lichess corpus** — millions of real human games at varied ELO. Would replace or augment Stockfish-generated corpus. SP5.
- **Distillation** — train a smaller student to mimic a larger teacher network's policy + value output. SP5.
- **Continuous pre-training during AZ** — periodically retrain on a fresh Stockfish corpus during refinement. Stabilizes against catastrophic forgetting in long runs.
