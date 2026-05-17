# Supervised Pre-training Deep Dive

> **Companion to:** [Foundational concepts](./2026-05-16-foundational-concepts.html) (AlphaZero basics — ply, MCTS, ResNet, self-play). This doc focuses on the SP3b-specific concepts that aren't covered there: supervised pre-training, the corpus pipeline, streaming data loaders, learning-rate schedules, overfitting + early stopping, parallelism, and how the pretrained checkpoint plugs into AlphaZero refinement.

This is the conceptual reference for [Sub-project 3b — Supervised Pre-training + AZ Refinement](../superpowers/specs/2026-05-16-supervised-pretraining-design.html). Use it when you want to understand *why* the SP3b code does what it does, not just *what* it does.

---

## 1. Why pre-training at all

Pure AlphaZero is **wall-clock infeasible on consumer hardware for chess.** When SP3 ran on an M4 Pro, the first iteration took 3.85 hours. Eighty iterations would be ~13 days of straight compute. That projection is fundamental: pure self-play from a random network is inefficient because the network's policy starts uniform-over-legal-moves, MCTS has nothing meaningful to bias its search with, games hit the 75-move rule constantly (random pieces never deliver checkmate), and the resulting `(state, π, z)` tuples are weak training signal.

The fix is to **bootstrap the network from a teacher** before unleashing the AZ loop. This shifts the problem from "discover chess strategy from nothing" to "improve on a teacher who already knows chess." Total wall-clock drops from ~13 days to ~15 hours.

The teacher in our case is Stockfish — we record what it plays in Stockfish-vs-Stockfish games, then train our network to imitate those moves and predict those outcomes. After ~3M positions of imitation, we hand off to AlphaZero self-play to *refine past the teacher* via MCTS exploration.

### The SP3b pipeline at a glance

<div style="background:#fffaf0; border:2px solid #2a3a8a; border-radius:18px; padding:14px 18px; margin:18px 0;">
<svg viewBox="0 0 760 280" preserveAspectRatio="xMidYMid meet" style="width:100%; height:auto; display:block; font-family:'Patrick Hand', system-ui, sans-serif;">
  <style>
    .sp3b-box { fill:#ffffff; stroke:#2a3a8a; stroke-width:2.2; rx:10; }
    .sp3b-box-blue { fill:rgba(58,85,196,0.10); stroke:#3a55c4; stroke-width:2.2; rx:10; }
    .sp3b-box-orange { fill:rgba(238,138,48,0.10); stroke:#d6741e; stroke-width:2.2; rx:10; }
    .sp3b-box-green { fill:rgba(58,168,96,0.10); stroke:#3aa860; stroke-width:2.2; rx:10; }
    .sp3b-title { font-size:18px; font-weight:700; fill:#2a3f9a; }
    .sp3b-sub   { font-size:12px; fill:#4a5680; }
    .sp3b-tag   { font-size:10px; fill:#6b7290; }
    .sp3b-num   { font-size:22px; font-weight:700; fill:#ee8a30; }
    .sp3b-arrow { stroke:#2a3a8a; stroke-width:2.2; fill:none; stroke-linecap:round; }
    .sp3b-arrowhead { fill:#2a3a8a; }
    .sp3b-shard { fill:#3aa860; opacity:0; animation: sp3b-shard-pop 4.2s linear infinite; }
    .sp3b-shard-batch { fill:#3a55c4; opacity:0; animation: sp3b-batch-flow 4.2s linear infinite; }
    .sp3b-net-pulse { fill:none; stroke:#d6741e; stroke-width:2; opacity:0; animation: sp3b-net-pulse 4.2s ease-in-out infinite; }
    .sp3b-game { fill:#d6741e; opacity:0; animation: sp3b-game-spawn 4.2s linear infinite; }
    .sp3b-stockfish { fill:#3a55c4; stroke:#2a3f9a; stroke-width:1.5; }
    @keyframes sp3b-shard-pop {
      0%, 5%   { opacity: 0; transform: translateX(0); }
      10%      { opacity: 1; transform: translateX(0); }
      30%      { opacity: 1; transform: translateX(0); }
      100%     { opacity: 0; transform: translateX(0); }
    }
    @keyframes sp3b-batch-flow {
      0%, 30%  { opacity: 0; }
      35%      { opacity: 1; }
      55%      { opacity: 1; }
      100%     { opacity: 0; }
    }
    @keyframes sp3b-net-pulse {
      0%, 35%  { opacity: 0; r: 30; }
      45%      { opacity: 0.7; r: 38; }
      55%      { opacity: 0; r: 30; }
      100%     { opacity: 0; r: 30; }
    }
    @keyframes sp3b-game-spawn {
      0%, 60%  { opacity: 0; }
      70%      { opacity: 1; }
      90%      { opacity: 1; }
      100%     { opacity: 0; }
    }
  </style>

  <!-- Phase 1: Corpus generation -->
  <rect x="20" y="50" width="220" height="180" class="sp3b-box-blue"/>
  <text x="130" y="74" text-anchor="middle" class="sp3b-num">1</text>
  <text x="130" y="92" text-anchor="middle" class="sp3b-title">Corpus generation</text>
  <text x="130" y="108" text-anchor="middle" class="sp3b-sub">10 workers · Stockfish vs Stockfish</text>
  <!-- 10 stockfish dots -->
  <g>
    <circle cx="45" cy="135" r="6" class="sp3b-stockfish"/>
    <circle cx="65" cy="135" r="6" class="sp3b-stockfish"/>
    <circle cx="85" cy="135" r="6" class="sp3b-stockfish"/>
    <circle cx="105" cy="135" r="6" class="sp3b-stockfish"/>
    <circle cx="125" cy="135" r="6" class="sp3b-stockfish"/>
    <circle cx="145" cy="135" r="6" class="sp3b-stockfish"/>
    <circle cx="165" cy="135" r="6" class="sp3b-stockfish"/>
    <circle cx="185" cy="135" r="6" class="sp3b-stockfish"/>
    <circle cx="205" cy="135" r="6" class="sp3b-stockfish"/>
    <circle cx="225" cy="135" r="6" class="sp3b-stockfish"/>
  </g>
  <text x="130" y="158" text-anchor="middle" class="sp3b-sub">writes shards →</text>
  <!-- Shard outputs (animated) -->
  <g>
    <rect class="sp3b-shard" x="40"  y="180" width="14" height="14" rx="2" style="animation-delay:0s"/>
    <rect class="sp3b-shard" x="60"  y="180" width="14" height="14" rx="2" style="animation-delay:0.15s"/>
    <rect class="sp3b-shard" x="80"  y="180" width="14" height="14" rx="2" style="animation-delay:0.3s"/>
    <rect class="sp3b-shard" x="100" y="180" width="14" height="14" rx="2" style="animation-delay:0.45s"/>
    <rect class="sp3b-shard" x="120" y="180" width="14" height="14" rx="2" style="animation-delay:0.6s"/>
    <rect class="sp3b-shard" x="140" y="180" width="14" height="14" rx="2" style="animation-delay:0.75s"/>
    <rect class="sp3b-shard" x="160" y="180" width="14" height="14" rx="2" style="animation-delay:0.9s"/>
    <rect class="sp3b-shard" x="180" y="180" width="14" height="14" rx="2" style="animation-delay:1.05s"/>
    <rect class="sp3b-shard" x="200" y="180" width="14" height="14" rx="2" style="animation-delay:1.2s"/>
    <rect class="sp3b-shard" x="220" y="180" width="14" height="14" rx="2" style="animation-delay:1.35s"/>
  </g>
  <text x="130" y="218" text-anchor="middle" class="sp3b-tag">~3M positions · ~60 MB compressed · ~5 hr</text>

  <!-- Arrow 1 → 2 -->
  <path d="M 244 140 L 274 140" class="sp3b-arrow"/>
  <polygon points="270,135 280,140 270,145" class="sp3b-arrowhead"/>

  <!-- Phase 2: Pre-training -->
  <rect x="284" y="50" width="220" height="180" class="sp3b-box-orange"/>
  <text x="394" y="74" text-anchor="middle" class="sp3b-num">2</text>
  <text x="394" y="92" text-anchor="middle" class="sp3b-title">Supervised pre-training</text>
  <text x="394" y="108" text-anchor="middle" class="sp3b-sub">15M-param network · 3 epochs max</text>

  <!-- Batches flowing in -->
  <g>
    <rect class="sp3b-shard-batch" x="298" y="140" width="10" height="10" rx="1.5" style="animation-delay:0s"/>
    <rect class="sp3b-shard-batch" x="314" y="140" width="10" height="10" rx="1.5" style="animation-delay:0.3s"/>
    <rect class="sp3b-shard-batch" x="330" y="140" width="10" height="10" rx="1.5" style="animation-delay:0.6s"/>
    <rect class="sp3b-shard-batch" x="346" y="140" width="10" height="10" rx="1.5" style="animation-delay:0.9s"/>
  </g>

  <!-- Network (a brain-like icon) -->
  <circle cx="394" cy="170" r="30" fill="#ffffff" stroke="#d6741e" stroke-width="2.2"/>
  <circle class="sp3b-net-pulse" cx="394" cy="170"/>
  <text x="394" y="166" text-anchor="middle" class="sp3b-sub" style="font-weight:700; fill:#d6741e;">NN</text>
  <text x="394" y="180" text-anchor="middle" class="sp3b-tag" style="fill:#d6741e;">15M params</text>

  <!-- CE + MSE labels -->
  <text x="430" y="148" class="sp3b-tag" style="fill:#3a55c4;">CE(policy)</text>
  <text x="430" y="160" class="sp3b-tag" style="fill:#3aa860;">+ MSE(value)</text>

  <text x="394" y="218" text-anchor="middle" class="sp3b-tag">lr warmup + cosine · early stop · ~30 min</text>

  <!-- Arrow 2 → 3 -->
  <path d="M 508 140 L 538 140" class="sp3b-arrow"/>
  <polygon points="534,135 544,140 534,145" class="sp3b-arrowhead"/>

  <!-- Phase 3: AZ refinement -->
  <rect x="548" y="50" width="200" height="180" class="sp3b-box-green"/>
  <text x="648" y="74" text-anchor="middle" class="sp3b-num">3</text>
  <text x="648" y="92" text-anchor="middle" class="sp3b-title">AZ refinement</text>
  <text x="648" y="108" text-anchor="middle" class="sp3b-sub">MCTS + self-play · 10 iters</text>

  <!-- Self-play games appearing -->
  <g>
    <circle class="sp3b-game" cx="580" cy="160" r="6" style="animation-delay:0s"/>
    <circle class="sp3b-game" cx="600" cy="150" r="6" style="animation-delay:0.2s"/>
    <circle class="sp3b-game" cx="625" cy="170" r="6" style="animation-delay:0.4s"/>
    <circle class="sp3b-game" cx="650" cy="155" r="6" style="animation-delay:0.6s"/>
    <circle class="sp3b-game" cx="675" cy="170" r="6" style="animation-delay:0.8s"/>
    <circle class="sp3b-game" cx="700" cy="155" r="6" style="animation-delay:1.0s"/>
    <circle class="sp3b-game" cx="720" cy="170" r="6" style="animation-delay:1.2s"/>
  </g>

  <text x="648" y="200" text-anchor="middle" class="sp3b-sub">→ checkpoints_refine/iter_NNNN.pt</text>
  <text x="648" y="218" text-anchor="middle" class="sp3b-tag">50 sims · 100 games · ~10 hr</text>

  <!-- "pretrained.pt" arrow label -->
  <text x="259" y="132" text-anchor="middle" class="sp3b-tag" style="fill:#2a3f9a;">shards</text>
  <text x="523" y="132" text-anchor="middle" class="sp3b-tag" style="fill:#2a3f9a;">pretrained.pt</text>

  <!-- Total wall-clock -->
  <text x="380" y="265" text-anchor="middle" class="sp3b-title" style="font-size:14px; fill:#3aa860;">Total: ~15 hours</text>
  <text x="380" y="20" text-anchor="middle" class="sp3b-tag" style="font-size:11px;">(vs ~13 days for cold-start AlphaZero)</text>
</svg>
</div>

The diagram is the whole sub-project in one frame: ten Stockfish workers write shards in parallel, the pretrain step streams those shards into a 15M-param network optimizing CE + MSE, and the resulting `pretrained.pt` becomes the starting point for AlphaZero refinement self-play. Everything that follows in this doc is the *why* behind each box.

---

## 2. Behavior cloning, formally

**Behavior cloning** is the simplest form of imitation learning. Given a corpus of `(state, action)` pairs from a teacher policy π_T, you train a student network π_S to minimize:

```
Loss = Σ over (s, a) in corpus:  -log π_S(a | s)
```

That's just cross-entropy with the teacher's action as the target one-hot. The student doesn't see rewards, doesn't search, doesn't plan — it just learns the conditional distribution `student(action | state) ≈ teacher(action | state)`.

Two crucial extensions for chess:

1. **Outcome regression.** We also have the game result `z ∈ {-1, 0, +1}` for every recorded state. We add a value head trained with MSE on `(state, z)`. This gives the network a position-evaluation prior — useful for MCTS later.

2. **Per-position POV normalization.** The `z` is recorded from each position's *mover's POV*, not white's POV. In a decisive game, adjacent plies have z values of opposite sign (winner's positions all +1, loser's all -1). In draws, every z is 0.

Combined loss:
```
total_loss = cross_entropy(predicted_policy, teacher_move_index) + mse(predicted_value, z)
```

This is **the exact same loss** that AlphaZero uses during self-play — only the source of the targets differs:

| Phase | target_policy | target_value |
|---|---|---|
| AZ self-play | MCTS visit distribution π | game outcome z |
| **Pre-training** | **one-hot(Stockfish's move)** | **game outcome z** |

Because the loss is identical, no code change in `_train_step` is needed. Only the data loader changes.

### The teacher ceiling

Behavior cloning has a structural limit: **the student can never exceed the teacher's level by imitation alone.** Empirically, students typically lose 100-300 ELO vs their teacher because:

- Limited model capacity vs the teacher's search depth
- Some teacher moves rely on deep tactical search the student can't reproduce from a single forward pass
- The student averages over many "Stockfish-like" moves and doesn't perfectly fit any one

So if Stockfish-at-0.05s plays at ~ELO 2400, the cloned student plays maybe ~2200. That's still ~700 ELO above our kill criterion (Stockfish-1500), which is fine.

**Why AZ refinement can exceed the teacher.** Once the student has reasonable priors, MCTS-on-the-student explores positions the teacher never reached and finds moves the teacher wouldn't play. The improvement signal comes from MCTS sharpening the network's own policy via visit counts, not from any external teacher. Empirically, ~10 refinement iters from a strong-pretrained start gain +200 to +400 ELO.

---

## 3. The corpus: Stockfish vs Stockfish

The teacher games are generated by `scripts/generate_chess_corpus.py`. The flow per game:

1. **Random opening (4 plies).** Without this, Stockfish-vs-Stockfish produces ~50 unique opening lines repeated thousands of times — useless diversity. Random first 4 plies give us ~`20⁴ ≈ 160K` distinct opening sequences from `chess.Board()`.

2. **Stockfish plays both sides** with `time_per_move=0.05s` until the game ends.

3. **Record every position** as `(encoded_state, move_played_idx, mover_player)`. We don't know the outcome until the game ends, so we accumulate this in memory.

4. **At game end, compute outcome** via python-chess's `board.outcome(claim_draw=True)`. The `claim_draw=True` is important — it makes the 50-move rule and 3-fold repetition terminate the game. Without it, games would only stop on checkmate, stalemate, insufficient material, or the unconditional 75-move/150-halfmove rule.

5. **Backfill z** for each recorded position from the final outcome × `mover_player`. A winner's positions all get +1, loser's all -1, draws all 0.

### Why Stockfish at 0.05s, not full strength

At 0.05s/move, Stockfish's effective ELO drops to ~2400 from its full ~3500. Three reasons we prefer it over full strength:

- **Throughput.** At 0.05s/move × ~80 plies/game × 50,000 games / 10 workers ≈ 5 hours wall-clock. Full strength (say 2s/move) would be ~40 hours.
- **Diversity.** Faster play means more varied moves — Stockfish explores more lines without time to converge on one "best" move. Better for student generalization.
- **Sufficient.** ELO 2400 is *vastly* above the SP3 kill criterion (1500). We don't need a stronger teacher than necessary.

### Draws are expected and OK

Stockfish-vs-Stockfish at fast time controls produces **~60-70% draws.** This is normal — both sides play accurately enough that ambiguous middle-game positions coast into 50-move-rule endgames. Our typical corpus has roughly:

```
17% wins, 66% draws, 17% losses
```

That 33% decisive rate is *enough* training signal for the value head — at 3M total positions, that's ~1M positions with z = ±1. The policy head benefits from all 3M positions regardless of outcome (Stockfish's chosen move is the target whether the game ended decisively or drew).

---

## 4. Storage: int8 shards on disk

The corpus is stored as a series of `.npz` shard files, **10,000 positions per shard, written incrementally** by each worker as its in-memory buffer fills.

Each shard contains three numpy arrays:

```
states       : int8   (10000, 20, 8, 8)  ← encoded board planes
move_indices : int32  (10000,)            ← flat action indices [0, 4672)
outcomes     : int8   (10000,)            ← z values in {-1, 0, +1}
```

### Why int8 and not float32?

The state encoding is nearly all 0s and 1s (piece presence, castling rights, etc.). Stored as `float32`, a single shard would be `10000 × 20 × 64 × 4 = 5.1 MB`. As `int8`: `10000 × 20 × 64 × 1 = 1.3 MB`. Compressed via `np.savez_compressed`, it shrinks further to ~190 KB per shard.

For 300 shards (full corpus): **~60 MB on disk**, vs ~1.5 GB if we'd stored as float32. The runtime cost is one cast `int8 → float32` per batch when loading — negligible compared to forward+backward time.

### Why 10,000 positions per shard?

A trade-off:
- **Smaller shards** → more file-system overhead, more `np.load` calls, but finer-grained progress reporting.
- **Bigger shards** → fewer disk operations but more memory per shard load.

10,000 strikes a balance: each shard is ~190KB compressed (fits easily in OS page cache), each worker accumulates one shard's worth of positions in ~170 games (~20 minutes per shard), and the training data loader can process one shard at a time with ~13 MB peak in-memory after the int8→float32 expansion.

### Why incremental writes?

If we accumulated *all* positions per worker in RAM and wrote one big file at the end, a worker crash would lose everything. By flushing every 10,000 positions, we lose at most the current buffer (~20 minutes of work) on failure. Plus, the user gets visible progress (shard files appearing) instead of a 5-hour silent void.

---

## 5. Streaming data: `iter_batches`

The pre-training loop never loads the whole corpus into memory. Instead, `iter_batches` is a **generator** that yields batches one at a time, reading shards on demand. The contract:

```python
def iter_batches(
    shards: list[Path],
    batch_size: int,
    device: torch.device,
    shuffle: bool,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    for shard_path in shards:
        data = np.load(shard_path)                            # ① one shard at a time
        n = data["states"].shape[0]
        order = torch.randperm(n).numpy() if shuffle else np.arange(n)  # ② per-shard shuffle
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            s = torch.from_numpy(data["states"][idx]).to(torch.float32).to(device)
            m = torch.from_numpy(data["move_indices"][idx]).to(torch.int64).to(device)
            z = torch.from_numpy(data["outcomes"][idx]).to(torch.float32).to(device)
            yield s, m, z                                     # ③ hand to training loop
```

Three key design choices:

**(1) Stream, don't load.** Peak memory stays bounded by one shard's worth (~13 MB float32) regardless of corpus size. Critical for systems where the full corpus wouldn't fit in RAM.

**(2) Per-shard shuffle, not global.** A true global shuffle across 300 shards would require either loading them all (defeats streaming) or doing complex multi-pass disk reads. Per-shard shuffle ensures **within** a shard's 10,000 positions, the order is randomized. The caller can also shuffle the **list of shards** between epochs for additional diversity. For 3M positions across 3 epochs, the network sees enough variety.

**(3) dtype casts at load time.** We saved `int8`/`int32`/`int8` on disk to minimize storage. At load time, we cast to the dtypes PyTorch expects: `float32` for state features, `int64` for class indices (cross-entropy expects int64), `float32` for value targets (MSE expects float).

### What "Iterator" means here

`iter_batches` is a Python **generator function** — it has `yield` instead of `return`. When you call `iter_batches(...)`, you don't get a list of batches; you get a generator object. Each `next()` call (implicit in a `for` loop) executes the function until the next `yield`, then suspends. This is what makes streaming work: the function doesn't materialize all batches up front.

---

## 6. Learning rate schedule: warmup + cosine decay

Pre-training uses a non-constant learning rate. For each training step:

```
lr(step) = peak_lr × (step + 1) / warmup_steps                            if step < warmup_steps
         = end_lr + 0.5 × (peak_lr - end_lr) × (1 + cos(π × progress))   otherwise
```

where `progress = (step - warmup_steps) / (total_steps - warmup_steps)`.

Visualized:

```
lr
5e-4 |       _____
     |      /     \___
     |     /          \___
     |    /               \___
     |   /                    \___
5e-5 |__/_________________________\____
     |
     +-warmup-|------- cosine decay ------> step
       (500)         (~35,500 steps)
```

### Why warmup

The Adam optimizer (and AdamW we use) maintains running estimates of the gradient's first and second moments. At step 0, these estimates are based on a single batch — wildly noisy. If we use the full `peak_lr` immediately, the very first update can produce a parameter step that's huge in magnitude but completely random in direction. That undoes the careful weight initialization (Xavier/He) and can make the loss explode in the first ~200 steps.

Warmup solves this by ramping the lr from 0 to `peak_lr` over the first `warmup_steps` (we use 500). During warmup, Adam's running stats accumulate; by step 500, they're reliable enough to handle the peak lr safely.

### Why cosine decay

After warmup, the network is rapidly fitting the bulk of the training data. We want big steps early to escape low-information regions of the loss landscape, then progressively smaller steps as we approach a good minimum (to fine-tune without bouncing around).

Cosine decay smoothly drops from `peak_lr` to `end_lr` following `0.5 × (1 + cos(πt))`. Simple, parameter-free, well-studied. Alternatives (linear, step, exponential) work fine too — cosine is just the most common modern choice.

### Why bigger nets want smaller lr

Rule of thumb from practice: `peak_lr ∝ 1 / √num_params`. So:
- 1M params  → peak_lr ≈ 2e-3
- **15M params → peak_lr ≈ 5e-4** (our pretrain config)
- 50M params → peak_lr ≈ 3e-4

Reason: bigger nets have noisier per-batch gradients (more parameters being estimated). High lr × noisy direction = thrashing rather than descent. Smaller lr keeps updates conservative enough that noise averages out across batches.

---

## 7. Overfitting and early stopping

A 15M-parameter network has way more capacity than necessary to memorize 3M positions. Left unchecked, it'll fit the training data so tightly that it stops generalizing — it learns "in this exact FEN, the move is e2-e4" instead of "in positions like this, develop the knight."

We detect this with a **held-out validation set** — 5% of shards reserved at corpus load time, never trained on:

```python
all_shards = sorted(shard_dir.glob("shard_*.npz"))
n_val = max(1, int(len(all_shards) * 0.05))   # ~15 shards from ~300
train_shards = all_shards[:-n_val]            # 95% for training
val_shards   = all_shards[-n_val:]            # 5% for validation
```

After each epoch, compute the loss on val_shards (forward-only, no gradient updates). If it's lower than the previous best, save a checkpoint and reset patience. If it's higher (or equal), increment patience. When patience reaches `early_stopping_patience` (we use 2), stop training entirely.

### The overfitting curve

```
loss
 |  \                       ← train loss keeps dropping (memorization)
 |   \
 |    \________
 |             \____________
 |
 |       _____
 |  ____/     \___           ← val loss bottoms out, then RISES
 |              \____        ← that's overfitting kicking in
 |                  \_______
 |              ↑
 |              stop here
 +------------------------> epoch
```

Training loss keeps falling forever — the network has enough parameters to memorize anything you throw at it. Validation loss shows the **true** quality: how well the network generalizes to unseen positions. When val starts rising, the network is no longer learning; it's memorizing. We stop and use the best-so-far checkpoint.

With our config (3 epochs max, patience 2), the typical pattern is: train loss drops 3 epochs in a row, val loss drops 2 epochs then plateaus, early stopping fires on epoch 3 if val doesn't improve. Or all 3 epochs run cleanly if the network never overfits at this corpus size.

### Why patience=2

Validation loss has noise — a single epoch where it ticks up slightly might be measurement noise, not real overfitting. We require **two consecutive non-improvements** before declaring overfitting and stopping. Lower patience (1) stops too eagerly; higher patience (5+) wastes compute on memorization.

---

## 8. Parallelism: processes, not threads

The corpus generator spawns 10 worker **processes** (not threads). The orchestrator uses `multiprocessing.get_context("spawn")` to create them. Each worker:

- Owns its own Stockfish subprocess (configured `Threads=1`)
- Owns its own RNG (seeded with `base_seed + worker_id`)
- Plays games independently — no IPC with other workers
- Writes its own shards directly to disk (independent filenames)

Why processes and not threads?

### Threads (Python) are limited by the GIL

The CPython interpreter has a Global Interpreter Lock that allows only one thread to execute Python bytecode at a time per process. This is fine for I/O-bound tasks where the GIL is released during waits (file reads, network calls). But our worker does Python-side work between Stockfish calls — encoding states, indexing moves, tracking buffers — and that work IS bytecode execution.

10 threads in one process would serialize this Python work through the GIL, getting ~no parallelism benefit on the Python side. Each thread's Stockfish subprocess would still parallelize (since subprocesses are independent OS processes), but the per-ply ~5-10ms of Python work per worker becomes a contention point.

Processes have no GIL issue: each is a fresh Python interpreter with its own memory space. Workers truly run in parallel up to physical core count.

### The "embarrassingly parallel" pattern

Corpus generation has **zero inter-worker communication** — workers don't share state, don't synchronize on locks, don't merge results. They each play games, each write shards, each exit. The orchestrator just `spawn`s them and `join`s them.

This is the textbook "embarrassingly parallel" pattern. The N-fold speedup is nearly linear up to physical core count. On a 10-physical-core machine, 10 workers achieve close to 10× the throughput of 1 worker. (Hyperthreaded virtual cores give diminishing returns because hyperthreads on the same physical core share execution units.)

### Why we still use threads in `parallel_selfplay.py`

The chess SP3 `parallel_selfplay.py` (for AlphaZero self-play) **does** use a thread for the NN-server. That's because the NN-server needs to share GPU memory with the model — moving the model between processes is expensive (full state-dict copy) and fragile (spawn-context model loading has known PyTorch issues with MPS especially). The server thread releases the GIL during `net(x)` (PyTorch tensors are C++ under the hood), so multiple worker processes can submit requests concurrently without thread contention.

So:
- **CPU-bound Python work + no shared GPU** → processes (corpus gen)
- **GPU work + shared model state** → thread inside the main process (NN-server in parallel_selfplay)
- **CPU-bound Python work + needs GPU access** → worker processes that send requests to a thread that owns the GPU (the full parallel_selfplay architecture)

---

## 9. The checkpoint contract

After pre-training completes, the `pretrained.pt` file has the **exact same shape** as the `iter_NNNN.pt` files that AlphaZero training produces:

```python
{
    "iteration":    0,                      # AZ refinement starts counting at iter 1
    "config":       <dataclass-as-dict>,    # informational only
    "best_net":     <state_dict>,           # ← this is what AZ refinement loads
    "candidate_net": <state_dict>,          # same as best_net (copy)
    "optimizer":    <optimizer state>,      # AdamW state from end of pretraining
    "_pretrain_epoch": N,                   # informational only
}
```

This format compatibility is **non-negotiable** — it's what lets `Trainer.load_from_checkpoint(pretrained.pt)` work without modification. The Trainer doesn't know or care whether the weights came from pre-training or from a prior AZ run; it just reads `ckpt["best_net"]` and proceeds.

### Why `candidate_net = copy(best_net)`

`Trainer` maintains two copies of the network: `best_net` (used to generate self-play games) and `candidate_net` (the one being trained). At the start of each iteration, gradient updates happen on `candidate_net`; periodically, `candidate_net`'s weights are copied to `best_net`. By initializing both to the pretrained weights, the first AZ refinement iteration plays self-play with a strong (pretrained) network and trains on top of it.

### Why `iteration=0`

AZ refinement starts at iter 1. If we saved `iteration=N` for some N > 0, the refinement loop would think it had already done N iterations and start at iter N+1. We set 0 so refinement counts from the beginning.

### The integration test that protects this contract

`tests/test_supervised.py::test_pretrained_checkpoint_loads_in_trainer_via_resume` literally runs pretrain → save → `Trainer.load_from_checkpoint` → verify weights match. If anyone changes the checkpoint format on either side and breaks compatibility, this test fails. It's a regression guard for the entire bridge between Phase 2 and Phase 3.

---

## 10. Sizing: params vs corpus size

How big should the network be? Two competing pressures:

**Bigger nets clone better.** Behavior cloning quality scales with capacity — a bigger network has more "room" to fit the teacher's policy distribution. Stockfish makes nuanced positional decisions; representing those requires capacity. A 1M-param net hits a quality ceiling around ELO 1800-2000 even with a perfect teacher. A 15M-param net can reach ~2400-2700. Larger is better.

**Smaller nets are faster to forward-pass.** MCTS during AZ refinement runs `num_simulations` × `games_per_iteration` × `~80 plies` ≈ 400K forward passes per iteration. A 4× bigger network means 4× more refinement wall-clock. There's a real trade-off.

### Rule of thumb: 10-20 positions per parameter

Empirically, behavior cloning works well when:

```
positions_in_corpus ≈ 10 × num_parameters
```

For our 15M-param net, that means ~150M positions ideally. We have ~3M positions, ~50× fewer. We're in "data-limited" regime, not "capacity-limited."

What does this mean in practice?
- The network can't fully memorize the corpus (good — prevents overfitting)
- The network will average over many similar positions, learning robust patterns
- The behavior cloning loss won't drop to near-zero; it'll plateau at maybe 1-3
- The student will fall further behind the teacher than at the ideal ratio (maybe ELO 2100 student from a 2400 teacher, vs 2300 student at full ratio)

This is *acceptable* for our use case because AZ refinement closes much of that gap. We don't need a near-perfect imitator — we need a starting point materially stronger than random.

### Why not just collect more games?

Corpus generation takes ~5 hours on 10 physical cores. Going from 50K games (3M positions) to 500K games (30M positions) would take ~50 hours and bring us closer to the ideal ratio. But:

- 30M positions still isn't 150M; we'd still be data-limited
- The marginal gain per additional position diminishes
- Diminishing returns past ~2-3× the current corpus size

If you wanted to push harder, the high-leverage move is **better data** (stronger teacher at `time_per_move=0.2` or higher) rather than **more data** (more games at the same time control).

---

## 11. Phase-by-phase: the whole pipeline

How everything fits together:

### Phase 1 — Corpus generation
```bash
python3 scripts/generate_chess_corpus.py --num-games 50000 --num-workers 10 --output data/chess_corpus/
```

Spawns 10 workers, each running Stockfish-vs-Stockfish games independently. Each worker writes its own shard files as buffers fill. Wall-clock: ~5-9 hours depending on machine. Output: ~300 shards totaling ~3M positions in `data/chess_corpus/`.

### Phase 2 — Supervised pre-training
```bash
python3 -m alphazero pretrain --config configs/chess-pretrain.toml
```

Loads shards, streams batches, trains a 15M-param network for up to 3 epochs with held-out validation and early stopping. Wall-clock: ~30 min on M4 Pro MPS, ~2-4 hours on CPU. Output: `pretrained.pt` (Trainer-compatible checkpoint).

### Phase 3 — AlphaZero refinement
```bash
python3 -m alphazero train --game chess --config configs/chess-refine.toml --resume-from pretrained.pt
```

Runs the existing AZ loop (now in `parallel_selfplay.py` with worker processes + NN-server thread) starting from the pretrained weights. Reduced parameters: 10 iterations × 100 games × 50 MCTS sims (vs SP3's 80/100/200). Wall-clock: ~10 hours. Output: `checkpoints_refine/iter_NNNN.pt` for each iteration; best is the final one.

### Evaluation
```bash
python3 -m alphazero eval --game chess --checkpoint checkpoints_refine/iter_0010.pt --num-games 100
```

Plays 100 games vs Stockfish at ELO=1500. Pass criterion: `win_rate >= 0.50`.

---

## 12. Glossary (SP3b-specific terms)

For AlphaZero basics (ply, MCTS, ResNet, PUCT, etc.), see the [foundational concepts glossary](./2026-05-16-foundational-concepts.html#glossary).

**Behavior cloning** — Supervised learning where the target action is a teacher policy's chosen action. Trains the student to imitate the teacher.

**Cosine decay** — Learning-rate schedule that smoothly drops from `peak_lr` to `end_lr` following `0.5 × (1 + cos(πt))`. Used after warmup.

**Early stopping** — Halt training when validation loss stops improving for `patience` consecutive epochs. Prevents overfitting; saves compute.

**Embarrassingly parallel** — A workload where workers don't communicate or share state. Linear speedup with worker count up to physical-core limit.

**Hold-out validation set** — Fraction of training data reserved from the optimizer — used only to measure generalization. We use 5% of corpus shards.

**`iter_batches`** — Python generator that streams batches from disk shards, casting dtypes and moving tensors to the target device. Bounded memory regardless of corpus size.

**Linear warmup** — Ramp learning rate linearly from 0 to `peak_lr` over the first `warmup_steps` steps. Lets Adam's running stats stabilize before applying full lr.

**NNUE** — "Efficiently Updatable Neural Network." Stockfish's built-in evaluation function (a small dense network, ~50M params, AVX2-optimized). Unrelated to our PyTorch AlphaZeroNet — Stockfish uses NNUE internally for its own search.

**Patience** — Counter for early stopping. Resets when validation loss improves; increments otherwise. Stop when patience ≥ `early_stopping_patience`.

**`pretrained.pt`** — Checkpoint produced by Phase 2 (supervised pre-training). Format-compatible with `Trainer.load_from_checkpoint` so Phase 3 can resume from it.

**Random opening** — Push N random legal moves before Stockfish takes over, giving game diversity. We use N=4.

**Shard** — One `.npz` file holding 10,000 encoded positions. Workers write incrementally so progress is visible.

**Spawn context** — `multiprocessing.get_context("spawn")` — the multiprocessing flavor that creates fresh Python processes (vs `fork`, which copies memory). Safer with CUDA, mandatory on Windows, recommended on macOS.

**Stockfish UCI** — Universal Chess Interface, the text-based protocol Stockfish speaks. `python-chess` wraps it. Configurable via `Threads`, `Hash`, `UCI_LimitStrength`, `UCI_Elo`.

**Teacher ceiling** — The structural limit on a student trained via behavior cloning: typically 100-300 ELO below the teacher. Why AZ refinement matters — it lets us exceed this bound.

**Throughput** — Positions processed per second by the training loop. Determines wall-clock for a fixed corpus size and epoch count.

**`time_per_move`** — Wall-clock budget given to Stockfish for each move during corpus generation. We use 0.05s. Lower → weaker Stockfish + more diverse games; higher → stronger Stockfish + fewer games for the same wall-clock budget.

**Warmup steps** — Number of initial training steps over which lr ramps from 0 to `peak_lr`. We use 500.

---

## 13. Further reading

- **The implementation:** [SP3b design spec](../superpowers/specs/2026-05-16-supervised-pretraining-design.html) and [implementation plan](../superpowers/plans/2026-05-16-supervised-pretraining.html)
- **The framework basics:** [Foundational concepts](./2026-05-16-foundational-concepts.html)
- **The visual primer for AlphaZero:** [Concepts page](../../concepts.html)
- **Self-test:** [Interactive quiz](./2026-05-16-alphazero-quiz.html)
- **The original AlphaGo paper** (supervised pre-training, then RL refinement): [Mastering the game of Go with deep neural networks and tree search](https://www.nature.com/articles/nature16961) (Silver et al., 2016)
- **AlphaZero paper** (skipped supervised pretraining; pure self-play): [A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play](https://www.science.org/doi/10.1126/science.aar6404) (Silver et al., 2018)
