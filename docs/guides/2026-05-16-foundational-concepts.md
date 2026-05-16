---
title: Foundational Concepts — AlphaZero from Scratch
date: 2026-05-16
kicker: Learning notes
---

# Foundational Concepts

A consolidated reference for the AlphaZero project — companion to the codebase and the visual primer at [`concepts.html`](../../concepts.html). Designed to be re-read whenever a term feels fuzzy.

This is the **text-heavy reference**; `concepts.html` is the visual companion. Both cover overlapping ground. Use whichever fits the moment.

---

## 1. The big picture in one paragraph

AlphaZero is the marriage of two ML traditions that, individually, cannot solve hard games:

- **Neural networks** give *fast intuition* ("this position looks bad for me") but cannot plan deeply.
- **Tree search** gives *deep planning* but combinatorially explodes — you cannot search all possible chess games (~10⁴³ positions).

AlphaZero combines them. The neural network gives the search a starting hunch (which moves to explore, how good each position is). The search refines that hunch by looking ahead a few moves. The refined judgment becomes the network's new training data. Both improve together; neither can do it alone.

---

## 2. The training loop in one diagram

```
                                          ┌──────────────────────┐
                                          │   best_net (frozen)   │
                                          └───────────┬──────────┘
                                                      │ used by
                                                      ▼
       ┌──────────────────────────────────────────────────────────┐
       │   1. SELF-PLAY                                            │
       │      100 games using MCTS + best_net                      │
       │      Each ply → (encoded_state, π, current_player)        │
       │      End of game → fill in z = ±1/0 per ply               │
       └──────────────────────┬───────────────────────────────────┘
                              │
                              ▼
       ┌──────────────────────────────────────────────────────────┐
       │   2. ADD TO REPLAY BUFFER (FIFO of recent tuples)         │
       └──────────────────────┬───────────────────────────────────┘
                              │
                              ▼
       ┌──────────────────────────────────────────────────────────┐
       │   3. TRAIN candidate_net                                  │
       │      1000 minibatches × 64 samples                       │
       │      loss = CE(policy, π) + MSE(value, z) + L2 reg        │
       └──────────────────────┬───────────────────────────────────┘
                              │
                              ▼
       ┌──────────────────────────────────────────────────────────┐
       │   4. PROMOTE  best_net ← candidate_net  (unconditional)   │
       └──────────────────────┬───────────────────────────────────┘
                              │
                              ▼
       ┌──────────────────────────────────────────────────────────┐
       │   5. (Periodic) EVAL vs ground-truth opponent             │
       └──────────────────────┬───────────────────────────────────┘
                              │
                              ▼
                       (loop back to step 1)
```

Each iteration's best_net is slightly stronger than the previous one. After 30-50 iterations on TTT, the agent reaches perfect play. After thousands of iterations on chess (with massive compute), the agent reaches superhuman.

---

## 3. Basic terminology

### Ply
A single move by one player. A chess full-move is 2 plies (white + black). TTT games are 5-9 plies. Connect 4 games average 20-30 plies.

### State (s)
The current board position. In our code, a NumPy array (`(3, 3)` for TTT, `(6, 7)` for Connect 4).

### Action (a)
A legal move from a state. Represented as an integer index — 0-8 for TTT (one per square), 0-6 for Connect 4 (one per column).

### Policy
A probability distribution over actions. Two flavors:
- **`P(s, a)`** (capital P) — the neural network's prior policy (one forward pass)
- **`π(s, a)`** (lowercase pi) — the MCTS-improved policy (after N simulations)

These are related but distinct. PUCT uses `P` to guide search; training pushes the NN to predict `π`.

### Value
A scalar in `[-1, +1]` indicating the predicted/actual game outcome from the current player's POV.
- **`v(s)`** — the neural network's predicted value (one forward pass)
- **`z`** — the actual realized outcome at the end of a game

### Episode / game
One complete play from the initial state to a terminal state. Produces a batch of training tuples (one per ply) sharing the same outcome `z`.

### Iteration
One full cycle of the training loop: self-play → buffer → training → promote → eval. NOT to be confused with an "epoch" (one pass through a dataset) in supervised learning.

---

## 4. Neural network basics

### What is a "neural network" in this project?

A function `f: state → (policy, value)` parameterized by weights. The weights are adjusted by gradient descent so that `f`'s predictions match the training targets `(π, z)`.

In our code, this is `AlphaZeroNet` in `src/alphazero/network.py`. It's a `torch.nn.Module` with two output heads.

### Conv layer (convolution)

A convolution slides a small **kernel** (weight matrix) across the input, computing a weighted sum at each spatial position. For a 3×3 kernel:

```
For each output position (x, y):
    output[c_out, x, y] = Σ_{c_in, dx, dy} W[c_out, c_in, dx, dy] · input[c_in, x+dx, y+dy]
```

Each "channel" of the input is processed separately; the kernel combines them with learned weights. Convolutions are great for grid-structured data (images, game boards) because they exploit spatial locality.

### 1×1 conv specifically

Kernel size of 1×1 — doesn't combine neighboring pixels at all. Just remixes channels at each spatial position. It's a per-pixel linear transformation.

Why?
- **Channel reduction**: backbone has 64 channels; we don't need that many for the policy head. A 1×1 conv collapses 64 → 2 channels.
- **Cheap**: fewer parameters than a 3×3 conv.

### Batch normalization (BN)

Normalizes activations across the batch dimension. For each channel:

```
mean_c, var_c = mean / variance across (batch, spatial)
output = (input - mean_c) / sqrt(var_c + ε) · γ_c + β_c
```

Where `γ` and `β` are learnable. BN keeps activations in a sane numeric range during training, stabilizes gradient flow, and lets you use higher learning rates. **Almost universal in modern CNNs.**

**Subtle quirk**: BN behaves differently in training vs eval mode. During training it uses batch statistics; during eval it uses running statistics accumulated during training. Always call `model.eval()` before inference and `model.train()` before training. Forgetting this is the source of many training bugs.

### ReLU (Rectified Linear Unit)

The simplest nonlinearity: `f(x) = max(0, x)`. Anything negative becomes 0; anything positive passes through. The nonlinear part is what lets neural networks learn complex functions — stacking linear layers without nonlinearities just gives you another linear function.

ReLU is the default activation because it's simple, fast, and avoids the vanishing-gradient problem of older activations like sigmoid/tanh.

### Flatten

Reshapes a multi-dimensional tensor to 2D, preserving the batch dimension. `(4, 2, 6, 7)` becomes `(4, 84)`. Needed because the next layer (Linear) expects 1D input per sample.

### Linear layer

Fully connected layer: `output = input @ W + b`. Maps `input_dim` features to `output_dim` features. Used at the end of each head to produce the final outputs.

### Logits and softmax

**Logits** are the raw pre-softmax outputs of a classification network. They can be any real number, positive or negative. To convert to probabilities, apply softmax:

```
prob[i] = exp(logit[i]) / Σ_j exp(logit[j])
```

We work in logits internally because they're more numerically stable (log-sum-exp trick avoids underflow on tiny probabilities). Loss functions like `cross_entropy(logits, target)` operate on logits directly.

### Backbone vs heads

In a multi-head network:
- **Backbone**: the shared feature extractor — everything before the branches
- **Heads**: small sub-networks at the end, each producing one output

In AlphaZeroNet:
```
input → backbone (input conv + N residual blocks) → features
                                                      ├── policy head
                                                      └── value head
```

The backbone is the big computation; the heads are tiny linear-conv-linear stacks.

---

## 5. Residual networks (ResNet)

### What is a residual block?

A composable layer module that computes:

```
y = ReLU(x + F(x))

Where F(x) is typically:
    conv → BatchNorm → ReLU → conv → BatchNorm
```

The `+x` is the **skip connection** (a.k.a. residual connection). It's the entire trick — without it, deep networks are hard to train.

### Why the skip connection matters

In a deep network, the gradient of the loss with respect to early-layer weights is the product of the Jacobian of every later layer. If each layer's Jacobian has magnitude `<1`, the product shrinks toward zero — gradients **vanish**, early layers stop learning.

With a skip connection:
```
y = F(x) + x
∂y/∂x = ∂F/∂x + 1
                ↑
        the "1" is the skip's contribution
```

That "1" guarantees gradient gets through, even when `∂F/∂x` is tiny. This is why ResNets can train hundreds of layers deep; previous architectures plateaued at maybe 20.

### "N residual blocks of K channels"

A common architecture sizing notation:
- **N**: how many blocks are stacked in sequence (depth)
- **K**: how many channels each block's intermediate features have (width)

For TTT: 4 blocks × 32 channels (~76K params). For Connect 4: 6 blocks × 64 channels (~600K params). For original AlphaZero on chess: 19 blocks × 256 channels (~46M params).

Tradeoff:
- More blocks (depth): hierarchical compositional reasoning, slower per forward pass
- More channels (width): more pattern variety per spatial position, quadratic in memory

### Forward pass

Given a board state, run it through the network to produce a policy and value:

```python
def forward(self, x):
    h = F.relu(self.input_bn(self.input_conv(x)))    # input conv → BN → ReLU
    h = self.blocks(h)                                # N residual blocks
    
    # Policy head
    p = F.relu(self.policy_bn(self.policy_conv(h)))
    policy_logits = self.policy_fc(p.flatten(start_dim=1))
    
    # Value head
    v = F.relu(self.value_bn(self.value_conv(h)))
    v = F.relu(self.value_fc1(v.flatten(start_dim=1)))
    value = torch.tanh(self.value_fc2(v)).squeeze(-1)
    
    return policy_logits, value
```

That's it. The same code runs on TTT, Connect 4, and chess — only the input shape and action_size differ.

### Backward pass (backpropagation)

How the network *learns*. Given a training tuple `(state, target_policy_π, target_value_z)`:

1. **Forward** through the net → predicted policy logits + predicted value
2. **Compute loss**:
   - Policy loss: cross-entropy between predicted distribution and `π`. Penalizes the net for putting probability mass on moves MCTS didn't visit.
   - Value loss: MSE between predicted value and `z`. Penalizes the net for thinking a position is good when it actually led to a loss.
   - Total = policy_loss + value_loss + L2 regularization (via weight decay in AdamW)
3. **Backward** (`loss.backward()`): PyTorch walks the computation graph in reverse, applying the chain rule to compute `∂loss/∂w` for every parameter. The skip connections keep these gradients meaningful 20+ layers deep.
4. **Optimizer step** (`optimizer.step()`): `w ← w - learning_rate · gradient` (AdamW also does per-param adaptive scaling).

After thousands of these steps, the network's predictions get sharper.

---

## 6. Monte Carlo Tree Search (MCTS)

### The four phases (per simulation)

For one MCTS simulation, starting at the current position:

**1. SELECT** — walk down the existing tree, picking the most promising child via PUCT until you hit a leaf (an unexpanded node).

**2. EXPAND** — call the NN: `priors, value = eval_fn(state)`. Create child nodes for each legal action, each with its NN-assigned prior.

**3. (Implicit "simulate")** — classical MCTS does random rollouts here. AlphaZero replaces them with the NN's value head output. The leaf's value is the NN's prediction.

**4. BACKUP** — walk back up the path. At each node: increment visit count `N`, add the leaf value to total `W`, flip sign (zero-sum game).

After 50-100 such simulations, the root's children have visit counts. Normalize to get `π = visits / total`. That's MCTS's "improved policy".

### PUCT — the selection rule

For each candidate action `a` from node `s`:

```
PUCT(s, a) = Q(s, a) + c_puct · P(s, a) · √(Σ_b N(s, b)) / (1 + N(s, a))
            └─exploit─┘   └────────────exploration──────────────┘

   Q(s, a) = mean value seen for action a so far
   P(s, a) = NN's prior probability for action a
   N(s, a) = visit count for action a
   c_puct  = exploration constant (1.5-2.5 typically)
```

MCTS picks `argmax_a PUCT(s, a)` at every step of every simulation. Two effects play out:
- **Q grows** for promising actions (exploitation)
- **U shrinks** for already-explored actions (since N appears in denominator)

The NN's prior `P` biases exploration toward moves it thinks are good — that's what makes AlphaZero so much more efficient than classical MCTS (which used uniform priors).

### Why MCTS sharpens the NN's hunch

This is the central insight, and it's subtle.

Suppose the NN's raw policy at the start of a Connect 4 game gives:
```
column:    0      1      2      3 (center)   4      5      6
prior:    0.10   0.12   0.15   0.30          0.15   0.12   0.06
```
Center is highest but only marginally. The NN isn't confident.

Now MCTS runs 100 simulations. Each one descends the tree guided by PUCT, evaluating leaf positions via the value head. Simulations through center tend to discover good positions; simulations through column 6 tend to discover bad positions.

After 100 sims, visit counts might be:
```
column:    0    1    2    3    4    5    6
visits:    2    3    8    78   5    3    1
π:         0.02 0.03 0.08 0.78 0.05 0.03 0.01
```

MCTS-improved `π` puts 78% on center — much sharper than the raw NN's 30%. **This sharpened π is what the NN trains on next iteration.** The NN learns to predict the sharper distribution. Next iteration its raw prior is already closer to 78% on center, so MCTS's exploration is even better targeted. Both improve.

### Dirichlet noise (root only, self-play only)

**Dirichlet distribution** is a probability distribution *over probability distributions*. A draw from `Dir(α)` is itself a vector summing to 1.

```python
np.random.dirichlet([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
# Returns something like:
# [0.05, 0.13, 0.31, 0.08, 0.12, 0.07, 0.24]   (sums to 1.0)
```

Parameter `α` controls concentration:
- Large α (e.g., 1.0): noise is near-uniform
- Small α (e.g., 0.3): noise is "spiky" — concentrated on a few actions

**How we apply it**: during self-play only, when MCTS expands the root, we mix the NN's prior `P` with a Dirichlet draw:

```
P_new[a] = (1 - ε) · P[a] + ε · Dir(α)[a]
```

With `ε = 0.25` and `α` set per-game (1.0 for TTT, 0.5 for Connect 4, 0.3 for chess).

**Why we need it**: MCTS with a fixed network and fixed starting state is *deterministic*. Without noise, every self-play game would explore the same tree → same π → similar games → narrow training data. Dirichlet noise perturbs the root once per game, fanning out the exploration.

**Why only at the root**: deeper noise would just be random distraction without purpose. The root is special because we want diverse game *openings*.

**Why only during self-play**: arena/eval should be deterministic so we get an accurate read on model strength.

### Temperature schedule

Controls how the MCTS visit distribution is converted into the actual move played.

| τ value | Behavior |
|---|---|
| **τ = 1** | Sample from π proportionally — exploration. Used early in the game. |
| **τ = 0** | Argmax — always pick the most-visited action. Used later. |

The switch happens at `temperature_threshold` plies:
- TTT: 6 plies (basically the whole game)
- Connect 4: 15 plies (cover the opening)
- Chess: ~30 plies

**Why**: with argmax for the whole game, every self-play game would reach the same positions. Sampling for the first N plies creates diverse openings; argmax later ensures the engame is played optimally.

Combined with Dirichlet noise at the root, the two mechanisms produce game-to-game diversity.

---

## 7. Self-play

### What it is

Two copies of the agent play each other. Each ply produces one training tuple:

```
(s, π, z)
   ↑   ↑   ↑
   │   │   eventual game outcome from this player's POV (+1, 0, -1)
   │   MCTS-improved policy at this position
   encoded canonical board state
```

A Connect 4 game produces ~25 plies → ~25 tuples per game. Self-play of 100 games per iteration → ~2500 tuples per iteration (×2 with symmetry augmentation).

### Why bootstrap works

This is the magic. At iteration 0, both agents are random networks. Games are won/lost by luck. **But MCTS, even with a random NN, sees a few moves ahead** — winning positions get found via tree expansion + the value head's (slightly-biased) output.

Over iterations:
- Value head learns "this kind of position leads to a win"
- Policy head learns "this is the move MCTS typically prefers"
- MCTS gets sharper as priors improve
- Self-play games get higher quality
- → better training data → better network → repeat

The agent climbs from random play to optimal play by **always training against targets that are slightly better than itself** (because MCTS sharpens the NN's own predictions).

### Symmetry augmentation

Many games have positions that are equivalent under some transformation (rotation, reflection). If π says "70% on center" for the original board, the same applies to all rotations/reflections.

| Game | # symmetries | Why |
|---|---|---|
| TTT | 8 | D₄ dihedral group: 4 rotations × 2 reflections (no constraint on the board) |
| Connect 4 | 2 | Identity + horizontal mirror only (gravity prevents vertical symmetry) |
| Chess | 1 | No symmetries — castling, pawn direction, en-passant all break them |

Augmentation = take each (state, π) and produce all equivalent transformed versions. For TTT, one ply becomes 8 training tuples. Same compute cost (we already ran MCTS); more training data; better generalization.

In our code: `Connect4.symmetries(encoded, policy)` returns `[(identity, π), (mirrored, mirrored_π)]`.

### Per-ply z-assignment

After a game ends, we know the final outcome (X won, O won, or draw). For each recorded ply, we assign `z` from THAT PLY'S MOVER'S POV:

- X-won game: plies where X moved get `z=+1`; plies where O moved get `z=-1`
- O-won game: vice versa
- Draw: all plies get `z=0`

This is **the most bug-prone part of the framework**. Getting the sign wrong silently corrupts the value head's training signal. Our `tests/test_selfplay.py` has hand-traced tests for the three cases (X wins, O wins, draw) to lock the behavior down.

---

## 8. The interlocking loop

### Why the combination is special

Three things make AlphaZero's combination uniquely powerful, none of which work alone:

**A. Search compresses NN noise.**
A single NN forward pass is noisy. Running 50-100 forward passes along different futures and aggregating produces a much more reliable judgment. MCTS is essentially a Monte Carlo *estimator* of which move is best, weighted by visit counts.

**B. The training target is *better than the model***.
In normal supervised learning, you train against labels at least as good as your model (e.g., human chess games). In AlphaZero, you train against MCTS-improved policies that are **strictly better than the raw NN they came from** — because MCTS used the NN as an oracle 50-100 times and aggregated. The NN is being told "be more like your own future-self-with-thinking-time."

**C. Self-play creates an opponent that scales with you.**
If you trained on fixed human games, you'd plateau at human level. Self-play creates an opponent exactly as strong as you are right now — neither too weak (no challenge) nor too strong (you can't learn from them). You're always playing at the edge of your competence, which is the optimal training signal.

### Why training data is "better than the model"

A single NN forward pass at position `s` gives a noisy `P(s, ·)` and `v(s)`. MCTS runs 50-100 of those forwards plus aggregation, giving:
- A much sharper `π(s, ·)` (visit distribution)
- A more reliable value estimate (the backed-up tree value)

The sharper π is the training target for the policy head — training pushes the NN to *be* what its own MCTS-augmented self computes. The actual game outcome `z` is the training target for the value head — training pushes the NN to predict eventual outcomes more accurately than its own current value estimates.

Both targets are slightly better than the current model. Train, repeat.

---

## 9. How to trace the code (to fully understand it)

A concrete reading path. ~4 hours of focused work to "fully understand."

### Phase 1: The big picture (15 min)
- README.md
- This document
- The 5 concept cards on `concepts.html`

Build the mental model before opening code.

### Phase 2: Trace ONE iteration end-to-end (30-45 min)

Top-down call graph:

```
cli.py:_cmd_train()              ← entry point
    └─ trainer.py:Trainer.run()  ← the iteration loop
        ├─ _run_self_play_iteration()
        │   └─ selfplay.py:run_one_game()
        │       ├─ mcts.py:MCTS.search()
        │       │   └─ MCTS._simulate() — select/expand/backup
        │       │       └─ _expand() — calls eval_fn
        │       └─ _assign_z()
        ├─ _train_step()
        │   └─ AlphaZeroNet.forward() + loss.backward()
        ├─ _promote_candidate_to_best()
        ├─ _eval_vs_opponent() (every 5 iters)
        └─ _save_checkpoint()
```

After this you know what data flows where.

### Phase 3: Read supporting modules (20-30 min)

The smaller pieces, top-to-bottom:
- `games/base.py` — abstract `Game` interface
- `games/tictactoe.py` — simplest concrete game
- `games/connect4.py` — same interface, different game
- `replay_buffer.py` — trivial FIFO
- `arena.py` — head-to-head match runner
- `solvers/tictactoe_solver.py` — perfect minimax
- `opponents/connect4_minimax.py` — α-β minimax

### Phase 4: Read tests as executable spec (30 min)

Tests are **executable specifications**. Each test shows expected behavior more concretely than any docstring.

For each component you traced, read its test file:
- `tests/test_game_tictactoe.py` (25 tests showing exact behavior)
- `tests/test_mcts.py` (12 tests with hand-computed expectations)
- `tests/test_selfplay.py` (z-assignment tests are illuminating)
- `tests/test_invariants.py` (6 invariant guards)

If you read `tests/test_mcts.py:test_backup_flips_sign_per_ply`, you'll see exactly how MCTS treats values — the test is more concrete than any comment.

### Phase 5: Trace at runtime (the gold standard, 30+ min)

Open a Python REPL or Jupyter cell and run a tiny example:

```python
from alphazero.games.tictactoe import TicTacToe
from alphazero.network import AlphaZeroNet
from alphazero.mcts import MCTS
import numpy as np
import torch

ttt = TicTacToe()
net = AlphaZeroNet(input_shape=(3, 3, 3), action_size=9, n_blocks=1, n_channels=4)
net.eval()

def eval_fn(encoded):
    with torch.no_grad():
        x = torch.from_numpy(encoded).float().unsqueeze(0)
        logits, value = net(x)
        priors = torch.softmax(logits, dim=-1).squeeze(0).numpy()
    return priors.astype(np.float32), float(value.item())

state = ttt.initial_state()
mcts = MCTS(ttt, eval_fn)
pi = mcts.search(state, num_simulations=20, add_root_noise=False)
print("MCTS visit distribution:", pi)
```

Run with `num_simulations=20` then `num_simulations=200`. Compare. You'll *feel* what MCTS does.

Use `import pdb; pdb.set_trace()` inside `MCTS._simulate()` to step through select/expand/backup one line at a time.

### Useful tracing tricks

1. **Track tensor shapes** mentally as you read forward functions. Write the shape at each layer.
2. **Run individual tests** with `pytest tests/.../test_name.py::test_specific -v -s` and add prints.
3. **`git log --oneline` + `git blame`** — commit messages explain *why* a piece exists.
4. **Read specs alongside code**: `docs/superpowers/specs/2026-05-15-alphazero-framework-tictactoe-design.html` for design rationale.

---

## 10. Glossary

| Term | One-line definition |
|---|---|
| **Action (a)** | A legal move from a state, represented as an integer index |
| **Activation function** | Nonlinear function applied between layers (we use ReLU) |
| **AdamW** | Adam optimizer with decoupled weight decay; converges faster than vanilla SGD |
| **Backbone** | Shared feature extractor of a multi-head network |
| **BatchNorm (BN)** | Normalizes activations across the batch dimension; stabilizes training |
| **Backup** | Final phase of MCTS simulation: walk up path, update visit counts, flip sign |
| **Backpropagation** | Algorithm to compute gradients via the chain rule, walking the computation graph in reverse |
| **Backward pass** | One step of computing gradients with respect to the loss |
| **Best net** | The frozen network used to generate self-play training data |
| **Candidate net** | The network being actively trained (becomes the new best after each iteration) |
| **Canonical state** | Board rewritten so current player's pieces are +1; lets NN never learn whose-turn-is-it |
| **Channel** | One feature map produced by a conv layer; "32 channels" = 32 features per spatial position |
| **Checkpoint** | Saved state of the trainer (best_net, candidate_net, optimizer, iteration) |
| **Conv (convolution)** | A learnable filter that slides across the input computing weighted sums |
| **1×1 conv** | A conv that only remixes channels per-pixel (no spatial mixing); used for channel reduction |
| **Cross-entropy** | Loss function measuring distance between predicted and target probability distributions |
| **c_puct** | Exploration constant in PUCT (~1.5-2.5) |
| **Dirichlet noise** | Random perturbation added to MCTS root priors during self-play for game-to-game diversity |
| **Epoch** | (Term from supervised learning) one pass through the dataset; we use "iteration" instead |
| **eval_fn** | Function `state → (priors, value)`; the boundary between MCTS and the NN |
| **Expand** | Phase of MCTS: call eval_fn at a leaf, create child nodes for each legal action |
| **Filter** | Synonym for "convolution kernel"; one filter produces one output channel |
| **Flatten** | Reshape multi-dimensional tensor → 1D (preserving batch dim) |
| **Forward pass** | Running a state through the network to produce predictions |
| **Game** | (a) The abstract Game class, (b) A single episode of self-play |
| **Head** | A small sub-network producing one output (policy head, value head) |
| **Inference** | Using a trained model to make predictions (no gradient updates) |
| **Iteration** | One cycle: self-play → buffer → train → promote → eval |
| **Kernel** | The weight matrix of a convolution; "3×3 kernel" means a 3-by-3 weight grid |
| **L2 regularization** | Penalizes large weights; helps prevent overfitting; applied via AdamW's weight_decay |
| **Leaf** | A node in the MCTS tree that hasn't been expanded yet |
| **Linear layer** | Fully connected layer: `output = input @ W + b` |
| **Logits** | Raw pre-softmax scores; can be any real number |
| **Loss** | Scalar quantifying prediction error; minimized via gradient descent |
| **MCTS** | Monte Carlo Tree Search — the search algorithm that turns the NN's hunch into a sharper decision |
| **Minibatch** | A small random subset of training data used for one gradient step (size 64 in our config) |
| **Minimax** | Classical game-tree search with alpha-beta pruning; our eval baseline for Connect 4 |
| **MSE (Mean Squared Error)** | Loss function used for the value head: `(predicted - target)²` |
| **Policy** | A probability distribution over actions |
| **Policy head** | NN sub-network outputting action logits |
| **Prior (P)** | The NN's pre-search probability for each action; used by PUCT |
| **PUCT** | Polynomial Upper Confidence applied to Trees — the MCTS selection rule |
| **Ply** | A single move by one player |
| **π (improved policy)** | MCTS visit-count distribution; the training target for the policy head |
| **Q-value** | Mean value seen for an action so far (`W / N` at a node) |
| **Replay buffer** | FIFO queue of recent training tuples (capacity 50K-200K) |
| **ReLU** | `max(0, x)` — the default nonlinearity |
| **Residual block** | `y = ReLU(x + F(x))` — the building block of ResNet |
| **ResNet** | A neural network architecture made of stacked residual blocks; trains deep networks reliably |
| **RL (Reinforcement Learning)** | ML paradigm where an agent learns by interacting with an environment and receiving rewards |
| **Self-play** | Two copies of the agent playing each other to generate training data |
| **Skip connection** | The `+x` in a residual block; lets gradients flow through deep networks |
| **Softmax** | Converts logits to a probability distribution: `exp(logit_i) / Σ_j exp(logit_j)` |
| **State (s)** | The current board position |
| **Symmetry** | A transformation that preserves the game's meaning (rotation, reflection) |
| **Temperature (τ)** | Controls how the MCTS visit distribution is converted to an action (τ=1: sample, τ=0: argmax) |
| **Tensor** | A multi-dimensional numerical array (PyTorch's equivalent of numpy ndarrays) |
| **Value head** | NN sub-network outputting a single scalar in [−1, +1] (predicted game outcome) |
| **Visit count (N)** | Number of MCTS simulations that have visited a node |
| **Weight** | A learnable parameter of the network |
| **z** | Actual realized outcome of a game from a specific ply's mover's POV (+1 / 0 / −1) |
| **Zero-sum** | Game where one player's gain is the other's loss (sum of values = 0) |

---

## Where each concept lives in the code

| Concept | File | Function/class |
|---|---|---|
| Game interface | `src/alphazero/games/base.py` | `Game` ABC |
| Tic-Tac-Toe | `src/alphazero/games/tictactoe.py` | `TicTacToe` |
| Connect 4 | `src/alphazero/games/connect4.py` | `Connect4` |
| ResNet | `src/alphazero/network.py` | `AlphaZeroNet`, `ResidualBlock` |
| Forward pass | `network.py:AlphaZeroNet.forward()` | one pass through the net |
| Backward pass | `trainer.py:Trainer._train_step()` | `loss.backward(); optimizer.step()` |
| MCTS | `src/alphazero/mcts.py` | `MCTS.search()`, `_simulate()` |
| PUCT | `mcts.py:MCTS._select_child()` | implementation of the formula |
| Self-play | `src/alphazero/selfplay.py` | `run_one_game()`, `_assign_z()` |
| Dirichlet noise | `mcts.py:MCTS._add_dirichlet_noise()` | mixed in at root expansion |
| Temperature schedule | `selfplay.py:run_one_game()` | the `if move_idx < temperature_threshold:` branch |
| Symmetries | `games/tictactoe.py`, `games/connect4.py` | `symmetries()` method on each game |
| Replay buffer | `src/alphazero/replay_buffer.py` | `ReplayBuffer` |
| Arena match runner | `src/alphazero/arena.py` | `play_match()`, `MatchResult` |
| Training loop | `src/alphazero/trainer.py` | `Trainer.run()` |
| TTT perfect solver | `src/alphazero/solvers/tictactoe_solver.py` | `solve_tictactoe_action()` |
| Connect 4 minimax | `src/alphazero/opponents/connect4_minimax.py` | `Connect4MinimaxOpponent` |
| CLI | `src/alphazero/cli.py` | `_cmd_train`, `_cmd_eval`, `_cmd_play` |

---

## Further reading

- **AlphaZero paper** (Silver et al. 2017): https://arxiv.org/abs/1712.01815
- **AlphaGo Zero paper** (Silver et al. 2017): https://www.nature.com/articles/nature24270 — earlier version with arena gating
- **alpha-zero-general** (Surag Nair): https://github.com/suragnair/alpha-zero-general — the canonical pedagogical implementation; our framework's spiritual ancestor
- **muzero-general** (Werner Duvaud): https://github.com/werner-duvaud/muzero-general — same spirit for MuZero, the successor algorithm
- **Visual primer**: [`concepts.html`](../../concepts.html) — animated SVGs of MCTS, ResNet, PUCT, replay buffer, arena
- **Sub-project 1 spec** (TTT): [`docs/superpowers/specs/2026-05-15-alphazero-framework-tictactoe-design.md`](../superpowers/specs/2026-05-15-alphazero-framework-tictactoe-design.md)
- **Sub-project 2 spec** (Connect 4): [`docs/superpowers/specs/2026-05-15-connect4-design.md`](../superpowers/specs/2026-05-15-connect4-design.md)
