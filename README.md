![python](https://img.shields.io/badge/python-%3E%3D%203.11-306998)
![pytorch](https://img.shields.io/badge/pytorch-%3E%3D%202.2-ee4c2c)
![tests](https://img.shields.io/badge/tests-118%20passing-brightgreen)
![status](https://img.shields.io/badge/sub--project%201-passes%20kill%20criterion-brightgreen)
![status](https://img.shields.io/badge/sub--project%202-impl%20complete-yellow)
![license](https://img.shields.io/badge/license-MIT-green)

# AlphaZero from Scratch

A learning-focused, [documented](https://venkata91.github.io/alphazero/) implementation of AlphaZero based on the DeepMind [paper](https://arxiv.org/abs/1712.01815) (Silver et al., 2017). Designed to be a clean, game-agnostic framework that scales unchanged from Tic-Tac-Toe to chess by changing only one file: the `Game` subclass.

The project's purpose is **learning** — building up the full AlphaZero pipeline (board representation → neural network → MCTS → self-play → training loop) from scratch, with the strict invariant that **only the `Game` class differs between games**. Everything else (network, MCTS, self-play, replay buffer, trainer) is reusable as-is.

This implementation is primarily for educational purposes — companion to a written design and decision trail at the [project site](https://venkata91.github.io/alphazero/). For a visual primer on the underlying concepts (MCTS, ResNet skip connections, PUCT, replay buffer, arena), see [`concepts.html`](https://venkata91.github.io/alphazero/concepts.html).

## Features

* [x] Pure Python + [PyTorch](https://github.com/pytorch/pytorch) — no external RL frameworks
* [x] **Game-agnostic framework**: same code trains TTT, Connect 4, chess by swapping a `Game` subclass
* [x] ResNet backbone with policy + value heads (configurable depth × channels)
* [x] PUCT-guided MCTS with Dirichlet root noise + temperature schedule
* [x] Self-play with symmetry augmentation for data efficiency
* [x] FIFO replay buffer, AdamW optimizer, training-step / iteration scheduling
* [x] Apple Silicon (MPS) + CUDA + CPU device auto-detection
* [x] **Checkpoint resume** (`--resume-from`) — survives Colab session disconnects
* [x] Self-describing checkpoints (config embedded; matches architecture on load)
* [x] **Google Colab notebook** with GPU + Google Drive persistence
* [x] Interactive `play` CLI for matches vs the trained agent
* [x] Six "silent-failure" invariant tests guarding the bug-prone parts of self-play and MCTS
* [x] [GitHub Pages site](https://venkata91.github.io/alphazero/) with full spec/plan docs
* [x] 118 unit tests + 2 kill-criterion E2E tests

### Future improvements

* [ ] Chess (Sub-project 3) — `python-chess` wrapper, scaled ResNet, ~5M params
* [ ] UCI engine protocol — play in Lichess / any chess GUI
* [ ] Distributed / parallel self-play (needed for chess scale)
* [ ] Distillation from existing strong solvers (Sub-project 5 hybrid)

## Demo

Trained Tic-Tac-Toe agent vs perfect minimax solver (200 games, alternating colors):

```
$ python -m alphazero eval --checkpoint checkpoints/iter_0050.pt --num-games 200
vs perfect solver (200 games): wins=0 draws=200 losses=0
```

TTT is a draw with perfect play. **Zero losses** = the trained agent matches optimal play as both first and second player — the formal kill criterion for Sub-project 1.

Interactive play against the trained network:

```
$ python -m alphazero play --checkpoint checkpoints/iter_0050.pt --as x

Loaded best_net from iteration 50.
You are X (moves first).
Action indices (row-major):
 0 | 1 | 2
-----------
 3 | 4 | 5
-----------
 6 | 7 | 8

   |   |
-----------
   |   |
-----------
   |   |

Your move [0, 1, 2, 3, 4, 5, 6, 7, 8]: 4
```

## Games implemented

| Game | Status | Network size | Notes |
|---|---|---|---|
| **Tic-Tac-Toe** | ✅ Passes kill criterion (0 losses vs perfect solver) | 4 blocks × 32 channels (~76K params) | Converges in ~50 iterations on CPU |
| **Connect 4** | ✅ Implementation complete, training pending | 6 blocks × 64 channels (~600K params) | Target: ≥80% win rate vs minimax-depth-8 |
| Chess | 🟡 Planned (Sub-project 3) | 10 blocks × 128 channels (~5M params) | Will wrap `python-chess` |
| Go | 🟡 Possible future extension | (TBD) | — |

Add a new game by creating one file in `src/alphazero/games/` that subclasses `Game`. Every other component of the framework is reused unchanged.

## Code structure

Four strict layers with well-defined boundaries:

```
┌────────────────────────────────────────────────────────────────┐
│  Trainer (orchestration)                                        │
│    self-play → buffer → train → eval-vs-opponent, looping       │
└──┬─────────────┬──────────────┬──────────────┬─────────────────┘
   │             │              │              │
   ▼             ▼              ▼              ▼
┌────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐
│SelfPlay│ │ Replay   │ │  Arena   │ │ Eval-vs-opponent │
│Worker  │ │  Buffer  │ │          │ │  (per game)      │
└───┬────┘ └──────────┘ └────┬─────┘ └──────────────────┘
    │                        │
    ▼                        ▼
┌────────────────┐  ┌───────────────────────────┐
│  MCTS          │  │  Game (abstract base +    │
│   PUCT, Dirich │  │  TicTacToe, Connect4)     │
└────────┬───────┘  └───────────────────────────┘
         │ calls eval_fn(state) → (priors, value)
         ▼
┌────────────────────────────────────────────────────────────────┐
│  AlphaZeroNet (PyTorch ResNet — policy + value heads)          │
│    pure forward pass; does not import MCTS or Game             │
└────────────────────────────────────────────────────────────────┘
```

The key boundary property: **`MCTS` does not import `AlphaZeroNet`, and `AlphaZeroNet` does not import `MCTS`**. They communicate through a function signature (`eval_fn`), which makes both testable in isolation and means adding a new game touches exactly one file.

For full architectural detail, see the Sub-project 1 spec: [TTT design doc](https://venkata91.github.io/alphazero/docs/superpowers/specs/2026-05-15-alphazero-framework-tictactoe-design.html).

## Getting started

### Installation

```bash
git clone https://github.com/venkata91/alphazero.git
cd alphazero
pip install -e ".[dev]"
```

Requires Python 3.11+. PyTorch is installed automatically; GPU support (CUDA / MPS) is auto-detected at runtime.

### Train

Tic-Tac-Toe (CPU, ~30-60 min):

```bash
python -m alphazero train --game tictactoe --config configs/tictactoe.toml
```

Connect 4 (MPS / CUDA recommended; ~1-6 hours depending on device):

```bash
python -m alphazero train --game connect4 --config configs/connect4.toml
```

Resume from a checkpoint:

```bash
python -m alphazero train --game connect4 --config configs/connect4.toml \
    --resume-from checkpoints/iter_0020.pt
```

### Evaluate

Play 200 games against the game's ground-truth opponent (perfect solver for TTT, minimax-depth-8 for Connect 4):

```bash
python -m alphazero eval --game tictactoe --checkpoint checkpoints/iter_0050.pt --num-games 200
python -m alphazero eval --game connect4   --checkpoint checkpoints/iter_0050.pt --num-games 200
```

### Play interactively

```bash
python -m alphazero play --game tictactoe --checkpoint checkpoints/iter_0050.pt --as x
```

For tighter search at play time (stronger play, same network):

```bash
python -m alphazero play --game tictactoe --checkpoint checkpoints/iter_0050.pt \
    --num-simulations 500 --as x
```

### Train on Google Colab (free GPU)

Open the included notebook in Colab — it clones the repo, installs deps, mounts Drive for checkpoint persistence, auto-resumes from interrupted runs, and downloads the final checkpoint locally.

1. Open https://colab.research.google.com/
2. **File → Open notebook → GitHub** → URL: `https://github.com/venkata91/alphazero/blob/main/notebooks/train_connect4_colab.ipynb`
3. **Runtime → Change runtime type → T4 GPU** (free) or L4/A100 (Pro)
4. **Runtime → Run all**

Wall-time on Colab T4: ~1-2 hours for Connect 4. See [`notebooks/README.md`](notebooks/README.md) for details.

### Config

Each game has its own TOML config in `configs/`. Edit fields like `n_blocks`, `num_simulations`, `num_iterations` to tune. The full schema lives in [`src/alphazero/config.py`](src/alphazero/config.py).

## Documentation

* **Visual concepts primer**: [`concepts.html`](https://venkata91.github.io/alphazero/concepts.html) — animated SVGs explaining MCTS, ResNet, PUCT, replay buffer, arena
* **Sub-project 1 (TTT)**: [spec](https://venkata91.github.io/alphazero/docs/superpowers/specs/2026-05-15-alphazero-framework-tictactoe-design.html) · [implementation plan](https://venkata91.github.io/alphazero/docs/superpowers/plans/2026-05-15-alphazero-framework-tictactoe.html)
* **Sub-project 2 (Connect 4)**: [spec](https://venkata91.github.io/alphazero/docs/superpowers/specs/2026-05-15-connect4-design.html) · [implementation plan](https://venkata91.github.io/alphazero/docs/superpowers/plans/2026-05-15-connect4.html)
* **Live site (all docs)**: https://venkata91.github.io/alphazero/

## What we learned (so far)

A few of the more interesting design corrections caught during implementation:

* **Arena gating stagnates training**: AlphaGo Zero (2017) gated network promotion behind a 55% win rate over 40 games. On TTT this completely stalled training because MCTS smoothed over small NN differences → most matches drew → win-rate hovered near 50% → gate never accepted → `best_net` froze at iter 5. The AlphaZero paper (2017) dropped arena gating; we now do the same. See [Sub-project 2 spec §6 "Lessons from Sub-project 1"](https://venkata91.github.io/alphazero/docs/superpowers/specs/2026-05-15-connect4-design.html).
* **MCTS sign convention is bug-prone**: the value returned by both `terminal_value` and the NN's value head is from the *leaf* player's POV. PUCT at the parent reads `child.Q` as "good for me (the parent)", so the leaf value needs negation before backup. Getting this wrong silently teaches the agent to give up as the second player.
* **Test the 6 invariants by name**: the project includes `tests/test_invariants.py` with one named test per silent-failure mode (`test_invariant_1_selfplay_uses_best_net_not_candidate`, etc.). When training stagnates, these usually catch it.

## Related work

* [AlphaZero paper](https://arxiv.org/abs/1712.01815) — Silver et al. 2017, primary reference
* [AlphaGo Zero paper](https://www.nature.com/articles/nature24270) — earlier work with arena gating
* [alpha-zero-general](https://github.com/suragnair/alpha-zero-general) — Surag Nair's well-known pedagogical AlphaZero implementation
* [muzero-general](https://github.com/werner-duvaud/muzero-general) — same spirit for MuZero (the successor algorithm without environment dynamics)
* [Leela Chess Zero](https://lczero.org/) — the open-source community AlphaZero-style chess engine

## Authors

* Venkata Krishnan Sowrirajan ([@venkata91](https://github.com/venkata91))
* Co-developed with Claude (Anthropic's coding assistant)

## Getting involved

* [GitHub Issues](https://github.com/venkata91/alphazero/issues) — bug reports, questions
* [Pull Requests](https://github.com/venkata91/alphazero/pulls) — contributions welcome
