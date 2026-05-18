![python](https://img.shields.io/badge/python-%3E%3D%203.11-306998)
![pytorch](https://img.shields.io/badge/pytorch-%3E%3D%202.2-ee4c2c)
![tests](https://img.shields.io/badge/tests-207%20passing-brightgreen)
![license](https://img.shields.io/badge/license-MIT-green)

# AlphaZero from Scratch

A clean, game-agnostic implementation of [AlphaZero](https://arxiv.org/abs/1712.01815) (Silver et al., 2017) built up from scratch in Python + PyTorch. The same framework — board representation, ResNet, PUCT-MCTS, self-play, replay buffer, trainer — scales unchanged from Tic-Tac-Toe to chess; adding a new game is one new `Game` subclass.

Built as a learning project. Each sub-project ships with a written design spec and implementation plan; full design trail is at [venkata91.github.io/alphazero](https://venkata91.github.io/alphazero/).

## Sub-projects shipped

| Sub-project | Status | Docs |
|---|---|---|
| SP1 — Framework + Tic-Tac-Toe (kill criterion: 0 losses vs perfect solver) | DONE | [spec](https://venkata91.github.io/alphazero/docs/superpowers/specs/2026-05-15-alphazero-framework-tictactoe-design.html) · [plan](https://venkata91.github.io/alphazero/docs/superpowers/plans/2026-05-15-alphazero-framework-tictactoe.html) |
| SP2 — Connect 4 (eval baseline: minimax-depth-8) | DONE | [spec](https://venkata91.github.io/alphazero/docs/superpowers/specs/2026-05-15-connect4-design.html) · [plan](https://venkata91.github.io/alphazero/docs/superpowers/plans/2026-05-15-connect4.html) |
| SP3 — Chess (8×8×73=4672 move encoding, 20-plane input, Stockfish eval, NN-server + parallel self-play workers) | DONE | [spec](https://venkata91.github.io/alphazero/docs/superpowers/specs/2026-05-16-chess-design.html) · [plan](https://venkata91.github.io/alphazero/docs/superpowers/plans/2026-05-16-chess.html) |
| SP3b — Supervised pre-training + AZ refinement (Stockfish-vs-Stockfish corpus → behavior cloning → AZ refinement) | DONE | [spec](https://venkata91.github.io/alphazero/docs/superpowers/specs/2026-05-16-supervised-pretraining-design.html) · [plan](https://venkata91.github.io/alphazero/docs/superpowers/plans/2026-05-16-supervised-pretraining.html) |
| SP4 — Productionize (UCI engine, lichess-bot, 2000+ ELO target) | PLANNED | — |
| SP5 — Hybrid search + distillation | PLANNED | — |

Full docs site: <https://venkata91.github.io/alphazero/>. Visual primer on MCTS / ResNet / PUCT / replay buffer / arena: [`concepts.html`](https://venkata91.github.io/alphazero/concepts.html).

## Setup

Requires Python 3.11+.

```bash
git clone https://github.com/venkata91/alphazero.git
cd alphazero
pip install -e ".[dev]"
```

Chess additionally needs the Stockfish binary (for the eval baseline and the supervised corpus generator). One-shot installer for macOS / Debian:

```bash
bin/setup.sh
```

GPU support (CUDA / MPS) is auto-detected at runtime.

## Quick start

Train each game's AlphaZero pipeline end-to-end:

```bash
python -m alphazero train --game tictactoe --config configs/tictactoe.toml
python -m alphazero train --game connect4  --config configs/connect4.toml
python -m alphazero train --game chess     --config configs/chess.toml
```

Chess supervised pre-training, then AZ refinement on top of the pretrained checkpoint:

```bash
python -m alphazero pretrain --config configs/chess-pretrain.toml
python -m alphazero train --game chess --config configs/chess-refine.toml \
    --resume-from pretrained.pt
```

Evaluate a checkpoint against the game's reference opponent (perfect minimax for TTT, depth-8 minimax for Connect 4, Stockfish for chess):

```bash
python -m alphazero eval --game tictactoe --checkpoint checkpoints/iter_0050.pt --num-games 200
python -m alphazero eval --game connect4  --checkpoint checkpoints/iter_0050.pt --num-games 200
python -m alphazero eval --game chess     --checkpoint checkpoints_refine/iter_0010.pt --num-games 20
```

Play interactively:

```bash
python -m alphazero play --game tictactoe --checkpoint checkpoints/iter_0050.pt --as x
python scripts/play_chess.py --checkpoint pretrained.pt --as white
```

Run as a UCI engine (drop into Lichess via [lichess-bot](https://github.com/lichess-bot-devs/lichess-bot), Arena, Cute Chess, etc.):

```bash
python scripts/uci_engine.py --checkpoint pretrained.pt --num-simulations 200
```

Resume any training run from a checkpoint with `--resume-from <path>`; checkpoints embed their config so the architecture rehydrates exactly.

## Playing against the trained model

`scripts/uci_engine.py` is a generic UCI wrapper around any AlphaZero chess checkpoint, so any UCI-compatible host can drive it. Two recommended paths:

### A) Cute Chess (local GUI) — recommended

A free desktop chess GUI that drives any UCI engine. Five-minute setup.

```bash
brew install --cask cute-chess   # on macOS
# or download from https://cutechess.com on Linux/Windows
```

Then in Cute Chess: Tools → Settings → Engines → Add (+), with:

| Field | Value |
|---|---|
| Name | `AlphaZero-SP3b` |
| Command | `/path/to/python3` (e.g., `/opt/miniconda3/bin/python3`) |
| Arguments | `/Users/<you>/git/alphazero/scripts/uci_engine.py --checkpoint /Users/<you>/git/alphazero/pretrained.pt` |
| Working directory | `/Users/<you>/git/alphazero` |
| Protocol | UCI |

Then **Game → New Game → Human vs Engine**. Drag-drop board, real piece graphics, full PGN export. Also supports engine-vs-engine matches if you want to pit `pretrained.pt` against `checkpoints_refine/iter_0010.pt`.

### B) Lichess-bot (online, public)

Wraps the same UCI engine and exposes it as a Lichess BOT account. Anyone on Lichess can challenge it; rating calibrates against real opponents over ~30+ games.

1. Create a regular Lichess account at <https://lichess.org>.
2. Get an OAuth token at <https://lichess.org/account/oauth/token> (scope: `bot:play`).
3. Upgrade the account to BOT (one-time, irreversible — pick a fresh account):
   ```bash
   curl -d '' https://lichess.org/api/bot/account/upgrade -H "Authorization: Bearer YOUR_TOKEN"
   ```
4. Clone and install lichess-bot:
   ```bash
   git clone https://github.com/lichess-bot-devs/lichess-bot.git
   cd lichess-bot
   pip install -r requirements.txt
   ```
5. Copy `config.yml.default` to `config.yml` and edit:
   - `token: "YOUR_TOKEN"` (the one from step 2)
   - `engine:` section — point to your `uci_engine.py`:
     ```yaml
     engine:
       dir: "/Users/<you>/git/alphazero"
       name: "python3 scripts/uci_engine.py --checkpoint pretrained.pt"
       protocol: "uci"
     ```
6. Run: `python3 lichess-bot.py` — bot is now online and accepting challenges at `https://lichess.org/@/<your-bot-name>`.

## Repo layout

```
src/alphazero/
  network.py          ResNet (policy + value heads)
  mcts.py             PUCT-guided MCTS with Dirichlet root noise
  selfplay.py         single-process self-play worker
  parallel_selfplay.py NN-server + worker pool (used for chess)
  replay_buffer.py    FIFO buffer with symmetry augmentation
  trainer.py          self-play → buffer → train loop
  arena.py            head-to-head match harness
  supervised.py       Stockfish-corpus behavior cloning (SP3b)
  corpus.py           on-disk training corpus reader/writer
  cli.py              `python -m alphazero` entrypoint
  games/              TicTacToe, Connect4, Chess (the only per-game code)
  opponents/          Stockfish wrapper, minimax baselines
  solvers/            TTT perfect solver, Connect 4 minimax
scripts/              generate_chess_corpus.py, play_chess.py, uci_engine.py
configs/              one TOML per game (+ pretrain / refine for chess)
tests/                unit + kill-criterion E2E tests, invariant suite
docs/                 spec + plan markdown (rendered to GitHub Pages)
```

Architectural invariant: `MCTS` does not import `AlphaZeroNet`, and `AlphaZeroNet` does not import `MCTS`. They communicate through an `eval_fn(state) -> (priors, value)` signature, which is why adding a game touches one file.

## Tests

```bash
python3 -m pytest -m "not slow"
```

207 tests pass currently. The `slow` marker gates 4 end-to-end kill-criterion runs that take minutes; deselect them for fast iteration.

`tests/test_invariants.py` and `tests/test_invariants_chess.py` codify the silent-failure modes that historically bite self-play / MCTS implementations (wrong-net data, sign convention, root-noise leakage into eval, etc.) — one named test per failure mode.

## Related work

* [AlphaZero paper](https://arxiv.org/abs/1712.01815) — Silver et al. 2017
* [AlphaGo Zero paper](https://www.nature.com/articles/nature24270) — earlier work with arena gating
* [alpha-zero-general](https://github.com/suragnair/alpha-zero-general) — Surag Nair's pedagogical implementation
* [Leela Chess Zero](https://lczero.org/) — open-source community AlphaZero-style chess engine

## Authors

* Venkata Krishnan Sowrirajan ([@venkata91](https://github.com/venkata91))
* Co-developed with Claude (Anthropic's coding assistant)

## License

MIT.
