# AlphaZero — Connect 4 Colab Notebook

Train an AlphaZero agent to play Connect 4 on a free Colab GPU (~1–2 hours on T4)
with automatic checkpoint persistence to Google Drive.

## Quick Start

1. Go to https://colab.research.google.com/
2. **File → Open notebook → GitHub**
   Paste the URL:
   ```
   https://github.com/venkata91/alphazero/blob/main/notebooks/train_connect4_colab.ipynb
   ```
3. **Runtime → Change runtime type → T4 GPU**
   (or L4 / A100 if you have Colab Pro)
4. **Runtime → Run all**
5. When prompted, approve Google Drive mount to enable checkpoint persistence.

## What the Notebook Does

| Step | Description |
|------|-------------|
| 1 | Verify GPU availability (`nvidia-smi`, PyTorch CUDA check) |
| 2 | Clone the repo from GitHub |
| 3 | Install the `alphazero` package and dependencies |
| 4 | Mount Google Drive; create `My Drive/alphazero/checkpoints/connect4` |
| 5 | Symlink `./checkpoints` → Drive; detect latest checkpoint for resume |
| 6 | Run training (`python -m alphazero train --game connect4 ...`) |
| 7 | List all Drive checkpoints; download the final one to your browser |
| 8 | *(Optional)* Evaluate the trained agent against minimax-depth-8 |

## Resuming After a Disconnect

Colab free-tier sessions time out after ~12 hours of inactivity.
Because all checkpoints are written to Drive via a symlink, simply
**Run all** in a new session — the notebook auto-detects the latest
checkpoint and passes `--resume-from` to the trainer.

## Checkpoint Location

```
My Drive / alphazero / checkpoints / connect4 / iter_NNNN.pt
```

## Config Details

- Architecture: 6 residual blocks × 64 channels
- MCTS simulations: 100
- Training iterations: 50
- Config file: `configs/connect4.toml`
