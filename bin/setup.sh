#!/usr/bin/env bash
# Install all dependencies for AlphaZero (incl. Stockfish for chess eval).
# Idempotent; safe to re-run.
set -euo pipefail

# Detect platform and install Stockfish binary
if [[ "$(uname)" == "Darwin" ]]; then
    if ! command -v stockfish &>/dev/null; then
        echo "Installing Stockfish via Homebrew..."
        brew install stockfish
    fi
elif [[ -f /etc/debian_version ]]; then
    if ! dpkg -s stockfish &>/dev/null; then
        echo "Installing Stockfish via apt..."
        sudo apt-get update && sudo apt-get install -y stockfish
    fi
else
    echo "Unknown platform; please install stockfish manually." >&2
fi

# Python deps
pip install -e ".[dev]"

# Verify Stockfish is usable from python-chess
python3 -c "
import chess.engine
eng = chess.engine.SimpleEngine.popen_uci('stockfish')
eng.configure({'UCI_LimitStrength': True, 'UCI_Elo': 1500})
print('✓ Stockfish OK; UCI_Elo configurable')
eng.quit()
"
