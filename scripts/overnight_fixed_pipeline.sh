#!/usr/bin/env bash
set -uo pipefail

cd /Users/vsowrira/git/alphazero || exit 1

PY=/export/apps/python/3.11/bin/python3
PIPELINE_LOG=overnight_fixed_pipeline.out
PRETRAIN_LOG=pretrain_fixed.out
EVAL_LOG=eval_pretrained_fixed_25.out
REFINE_LOG=refine_fixed.out

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$PIPELINE_LOG"
}

log "overnight pipeline started"
PRETRAIN_PATTERN='python3 -u -m alphazero pretrain --config configs/chess-pretrain-fixed.toml'

log "waiting for fixed pretraining process to finish"

while pgrep -f "$PRETRAIN_PATTERN" >/dev/null 2>&1; do
  sleep 60
done

log "fixed pretraining process finished"
if [ -f "$PRETRAIN_LOG" ]; then
  log "last pretraining log lines:"
  tail -20 "$PRETRAIN_LOG" | tee -a "$PIPELINE_LOG"
fi

if [ ! -f pretrained_fixed.pt ]; then
  log "ERROR: pretrained_fixed.pt is missing; aborting eval/refinement"
  exit 1
fi

"$PY" - <<'PY' >> "$PIPELINE_LOG" 2>&1
import torch
ckpt = torch.load("pretrained_fixed.pt", map_location="cpu", weights_only=False)
print("pretrained_fixed.pt metadata:")
print("  _pretrain_epoch:", ckpt.get("_pretrain_epoch"))
print("  iteration:", ckpt.get("iteration"))
print("  output_checkpoint:", ckpt.get("config", {}).get("output_checkpoint"))
PY

log "running eval of pretrained_fixed.pt vs Stockfish 1500; log=$EVAL_LOG"
caffeinate -dims "$PY" -u -m alphazero eval \
  --game chess \
  --checkpoint pretrained_fixed.pt \
  --config configs/chess-refine-fixed.toml \
  --num-games 25 \
  > "$EVAL_LOG" 2>&1
eval_status=$?
log "pretrained eval exited with status $eval_status"
tail -40 "$EVAL_LOG" | tee -a "$PIPELINE_LOG"

log "starting refinement from pretrained_fixed.pt; log=$REFINE_LOG"
caffeinate -dims "$PY" -u -m alphazero train \
  --game chess \
  --config configs/chess-refine-fixed.toml \
  --resume-from pretrained_fixed.pt \
  > "$REFINE_LOG" 2>&1
refine_status=$?
log "refinement exited with status $refine_status"
tail -60 "$REFINE_LOG" | tee -a "$PIPELINE_LOG"
exit "$refine_status"
