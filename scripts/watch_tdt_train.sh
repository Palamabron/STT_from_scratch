#!/usr/bin/env bash
# Monitor TDT training; restart on crash (OOM -> sm-oom, else sm).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$REPO_ROOT/logs/watch_tdt.log"
TRAIN_LOG="$REPO_ROOT/logs/train_tdt_4090_45m.log"
TARGET="${1:-sm}"

log() { echo "$(date -Iseconds) $*" | tee -a "$LOG"; }

is_training() {
  pgrep -f "\.venv/bin/python3 -m SpeechToText\.models\.tdt\.train" >/dev/null 2>&1
}

last_step() {
  grep -oE 'step=[0-9]+' "$TRAIN_LOG" 2>/dev/null | tail -1 | cut -d= -f2 || echo 0
}

restart() {
  local make_target="$1"
  log "Restarting: make train-tdt-4090-${make_target}"
  cd "$REPO_ROOT"
  make "train-tdt-4090-${make_target}" >>"$TRAIN_LOG" 2>&1 &
  sleep 30
}

log "Watchdog started (target=train-tdt-4090-${TARGET})"
STALL_COUNT=0
LAST_STEP="$(last_step)"

while true; do
  sleep 300
  if is_training; then
    CUR="$(last_step)"
    if [[ "$CUR" == "$LAST_STEP" ]] && [[ "$CUR" -gt 0 ]]; then
      STALL_COUNT=$((STALL_COUNT + 1))
      log "WARN: step stuck at $CUR (${STALL_COUNT}x)"
      if [[ "$STALL_COUNT" -ge 6 ]]; then
        log "Stalled 30+ min — killing and restarting"
        pkill -f "SpeechToText.models.tdt.train" || true
        sleep 10
        restart "$TARGET"
        STALL_COUNT=0
        LAST_STEP="$(last_step)"
      fi
    else
      STALL_COUNT=0
      LAST_STEP="$CUR"
      log "OK: step=$CUR"
    fi
    continue
  fi

  log "Training not running — checking log for cause"
  if tail -30 "$TRAIN_LOG" 2>/dev/null | grep -q "OutOfMemoryError"; then
    log "OOM detected -> sm-oom"
    TARGET=sm-oom
  fi
  restart "$TARGET"
  STALL_COUNT=0
  LAST_STEP="$(last_step)"
done
