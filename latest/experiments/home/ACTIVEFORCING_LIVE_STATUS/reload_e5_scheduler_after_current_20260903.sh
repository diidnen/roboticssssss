#!/usr/bin/env bash
set -euo pipefail

LIVE=/home/exouser/ACTIVEFORCING_LIVE_STATUS
LOG=${LIVE}/GPU_QUEUE_HANDOFF.log
PARENT=2237816
WORKER=2240890
SHARD=/media/volume/newdata/exouser/activeforcing_e5_shards_20260902/ACTIVEFORCING_E5_FRESH_UTILITY_E2E_20260903_014326_task5_offset04
LINK=/home/exouser/FORTE/analysis/results/ACTIVEFORCING_E5_FRESH_UTILITY_E2E_20260903_014326_task5_offset04

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" >> "$LOG"
}

log "ten_gib_reload_watcher_start pid=$$ parent=$PARENT worker=$WORKER"
while kill -0 "$WORKER" 2>/dev/null; do
  sleep 2
done
log "current_worker_exited"

for _ in $(seq 1 20); do
  if [ -e "$LINK" ]; then
    break
  fi
  if ! kill -0 "$PARENT" 2>/dev/null; then
    break
  fi
  sleep 2
done

if kill -0 "$PARENT" 2>/dev/null; then
  kill -TERM "$PARENT"
  log "old_scheduler_reloaded_after_current_shard"
fi

for _ in $(seq 1 15); do
  if ! kill -0 "$PARENT" 2>/dev/null; then
    break
  fi
  sleep 1
done

if pgrep -af '[r]un_e5_fresh_utility.py' >/dev/null; then
  log "reload_deferred_existing_e5_worker_detected"
  exit 21
fi

tmux new-session -d -s af_e5_resume_20260903f env \
  AF_GPU_GATE=YES \
  GLOBAL_GPU_TOKEN=ACTIVEFORCING_GLOBAL_RECOVERY_20260902 \
  CAMPAIGN_TOKEN=ACTIVEFORCING_E5_MASS_HANDOFF_20260902 \
  SHARD_TOKEN=ACTIVEFORCING_E5_NEXT_ATOMIC_SHARD_20260902 \
  /bin/bash "$LIVE/resume_gpu_queue_after_mass.sh"
log "ten_gib_scheduler_started tmux=af_e5_resume_20260903f"
