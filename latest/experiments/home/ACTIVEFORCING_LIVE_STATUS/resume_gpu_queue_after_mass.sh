#!/usr/bin/env bash
set -euo pipefail

LIVE=/home/exouser/ACTIVEFORCING_LIVE_STATUS
SCHED=/home/exouser/FORTE/analysis/results/ACTIVEFORCING_E5_FRESH_UTILITY_E2E_20260902_053006/core_gpu_scheduler.py
LOG=${LIVE}/GPU_QUEUE_HANDOFF.log

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" >> "$LOG"
}

log "watcher_start pid=$$"

while pgrep -af '[c]ollect_mass_structured_formal.py' >/dev/null; do
  log "waiting_for_mass"
  sleep 20
done

# Allow the pre-existing wait wrapper to observe Mass completion and exit/fail-closed.
sleep 15

if pgrep -af '[r]un_e5_fresh_utility.py' >/dev/null; then
  log "existing_e5_worker_detected_fail_closed"
  exit 21
fi
if pgrep -af '[r]eplay_demos_with_camera.py' >/dev/null; then
  log "existing_e3_worker_detected_fail_closed"
  exit 22
fi

for sample in 1 2; do
  if ! nvidia-smi --query-gpu=utilization.gpu,memory.free --format=csv,noheader,nounits >> "$LOG" 2>&1; then
    log "nvidia_query_failed"
    exit 20
  fi
  if [[ "$sample" == 1 ]]; then
    sleep 3
  fi
done

free_bytes=$(df -B1 --output=avail /home/exouser | tail -1 | tr -d ' ')
if (( free_bytes < 10 * 1024 * 1024 * 1024 )); then
  log "disk_gate_failed free_bytes=$free_bytes"
  exit 23
fi

log "admission_pass tokens=GLOBAL_GPU_TOKEN,CAMPAIGN_TOKEN,SHARD_TOKEN free_bytes=$free_bytes"
export AF_GPU_GATE=YES
export GLOBAL_GPU_TOKEN=ACTIVEFORCING_GLOBAL_RECOVERY_20260902
export CAMPAIGN_TOKEN=ACTIVEFORCING_E5_MASS_HANDOFF_20260902
export SHARD_TOKEN=ACTIVEFORCING_E5_NEXT_ATOMIC_SHARD_20260902

exec /usr/bin/python3 "$SCHED"
