#!/usr/bin/env bash
set -u

E5_ROOT=/home/exouser/FORTE/analysis/results/ACTIVEFORCING_E5_FRESH_UTILITY_E2E_20260902_053006
LOG=/home/exouser/ACTIVEFORCING_P0_SUPERVISOR.log
LOCK=/tmp/activeforcing_p0_supervisor.lock

exec 9>"$LOCK"
flock -n 9 || exit 0

while true; do
  if test -f "$E5_ROOT/E5_CURRENT_UTILITY_FRESH_RESET_TO_END_E2E_COMPLETE"; then
    exit 0
  fi
  if ! nvidia-smi --query-gpu=utilization.gpu,memory.free --format=csv,noheader,nounits >/dev/null 2>&1; then
    sleep 30
    continue
  fi
  if ps -eo cmd= | rg -q 'core_gpu_scheduler.py'; then
    sleep 30
    continue
  fi
  {
    date -u
    echo 'starting protected P0 scheduler after GPU recovery'
  } >> "$LOG"
  cd "$E5_ROOT" || exit 2
  AF_GPU_GATE=YES \
  GLOBAL_GPU_TOKEN=ACTIVEFORCING_GLOBAL_RECOVERY_20260902 \
  CAMPAIGN_TOKEN=ACTIVEFORCING_E5_MASS_HANDOFF_20260902 \
  SHARD_TOKEN=ACTIVEFORCING_E5_NEXT_ATOMIC_SHARD_20260902 \
  /usr/bin/python3 -u "$E5_ROOT/core_gpu_scheduler.py" >> "$LOG" 2>&1
  status=$?
  echo "scheduler_exit=$status" >> "$LOG"
  if test -f "$E5_ROOT/E5_CURRENT_UTILITY_FRESH_RESET_TO_END_E2E_COMPLETE"; then
    exit 0
  fi
  sleep 30
done
