#!/usr/bin/env bash
set -euo pipefail

out=/home/exouser/FORTE/activeforcing_probe_conditioned_wm_20260901_064627
mkdir -p "$out/logs"
pids=()
for fold in 0 1 2; do
  for seed in 0 1 2; do
    /usr/bin/python3 /home/exouser/FORTE/run_probe_conditioned_wm.py shard \
      --out "$out" --fold "$fold" --seed "$seed" \
      >"$out/logs/fold${fold}_seed${seed}.log" 2>&1 &
    pids+=("$!")
  done
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
