#!/usr/bin/env bash
set -euo pipefail

OUT="${1:?output directory required}"
mkdir -p "$OUT/ensemble_logs"
pids=()
for fold in 0 1 2; do
  /usr/bin/python3 /home/exouser/FORTE/run_probe_scalar_direct_residual.py ensemble_shard \
    --out "$OUT" --fold "$fold" >"$OUT/ensemble_logs/fold${fold}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done
