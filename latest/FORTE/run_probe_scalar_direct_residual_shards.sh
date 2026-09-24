#!/usr/bin/env bash
set -euo pipefail

OUT="${1:?output directory required}"
mkdir -p "$OUT/logs"
pids=()
for fold in 0 1 2; do
  for seed in 0 1 2; do
    /usr/bin/python3 /home/exouser/FORTE/run_probe_scalar_direct_residual.py shard \
      --out "$OUT" --fold "$fold" --seed "$seed" \
      >"$OUT/logs/fold${fold}_seed${seed}.log" 2>&1 &
    pids+=("$!")
  done
done
for pid in "${pids[@]}"; do
  wait "$pid"
done
