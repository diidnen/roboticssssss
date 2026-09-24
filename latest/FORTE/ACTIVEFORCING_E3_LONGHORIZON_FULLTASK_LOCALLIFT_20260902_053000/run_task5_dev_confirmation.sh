#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT="${1:?timestamped pilot output root required}"
TRAIN_GATE="$OUT_ROOT/E3_TASK5_TRAIN_REGIME_GATE.json"
CELL_LAUNCHER=/home/exouser/FORTE/ACTIVEFORCING_E3_LONGHORIZON_FULLTASK_LOCALLIFT_20260902_053000/launch_task5_pilot_cell.sh

if [[ ! -f "$TRAIN_GATE" ]]; then
  echo "missing frozen TRAIN gate: $TRAIN_GATE" >&2
  exit 4
fi
if ! python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["gate_pass"] else 1)' "$TRAIN_GATE"; then
  echo "TRAIN gate did not pass; DEV launch is prohibited" >&2
  exit 4
fi

for root_seed in 7500 7501; do
  for friction in 0.2 0.6 1.0; do
    for force_n in 1 2 3 4 5; do
      cell="$OUT_ROOT/DEV_root${root_seed}_mu${friction}_F${force_n}N"
      if [[ -d "$cell" ]]; then
        if compgen -G "$cell/logs/*_episodes.csv" >/dev/null && compgen -G "$cell/logs/*_steps.csv" >/dev/null; then
          echo "resume: keeping completed cell $cell"
          continue
        fi
        echo "refusing incomplete existing cell: $cell" >&2
        exit 3
      fi
      if ! pgrep -f '/home/exouser/FORTE_mass/qualify_mass_(tasks|structure)\.py' >/dev/null; then
        echo "MASS protected process not found; fail-closed before next cell" >&2
        exit 75
      fi
      IFS=',' read -r gpu_util gpu_free < <(
        nvidia-smi --query-gpu=utilization.gpu,memory.free --format=csv,noheader,nounits
      )
      gpu_util="${gpu_util//[[:space:]]/}"
      gpu_free="${gpu_free//[[:space:]]/}"
      if (( gpu_util >= 70 || gpu_free <= 10000 )); then
        echo "GPU gate closed before root=$root_seed mu=$friction F=$force_n: util=$gpu_util freeMiB=$gpu_free" >&2
        exit 75
      fi
      bash "$CELL_LAUNCHER" "$OUT_ROOT" DEV "$root_seed" "$friction" "$force_n"
    done
  done
done
