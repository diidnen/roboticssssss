#!/usr/bin/env bash
set -euo pipefail

BASE=/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911
REPO="$BASE/RoboTwin"
PY="$BASE/venv_robotwin/bin/python"
OUT="$BASE/experiments/af_dump_bin_bigbin_rootlocal_forcecal_20260912"
CHECKPOINT="$BASE/checkpoints/pi0_robotwin_30000/30000"

for policy_seed in 90200002 100200002 110200002 120200002; do
  "$PY" "$REPO/scripts/rootlocal_force_calibration.py" \
    --repo "$REPO" \
    --checkpoint "$CHECKPOINT" \
    --out "$OUT" \
    --policy-seed "$policy_seed" \
    --case 0.25,2 --case 0.25,2.5 --case 0.25,3 \
    --case 0.55,2 --case 0.55,2.5 --case 0.55,3 \
    --case 0.85,3 --case 0.85,3.5 --case 0.85,4
  cp -p "$OUT/manifest.json" "$OUT/manifest_repeat_${policy_seed}.json"
done
