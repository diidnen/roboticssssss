#!/usr/bin/env bash
set -euo pipefail

BASE=/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911
REPO="$BASE/RoboTwin"
EXP="$BASE/experiments/af_dump_bin_bigbin_rootlocal_forcecal_20260912"
PY="$BASE/venv_robotwin/bin/python3"
CKPT="$BASE/checkpoints/pi0_robotwin_30000/30000"
BELIEF="$EXP/models_rootlocal_v3/training_report.json"
FREEZE="$EXP/models_isotonic_v1/selection_freeze.json"
export PATH="$BASE/runtime_bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

cd "$REPO"

for policy_seed in 140200002 150200002; do
  out="$EXP/final_${policy_seed}"
  mkdir -p "$out"
  if [[ ! -f "$out/selection_lock.json" ]]; then
    "$PY" scripts/rootlocal_lock_selector.py \
      --repo "$REPO" \
      --checkpoint "$CKPT" \
      --belief-report "$BELIEF" \
      --selector-freeze "$FREEZE" \
      --out "$out" \
      --policy-seed "$policy_seed" \
      >"$out/lock_stdout.log" 2>&1
  fi
done

for policy_seed in 140200002 150200002; do
  out="$EXP/final_${policy_seed}"
  "$PY" scripts/rootlocal_force_calibration.py \
    --repo "$REPO" \
    --checkpoint "$CKPT" \
    --out "$out" \
    --root-seed 200002 \
    --root-slot 1 \
    --policy-seed "$policy_seed" \
    --case 0.25,2 --case 0.25,2.5 --case 0.25,3 --case 0.25,3.5 --case 0.25,4 \
    --case 0.55,2 --case 0.55,2.5 --case 0.55,3 --case 0.55,3.5 --case 0.55,4 \
    --case 0.85,2 --case 0.85,2.5 --case 0.85,3 --case 0.85,3.5 --case 0.85,4 \
    >"$out/grid_stdout.log" 2>&1
done

date -u +%FT%TZ >"$EXP/final_online_complete.txt"
