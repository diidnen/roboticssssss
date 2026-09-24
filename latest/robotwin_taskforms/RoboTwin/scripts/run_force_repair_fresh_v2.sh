#!/usr/bin/env bash
set -euo pipefail
BASE=/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911
OUT="$BASE/experiments/af_dump_bin_bigbin_force_repair_20260912_v2"
ROOT_PID=$(<"$OUT/fresh_root_qualification.pid")
while kill -0 "$ROOT_PID" 2>/dev/null; do sleep 30; done
PY="$BASE/venv_robotwin/bin/python"
REPO="$BASE/RoboTwin"
"$PY" "$REPO/scripts/evaluate_af_taskforms_fresh_online.py" \
  --experiment "$OUT" \
  --fresh-root-map "$OUT/fresh_root_map.json" \
  --training-report "$OUT/models/training_summary.json" \
  --inference-report "$OUT/repaired_inference_report.json" \
  > "$OUT/fresh_online.log" 2>&1
cp -f "$OUT/fresh_test_summary.md" "$OUT/fresh_test_summary.provisional.md" 2>/dev/null || true
date -Is > "$OUT/fresh_online.done"
