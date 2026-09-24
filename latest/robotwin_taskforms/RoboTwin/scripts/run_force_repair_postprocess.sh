#!/usr/bin/env bash
set -euo pipefail
BASE=/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911
OUT="$BASE/experiments/af_dump_bin_bigbin_force_repair_20260912_v1"
SRC="$BASE/experiments/af_dump_bin_bigbin_forcegrid324_v1"
PID=$(<"$OUT/qualification.pid")
while kill -0 "$PID" 2>/dev/null; do sleep 30; done
PY="$BASE/venv_robotwin/bin/python"
REPO="$BASE/RoboTwin"
"$PY" "$REPO/scripts/make_force_repair_artifacts.py" --experiment "$SRC" --out "$OUT" --repeated "$OUT/qualification_branches.jsonl"
"$PY" "$REPO/scripts/prepare_repaired_dataset.py" --experiment "$SRC" --repeated "$OUT/qualification_branches.jsonl" --out "$OUT"
"$PY" "$REPO/scripts/train_repaired_models.py" --records "$OUT/repaired_records.jsonl" --out "$OUT/models" --device cpu
"$PY" "$REPO/scripts/evaluate_repaired_models.py" --records "$OUT/repaired_records.jsonl" --training-report "$OUT/models/training_summary.json" --out "$OUT/repaired_inference_report.json"
date -Is > "$OUT/postprocess.done"
