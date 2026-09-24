#!/usr/bin/env bash
set -euo pipefail
ROOT=/media/volume/data/exouser/activeforcing_table_push_v6_20260910
PY=/media/volume/newdata/exouser/flowdagger_e960a/e965_sirius_retrospective_correction_efficiency/env/bin/python
export PYTHONPATH=/home/exouser/Tabero/analysis/results/pi0_nontransport_20260910/client_venv/lib/python3.10/site-packages:$ROOT/code:/media/volume/data/exouser/activeforcing_table_push_v5_20260910/code
exec > >(tee -a /home/exouser/activeforcing_table_push_20260910.log) 2>&1
echo "[V6 R2] start $(date -Is)"
for step in 55 57; do
  "$PY" -u "$ROOT/code/r2_branch.py" "$step" untouched 0
  "$PY" -u "$ROOT/code/r2_branch.py" "$step" disabled 0
  for force in 10 15 20 25 30; do
    "$PY" -u "$ROOT/code/r2_branch.py" "$step" minimum "$force"
  done
done
"$PY" -u "$ROOT/code/aggregate_r2.py"
echo "[V6 R2] complete $(date -Is)"
