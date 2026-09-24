#!/usr/bin/env bash
set -euo pipefail
ROOT=/media/volume/data/exouser/activeforcing_knob_damping_v2_20260911
PY=/media/volume/newdata/exouser/flowdagger_e960a/e965_sirius_retrospective_correction_efficiency/env/bin/python
export PYTHONPATH=/home/exouser/Tabero/analysis/results/pi0_nontransport_20260910/client_venv/lib/python3.10/site-packages:$ROOT/code:/media/volume/data/exouser/activeforcing_table_push_v5_20260910/code
exec > >(tee -a /home/exouser/activeforcing_table_push_20260910.log) 2>&1
echo "[KNOB R2] start $(date -Is)"
"$PY" -u "$ROOT/code/r2_branch.py" untouched 0
"$PY" -u "$ROOT/code/r2_branch.py" disabled 0
for torque in 0 2 4 6 8 12; do
  "$PY" -u "$ROOT/code/r2_branch.py" minimum "$torque"
done
"$PY" -u "$ROOT/code/aggregate_r2.py"
echo "[KNOB R2] complete $(date -Is)"
