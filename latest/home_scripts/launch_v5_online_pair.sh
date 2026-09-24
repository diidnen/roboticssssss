#!/usr/bin/env bash
set -euo pipefail
ROOT=/media/volume/data/exouser/activeforcing_table_push_v5_20260910
PY=/media/volume/newdata/exouser/flowdagger_e960a/e965_sirius_retrospective_correction_efficiency/env/bin/python
export PYTHONPATH=/home/exouser/SoftVTBench/openpi/upstream/packages/openpi-client/src:/home/exouser/Tabero/analysis/results/pi0_nontransport_20260910/client_venv/lib/python3.10/site-packages
exec > >(tee -a /home/exouser/activeforcing_table_push_20260910.log) 2>&1
mkdir -p "$ROOT/online"
echo "[V5 ONLINE] untouched start $(date -Is)"
"$PY" -u "$ROOT/code/online_rollout.py" untouched
echo "[V5 ONLINE] hybrid start $(date -Is)"
"$PY" -u "$ROOT/code/online_rollout.py" hybrid
echo "[V5 ONLINE] adjudicate $(date -Is)"
"$PY" -u "$ROOT/code/aggregate_online.py"
echo "[V5 ONLINE] complete $(date -Is)"
