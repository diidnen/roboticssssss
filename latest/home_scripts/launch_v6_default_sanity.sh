#!/usr/bin/env bash
set -euo pipefail
ROOT=/media/volume/data/exouser/activeforcing_table_push_v6_20260910
PY=/media/volume/newdata/exouser/flowdagger_e960a/e965_sirius_retrospective_correction_efficiency/env/bin/python
export PYTHONPATH=/home/exouser/SoftVTBench/openpi/upstream/packages/openpi-client/src:/home/exouser/Tabero/analysis/results/pi0_nontransport_20260910/client_venv/lib/python3.10/site-packages:$ROOT/code:/media/volume/data/exouser/activeforcing_table_push_v5_20260910/code
export AF_RUN_KIND=calibration
export AF_TABLE_MU=0.6
export AF_TARGET_N=10
exec > >(tee -a /home/exouser/activeforcing_table_push_20260910.log) 2>&1
echo "[V6 DEFAULT-MU SANITY] start $(date -Is)"
"$PY" -u "$ROOT/code/default_sanity_pair.py"
echo "[V6 DEFAULT-MU SANITY] complete $(date -Is)"
