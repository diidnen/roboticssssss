#!/usr/bin/env bash
set -euo pipefail
ROOT=/media/volume/data/exouser/activeforcing_knob_damping_v1_20260911
PY=/media/volume/newdata/exouser/flowdagger_e960a/e965_sirius_retrospective_correction_efficiency/env/bin/python
export PYTHONPATH=/home/exouser/SoftVTBench/openpi/upstream/packages/openpi-client/src:/home/exouser/Tabero/analysis/results/pi0_nontransport_20260910/client_venv/lib/python3.10/site-packages
exec > >(tee -a /home/exouser/activeforcing_table_push_20260910.log) 2>&1
for damping in 32 64 128; do
  export KNOB_DAMPING=$damping
  echo "[KNOB NOQUERY HIGH SWEEP] damping=${damping} start $(date -Is)"
  "$PY" -u "$ROOT/code/noquery_sweep.py"
done
echo "[KNOB NOQUERY HIGH SWEEP] complete $(date -Is)"
