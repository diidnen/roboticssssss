#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/exouser/Tabero"
OUT="${ROOT}/analysis/results/m1r2_real_cross_task_probe_20260822_133903"
PY="/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python"
WARP_CORE="/media/volume/newdata/exouser/tabero/env_isaaclab51/lib/python3.11/site-packages/isaacsim/extscache/omni.warp.core-1.8.2+lx64"
OPENPI_CLIENT_SRC="${ROOT}/benchmarks/openpi/openpi-client/src"

export PYTHONNOUSERSITE=1
export PYTHONPATH="${WARP_CORE}:${ROOT}:${OPENPI_CLIENT_SRC}"
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export TABERO_ROOT="${ROOT}"
export HDF5_TRAJ_SOURCE_DIR="${ROOT}/benchmarks/datasets/libero/assembled_hdf5"
export LIBERO_CONFIG_DIR="${ROOT}/benchmarks/datasets/libero/config"
export LIBERO_ASSETS_DATA_DIR="${ROOT}/benchmarks/datasets/libero/USD"
export M1R2_OUT="${OUT}"
export M1R2_N_SEEDS="${M1R2_N_SEEDS:-20}"
export M1R2_MUS="${M1R2_MUS:-0.2,0.5,1.0}"
export M1R2_RESUME="${M1R2_RESUME:-1}"

cd "${ROOT}"
for task in 0 1 2 5 6; do
  export M1R2_TASK_ID="${task}"
  echo "[M1R2] collecting task ${task}, seeds=${M1R2_N_SEEDS}, mus=${M1R2_MUS}" | tee -a "${OUT}/logs/run_collect_all.log"
  "${PY}" -u "${OUT}/scripts/m1r2_collect_real_probe.py" 2>&1 | tee -a "${OUT}/logs/run_collect_all.log"
  "${PY}" - <<PY
import csv
from pathlib import Path
expected = int("${M1R2_N_SEEDS}") * len("${M1R2_MUS}".split(","))
p = Path("${OUT}") / f"TASK${task}_PROBE.csv"
if not p.exists():
    raise SystemExit(f"missing {p}")
with p.open() as f:
    n = sum(1 for _ in csv.DictReader(f))
if n < expected:
    raise SystemExit(f"{p.name} has {n} rows, expected >= {expected}")
print(f"[M1R2] verified {p.name}: {n} rows")
PY
  echo "[M1R2] completed task ${task}" | tee -a "${OUT}/logs/run_collect_all.log"
done
