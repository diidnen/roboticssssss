#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/exouser/Tabero"
OUT="${P4R2_OUT:-/home/exouser/Tabero/analysis/results/p4r2_contact_feasible_multi_primitive_20260823_070150}"
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
export P4R2_OUT="${OUT}"
export P4R2_MUS="${P4R2_MUS:-0.2,0.5,1.0}"
export P4R2_RESUME="${P4R2_RESUME:-1}"

mkdir -p "${OUT}/logs" "${OUT}/P4R2_TIMESTEP_TELEMETRY"
cd "${ROOT}"

run_split() {
  local split="$1"
  local seed_offset="$2"
  local n_seeds="$3"
  export P4R2_SPLIT="${split}"
  export P4R2_SEED_OFFSET="${seed_offset}"
  export P4R2_N_SEEDS="${n_seeds}"
  for primitive in S L; do
    export P4R2_PRIMITIVE="${primitive}"
    for task in 0 1 2 5 6; do
      export P4R2_TASK_ID="${task}"
      echo "[P4R2] collecting split=${split} primitive=${primitive} task=${task} seeds=${n_seeds} offset=${seed_offset}" | tee -a "${OUT}/logs/run_collect_all.log"
      "${PY}" -u "${OUT}/scripts/p4r2_collect_probe.py" 2>&1 | tee -a "${OUT}/logs/run_collect_all.log"
      "${PY}" - <<PY
import csv
from pathlib import Path
expected = int("${n_seeds}") * len("${P4R2_MUS}".split(","))
p = Path("${OUT}") / f"P4R2_${split}_${primitive}_TASK${task}_PROBE.csv"
if not p.exists():
    raise SystemExit(f"missing {p}")
with p.open() as f:
    n = sum(1 for _ in csv.DictReader(f))
if n < expected:
    raise SystemExit(f"{p.name} has {n} rows, expected >= {expected}")
print(f"[P4R2] verified {p.name}: {n} rows")
PY
    done
  done
}

run_split development "${P4R2_DEV_SEED_OFFSET:-1000}" "${P4R2_DEV_N_SEEDS:-5}"
"${PY}" -u "${OUT}/scripts/p4r2_freeze_protocol.py" 2>&1 | tee -a "${OUT}/logs/run_collect_all.log"
run_split main "${P4R2_MAIN_SEED_OFFSET:-2000}" "${P4R2_MAIN_N_SEEDS:-20}"
"${PY}" -u "${OUT}/scripts/p4r2_analyze.py" 2>&1 | tee -a "${OUT}/logs/run_collect_all.log"
