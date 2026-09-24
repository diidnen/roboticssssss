#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT="${1:?new timestamped output root required}"
CELL="$OUT_ROOT/task5"
WRAPPER=/home/exouser/FORTE/ACTIVEFORCING_E3_LONGHORIZON_FULLTASK_LOCALLIFT_20260902_053000/run_e3_qualification_wrapper.py
PY=/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python
WARP_CORE=/media/volume/newdata/exouser/tabero/env_isaaclab51/lib/python3.11/site-packages/isaacsim/extscache/omni.warp.core-1.8.2+lx64
OPENPI_CLIENT=/home/exouser/Tabero/benchmarks/openpi/openpi-client/src

if [[ -e "$OUT_ROOT" ]]; then
  echo "refusing existing stability-gate directory: $OUT_ROOT" >&2
  exit 3
fi
if ! pgrep -f '/home/exouser/FORTE_mass/qualify_mass_(tasks|structure)\.py' >/dev/null; then
  echo "MASS protected process not found; fail-closed" >&2
  exit 75
fi
IFS=',' read -r gpu_util gpu_free < <(
  nvidia-smi --query-gpu=utilization.gpu,memory.free --format=csv,noheader,nounits
)
gpu_util="${gpu_util//[[:space:]]/}"
gpu_free="${gpu_free//[[:space:]]/}"
if (( gpu_util >= 70 || gpu_free <= 10000 )); then
  echo "GPU gate closed: util=$gpu_util freeMiB=$gpu_free" >&2
  exit 75
fi
mkdir -p "$CELL"

E3_OBJECT_NAME=black_book_1 E3_FORCE_N=5.0 TABERO_ROOT=/home/exouser/Tabero \
  PYTHONNOUSERSITE=1 PYTHONPATH="$WARP_CORE:/home/exouser/Tabero:$OPENPI_CLIENT" \
  HDF5_TRAJ_SOURCE_DIR=/home/exouser/Tabero/benchmarks/datasets/libero/assembled_hdf5 \
  LIBERO_CONFIG_DIR=/home/exouser/Tabero/benchmarks/datasets/libero/config \
  LIBERO_ASSETS_DATA_DIR=/home/exouser/Tabero/benchmarks/datasets/libero/USD \
  OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
  "$PY" -u "$WRAPPER" \
    --server-host 127.0.0.1 --server-port 18881 \
    --task-suite libero_10 --task-id 5 \
    --language-instruction "pick up the book and place it in the back compartment of the caddy" \
    --control-mode tactile --task Isaac-Libero-Franka-Hybrid-Tactile-v0 \
    --num-total-experiments 5 --max-inference-steps 50 --replan-steps 10 \
    --seed 7410 --no-randomize-light --b5-output-dir "$CELL" \
    --b5-method E3_TRAIN_QUALIFICATION_F5_STABILITY --b5-friction 0.6
