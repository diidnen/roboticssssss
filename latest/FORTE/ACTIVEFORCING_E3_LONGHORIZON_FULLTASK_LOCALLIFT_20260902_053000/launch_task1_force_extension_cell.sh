#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT="${1:?timestamped output root required}"
SPLIT="${2:?TRAIN or DEV required}"
ROOT_SEED="${3:?root seed required}"
FRICTION="${4:?friction required}"
FORCE_N="${5:?force N required}"

case "$SPLIT" in TRAIN|DEV) ;; *) echo "invalid split: $SPLIT" >&2; exit 2 ;; esac
case "$ROOT_SEED" in 7600|7700|7701) ;; *) echo "unregistered root seed: $ROOT_SEED" >&2; exit 2 ;; esac
case "$FRICTION" in 0.2|0.6|1.0) ;; *) echo "unregistered friction: $FRICTION" >&2; exit 2 ;; esac
case "$FORCE_N" in 1|1.0|2|2.0|4|4.0|5|5.0|6|6.0) ;; *) echo "force outside frozen task1 1/2/4/5/6 N grid: $FORCE_N" >&2; exit 2 ;; esac

WRAPPER=/home/exouser/FORTE/ACTIVEFORCING_E3_LONGHORIZON_FULLTASK_LOCALLIFT_20260902_053000/run_e3_qualification_wrapper.py
PY=/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python
WARP_CORE=/media/volume/newdata/exouser/tabero/env_isaaclab51/lib/python3.11/site-packages/isaacsim/extscache/omni.warp.core-1.8.2+lx64
OPENPI_CLIENT=/home/exouser/Tabero/benchmarks/openpi/openpi-client/src
CELL="$OUT_ROOT/${SPLIT}_root${ROOT_SEED}_mu${FRICTION}_F${FORCE_N}N"

if [[ -e "$CELL" ]]; then
  echo "refusing to overwrite existing cell: $CELL" >&2
  exit 3
fi
mkdir -p "$CELL"

E3_OBJECT_NAME=cream_cheese_1 E3_FORCE_N="$FORCE_N" TABERO_ROOT=/home/exouser/Tabero \
  PYTHONNOUSERSITE=1 PYTHONPATH="$WARP_CORE:/home/exouser/Tabero:$OPENPI_CLIENT" \
  HDF5_TRAJ_SOURCE_DIR=/home/exouser/Tabero/benchmarks/datasets/libero/assembled_hdf5 \
  LIBERO_CONFIG_DIR=/home/exouser/Tabero/benchmarks/datasets/libero/config \
  LIBERO_ASSETS_DATA_DIR=/home/exouser/Tabero/benchmarks/datasets/libero/USD \
  OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
  "$PY" -u "$WRAPPER" \
    --server-host 127.0.0.1 --server-port 18881 \
    --task-suite libero_object --task-id 1 \
    --language-instruction "pick up the cream cheese and place it in the basket" \
    --control-mode tactile --task Isaac-Libero-Franka-Hybrid-Tactile-v0 \
    --num-total-experiments 1 --max-inference-steps 50 --replan-steps 10 \
    --seed "$ROOT_SEED" --no-randomize-light --b5-output-dir "$CELL" \
    --b5-method "E3_TASK1_${SPLIT}_FORCE_EXTENSION" --b5-friction "$FRICTION"
