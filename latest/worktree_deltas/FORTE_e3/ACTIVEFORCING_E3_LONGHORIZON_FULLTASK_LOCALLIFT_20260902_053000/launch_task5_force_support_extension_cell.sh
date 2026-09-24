#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT="${1:?canonical task5 pilot root required}"
FORCE_N="${2:?extension force required: 6, 7, or 8}"
case "$FORCE_N" in 6|6.0|7|7.0|8|8.0) ;; *) echo "unsupported extension force: $FORCE_N" >&2; exit 2 ;; esac

WRAPPER=/home/exouser/FORTE_e3/ACTIVEFORCING_E3_LONGHORIZON_FULLTASK_LOCALLIFT_20260902_053000/run_e3_reserve_qualification_wrapper.py
PY=/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python
TABERO_E3=/media/volume/newdata/exouser/Tabero_e3lh
WARP_CORE=/media/volume/newdata/exouser/tabero/env_isaaclab51/lib/python3.11/site-packages/isaacsim/extscache/omni.warp.core-1.8.2+lx64
OPENPI_CLIENT="$TABERO_E3/benchmarks/openpi/openpi-client/src"
CELL="$OUT_ROOT/TRAIN_root7400_mu0.6_F${FORCE_N}N"

if [[ -e "$CELL" ]]; then
  echo "refusing to overwrite existing cell: $CELL" >&2
  exit 3
fi
mkdir -p "$CELL"

E3_OBJECT_NAME=black_book_1 E3_FORCE_N="$FORCE_N" TABERO_ROOT="$TABERO_E3" \
  PYTHONNOUSERSITE=1 PYTHONPATH="$WARP_CORE:$TABERO_E3:$OPENPI_CLIENT" \
  HDF5_TRAJ_SOURCE_DIR=/media/volume/newdata/exouser/tabero/data/Isaaclab_Libero/assembled_hdf5 \
  LIBERO_CONFIG_DIR="$TABERO_E3/benchmarks/datasets/libero/config" \
  LIBERO_ASSETS_DATA_DIR=/media/volume/newdata/exouser/tabero/data/Isaaclab_Libero/USD \
  OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
  "$PY" -u "$WRAPPER" \
    --server-host 127.0.0.1 --server-port 18881 \
    --task-suite libero_10 --task-id 5 \
    --language-instruction "pick up the book and place it in the back compartment of the caddy" \
    --control-mode tactile --task Isaac-Libero-Franka-Hybrid-Tactile-v0 \
    --num-total-experiments 1 --max-inference-steps 50 --replan-steps 10 \
    --seed 7400 --no-randomize-light --b5-output-dir "$CELL" \
    --b5-method E3_TRAIN_FORCE_SUPPORT_EXTENSION --b5-friction 0.6
