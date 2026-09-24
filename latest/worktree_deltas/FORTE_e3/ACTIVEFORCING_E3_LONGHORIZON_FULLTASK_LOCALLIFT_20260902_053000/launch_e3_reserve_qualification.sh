#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT="${1:?timestamped output root required}"
CANDIDATE="${2:?candidate required: t2, t8, or g3}"

WRAPPER=/home/exouser/FORTE_e3/ACTIVEFORCING_E3_LONGHORIZON_FULLTASK_LOCALLIFT_20260902_053000/run_e3_reserve_qualification_wrapper.py
PY=/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python
TABERO_E3=/media/volume/newdata/exouser/Tabero_e3lh
WARP_CORE=/media/volume/newdata/exouser/tabero/env_isaaclab51/lib/python3.11/site-packages/isaacsim/extscache/omni.warp.core-1.8.2+lx64
OPENPI_CLIENT="$TABERO_E3/benchmarks/openpi/openpi-client/src"

case "$CANDIDATE" in
  t2)
    SUITE=libero_10
    TASK_ID=2
    OBJECT=moka_pot_1
    INSTRUCTION="turn on the stove and put the moka pot on it"
    ;;
  t8)
    SUITE=libero_10
    TASK_ID=8
    OBJECT=moka_pot_1
    INSTRUCTION="put both moka pots on the stove"
    ;;
  g3)
    SUITE=libero_goal
    TASK_ID=3
    OBJECT=akita_black_bowl_1
    INSTRUCTION="open the top drawer and put the bowl inside"
    ;;
  *)
    echo "unsupported candidate: $CANDIDATE" >&2
    exit 2
    ;;
esac

CELL="$OUT_ROOT/$CANDIDATE"
if [[ -e "$CELL" ]]; then
  echo "refusing to overwrite existing cell: $CELL" >&2
  exit 3
fi
mkdir -p "$CELL"

E3_OBJECT_NAME="$OBJECT" E3_FORCE_N=8.0 TABERO_ROOT="$TABERO_E3" \
  PYTHONNOUSERSITE=1 PYTHONPATH="$WARP_CORE:$TABERO_E3:$OPENPI_CLIENT" \
  HDF5_TRAJ_SOURCE_DIR=/media/volume/newdata/exouser/tabero/data/Isaaclab_Libero/assembled_hdf5 \
  LIBERO_CONFIG_DIR="$TABERO_E3/benchmarks/datasets/libero/config" \
  LIBERO_ASSETS_DATA_DIR=/media/volume/newdata/exouser/tabero/data/Isaaclab_Libero/USD \
  OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
  "$PY" -u "$WRAPPER" \
    --server-host 127.0.0.1 --server-port 18881 \
    --task-suite "$SUITE" --task-id "$TASK_ID" \
    --language-instruction "$INSTRUCTION" \
    --control-mode tactile --task Isaac-Libero-Franka-Hybrid-Tactile-v0 \
    --num-total-experiments 1 --max-inference-steps 80 --replan-steps 10 \
    --seed 7600 --no-randomize-light --b5-output-dir "$CELL" \
    --b5-method E3_RESERVE_NOMINAL_ROBUST_8N --b5-friction 0.6
