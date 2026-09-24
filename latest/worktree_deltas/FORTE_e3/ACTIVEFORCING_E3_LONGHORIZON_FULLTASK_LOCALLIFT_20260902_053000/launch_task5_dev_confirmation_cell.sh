#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT="${1:?existing E3 pilot output root required}"
ROOT_SEED="${2:?DEV root seed required (7500 or 7501)}"
FORCE_N="${3:?DEV force required (4, 6, or 7 N)}"

case "$ROOT_SEED" in
  7500|7501) ;;
  *) echo "refusing non-frozen DEV root: $ROOT_SEED" >&2; exit 2 ;;
esac
case "$FORCE_N" in
  4|6|7) ;;
  *) echo "refusing non-frozen DEV anchor force: $FORCE_N" >&2; exit 2 ;;
esac

ART=/home/exouser/FORTE_e3/ACTIVEFORCING_E3_LONGHORIZON_FULLTASK_LOCALLIFT_20260902_053000
WRAPPER="$ART/run_e3_reserve_qualification_wrapper.py"
TABERO_E3=/media/volume/newdata/exouser/Tabero_e3lh
ISAAC_PY=/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python
OPENPI_CLIENT="$TABERO_E3/benchmarks/openpi/openpi-client/src"
WARP_CORE=/media/volume/newdata/exouser/tabero/env_isaaclab51/lib/python3.11/site-packages/isaacsim/extscache/omni.warp.core-1.8.2+lx64
RAW_DATA=/media/volume/newdata/exouser/tabero/data/Isaaclab_Libero
CELL="$OUT_ROOT/DEV_root${ROOT_SEED}_mu0.6_F${FORCE_N}N"

[[ -f "$ART/E3_TASK5_DEV_CONFIRMATION_PLAN.json" ]] || { echo "missing frozen DEV plan" >&2; exit 3; }
[[ -d "$TABERO_E3" ]] || { echo "missing isolated Tabero worktree" >&2; exit 3; }
[[ -x "$ISAAC_PY" ]] || { echo "missing Isaac Python" >&2; exit 3; }
[[ -d "$RAW_DATA" ]] || { echo "missing immutable raw data" >&2; exit 3; }
if [[ -e "$CELL" ]]; then
  echo "refusing to overwrite existing DEV cell: $CELL" >&2
  exit 4
fi
mkdir -p "$CELL"

E3_OBJECT_NAME=black_book_1 E3_FORCE_N="$FORCE_N" TABERO_ROOT="$TABERO_E3" \
  PYTHONNOUSERSITE=1 PYTHONPATH="$WARP_CORE:$TABERO_E3:$OPENPI_CLIENT" \
  HDF5_TRAJ_SOURCE_DIR="$RAW_DATA/assembled_hdf5" \
  LIBERO_CONFIG_DIR="$TABERO_E3/benchmarks/datasets/libero/config" \
  LIBERO_ASSETS_DATA_DIR="$RAW_DATA/USD" \
  OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
  "$ISAAC_PY" -u "$WRAPPER" \
    --server-host 127.0.0.1 --server-port 18881 \
    --task-suite libero_10 --task-id 5 \
    --language-instruction "pick up the book and place it in the back compartment of the caddy" \
    --control-mode tactile --task Isaac-Libero-Franka-Hybrid-Tactile-v0 \
    --num-total-experiments 1 --max-inference-steps 50 --replan-steps 10 \
    --seed "$ROOT_SEED" --no-randomize-light \
    --b5-output-dir "$CELL" \
    --b5-method E3_DEV_FIXED_ANCHOR_CONFIRMATION --b5-friction 0.6
