#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT="${1:?timestamped output root required}"
CANDIDATE="${2:?candidate task id required: 3, 9, or 5}"
WRAPPER=/home/exouser/FORTE/ACTIVEFORCING_E3_LONGHORIZON_FULLTASK_LOCALLIFT_20260902_053000/run_e3_qualification_wrapper.py
PY=/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python
WARP_CORE=/media/volume/newdata/exouser/tabero/env_isaaclab51/lib/python3.11/site-packages/isaacsim/extscache/omni.warp.core-1.8.2+lx64
OPENPI_CLIENT=/home/exouser/Tabero/benchmarks/openpi/openpi-client/src

run_one() {
  local task_id="$1"
  local object_name="$2"
  local instruction="$3"
  local out="$OUT_ROOT/task${task_id}"
  mkdir -p "$out"
  E3_OBJECT_NAME="$object_name" E3_FORCE_N=8.0 TABERO_ROOT=/home/exouser/Tabero \
    PYTHONNOUSERSITE=1 PYTHONPATH="$WARP_CORE:/home/exouser/Tabero:$OPENPI_CLIENT" \
    HDF5_TRAJ_SOURCE_DIR=/home/exouser/Tabero/benchmarks/datasets/libero/assembled_hdf5 \
    LIBERO_CONFIG_DIR=/home/exouser/Tabero/benchmarks/datasets/libero/config \
    LIBERO_ASSETS_DATA_DIR=/home/exouser/Tabero/benchmarks/datasets/libero/USD \
    OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
    "$PY" -u "$WRAPPER" \
      --server-host 127.0.0.1 --server-port 18881 \
      --task-suite libero_10 --task-id "$task_id" \
      --language-instruction "$instruction" \
      --control-mode tactile --task Isaac-Libero-Franka-Hybrid-Tactile-v0 \
      --num-total-experiments 1 --max-inference-steps 50 --replan-steps 10 \
      --seed 7300 --no-randomize-light --b5-output-dir "$out" \
      --b5-method E3_NOMINAL_ROBUST_8N --b5-friction 0.6
}

case "$CANDIDATE" in
  3) run_one 3 akita_black_bowl_1 "put the black bowl in the bottom drawer of the cabinet and close it" ;;
  9) run_one 9 white_yellow_mug_1 "put the yellow and white mug in the microwave and close it" ;;
  5) run_one 5 black_book_1 "pick up the book and place it in the back compartment of the caddy" ;;
  *) echo "unsupported candidate: $CANDIDATE" >&2; exit 2 ;;
esac
