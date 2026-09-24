#!/usr/bin/env bash
set -euo pipefail

if [[ "${E3_GPU_GATE_GO:-}" != "YES" ]]; then
  echo "FAIL_CLOSED: set E3_GPU_GATE_GO=YES only after coordinator MASS/GPU audit" >&2
  exit 2
fi

TABERO_ROOT=/media/volume/newdata/exouser/Tabero_e3lh
PYTHON_BIN=/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python
SOURCE_ROOT=/media/volume/newdata/exouser/activeforcing_e3/PROJECT_LIBERO_TASK5_DEMOS_20260902/assembled_hdf5
OUT_ROOT=/media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_REPLAY5_20260902_105400_RETRY1/smoke_demo1
OUT_HDF5=${OUT_ROOT}/replayed_demos/libero_10_task5_book_caddy_onboarding_demo01_7dpf_demo.hdf5
OUT_VIDEOS=${OUT_ROOT}/video_datasets
FAILURE_JSON=${OUT_ROOT}/failure_demo1.jsonl
LOG_FILE=${OUT_ROOT}/replay_demo1.log

if [[ -e "${OUT_ROOT}" ]]; then
  echo "FAIL_CLOSED: output root already exists and may not be reused: ${OUT_ROOT}" >&2
  exit 3
fi

gpu_line=$(nvidia-smi --query-gpu=utilization.gpu,memory.free --format=csv,noheader,nounits | head -n 1)
gpu_util=${gpu_line%%,*}
gpu_free=${gpu_line##*,}
gpu_util=${gpu_util// /}
gpu_free=${gpu_free// /}
if (( gpu_util >= 45 || gpu_free <= 18432 )); then
  echo "FAIL_CLOSED: tactile replay requires util<45% and free>18GiB; observed util=${gpu_util}% free=${gpu_free}MiB" >&2
  exit 4
fi

mkdir -p "${OUT_ROOT}/replayed_demos" "${OUT_VIDEOS}"

export HDF5_TRAJ_SOURCE_DIR="${SOURCE_ROOT}"
export OUTPUT_REPLAYED_DEMOS_DIR="${OUT_ROOT}/replayed_demos"
export OUTPUT_REPLAYED_VIDEOS_DIR="${OUT_VIDEOS}"
export REPLAYED_DEMOS_DIR="${OUT_ROOT}/replayed_demos"
export TABERO_FFMPEG_BIN=/media/volume/newdata/exouser/softvtbench/miniforge3/bin/ffmpeg
export USE_TABERO_TASKS=0

cd "${TABERO_ROOT}"
exec "${PYTHON_BIN}" scripts/tools/replay_demos_with_camera.py \
  --task Isaac-Libero-Franka-Replay-Camera-Tactile-v0 \
  --task_suite libero_10 \
  --task_id 5 \
  --demo_id 1 \
  --num_envs 1 \
  --video \
  --camera_view_list agentview eye_in_hand \
  --tactile_sensor_list gsmini_left gsmini_right \
  --tactile_output_type tactile_rgb \
  --recorder_type 7dpf \
  --dump_data \
  --output_file "${OUT_HDF5}" \
  --output_failure_record_file "${FAILURE_JSON}" \
  --headless 2>&1 | tee "${LOG_FILE}"
