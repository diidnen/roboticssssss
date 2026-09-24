#!/usr/bin/env bash
set -euo pipefail

if [[ "${E3_GPU_GATE_GO:-}" != "YES" ]]; then
  echo "FAIL_CLOSED: coordinator must set E3_GPU_GATE_GO=YES after fresh nvidia-smi/ps/pgrep audit" >&2
  exit 2
fi
if [[ $# -ne 2 ]]; then
  echo "usage: $0 IMMUTABLE_SUBSET_ROOT FRESH_OUTPUT_ROOT" >&2
  exit 2
fi

TABERO_ROOT=/media/volume/newdata/exouser/Tabero_e3lh
PYTHON_BIN=/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python
SUBSET_ROOT=$1
OUT_ROOT=$2
SOURCE_ROOT=${SUBSET_ROOT}/assembled_hdf5
INPUT_MANIFEST=${SUBSET_ROOT}/REMAINING4_INPUT_MANIFEST.json
OUT_HDF5=${OUT_ROOT}/replayed_demos/libero_10_task5_book_caddy_onboarding_remaining4_7dpf_demo.hdf5
OUT_VIDEOS=${OUT_ROOT}/video_datasets
FAILURE_JSON=${OUT_ROOT}/failure_remaining4.jsonl
LOG_FILE=${OUT_ROOT}/replay_remaining4.log
FFMPEG=/media/volume/newdata/exouser/softvtbench/miniforge3/bin/ffmpeg
REPLAY_UTILS=${TABERO_ROOT}/scripts/tools/common/replay_utils.py

if [[ ! -f "${INPUT_MANIFEST}" ]]; then
  echo "FAIL_CLOSED: missing frozen remaining4 input manifest: ${INPUT_MANIFEST}" >&2
  exit 3
fi
if [[ -e "${OUT_ROOT}" ]]; then
  echo "FAIL_CLOSED: output root already exists and may not be reused: ${OUT_ROOT}" >&2
  exit 3
fi
if [[ ! -x "${FFMPEG}" ]]; then
  echo "FAIL_CLOSED: frozen ffmpeg is missing or not executable: ${FFMPEG}" >&2
  exit 3
fi

"${PYTHON_BIN}" - "${INPUT_MANIFEST}" "${SOURCE_ROOT}" <<'PY'
import hashlib
import json
import pathlib
import sys

manifest_path = pathlib.Path(sys.argv[1]).resolve()
source_root = pathlib.Path(sys.argv[2]).resolve()
manifest = json.loads(manifest_path.read_text())
if manifest.get("status") != "REMAINING4_TRAIN_INPUT_SUBSET_FROZEN":
    raise SystemExit("FAIL_CLOSED: remaining4 input status is not frozen")
if manifest.get("selected_demo_ids") != [2, 11, 12, 19]:
    raise SystemExit("FAIL_CLOSED: batch must contain exactly IDs [2,11,12,19]")
if manifest.get("excluded_smoke_demo_id") != 1:
    raise SystemExit("FAIL_CLOSED: ID1 exclusion is not asserted")
if manifest.get("split") != "TRAIN" or manifest.get("task") != "libero_10/task5":
    raise SystemExit("FAIL_CLOSED: split/task mismatch")
path = pathlib.Path(manifest["subset_hdf5"]).resolve()
if path.parent != source_root or not path.is_file():
    raise SystemExit("FAIL_CLOSED: subset path/root mismatch")
digest = hashlib.sha256(path.read_bytes()).hexdigest()
if digest != manifest.get("subset_hdf5_sha256"):
    raise SystemExit("FAIL_CLOSED: immutable subset HDF5 hash mismatch")
PY

replay_utils_sha=$(sha256sum "${REPLAY_UTILS}" | awk '{print $1}')
ffmpeg_sha=$(sha256sum "${FFMPEG}" | awk '{print $1}')
if [[ "${replay_utils_sha}" != "bd9a831c34219fb5ae5f318863784b0daba6809d228fd1448f23fc941c9a6097" ]]; then
  echo "FAIL_CLOSED: patched replay_utils hash mismatch: ${replay_utils_sha}" >&2
  exit 3
fi
if [[ "${ffmpeg_sha}" != "8ca0469917dae545e734473175974de1aa1e500a1f5fb7f0208513cb023bf495" ]]; then
  echo "FAIL_CLOSED: frozen ffmpeg hash mismatch: ${ffmpeg_sha}" >&2
  exit 3
fi

gpu_line=$(nvidia-smi --query-gpu=utilization.gpu,memory.free --format=csv,noheader,nounits | head -n 1)
gpu_util=${gpu_line%%,*}
gpu_free=${gpu_line##*,}
gpu_util=${gpu_util// /}
gpu_free=${gpu_free// /}
if (( gpu_util >= 45 || gpu_free <= 18432 )); then
  echo "FAIL_CLOSED: batch requires util<45% and free>18GiB; util=${gpu_util}% free=${gpu_free}MiB" >&2
  exit 4
fi

mkdir -p "${OUT_ROOT}/replayed_demos" "${OUT_VIDEOS}"
{
  echo "input_manifest=${INPUT_MANIFEST}"
  echo "replay_utils_sha256=${replay_utils_sha}"
  echo "ffmpeg_sha256=${ffmpeg_sha}"
  echo "selected_demo_ids=2,11,12,19"
  echo "excluded_demo_id=1"
  echo "prelaunch_gpu_util_pct=${gpu_util}"
  echo "prelaunch_gpu_free_mib=${gpu_free}"
} > "${OUT_ROOT}/BATCH_PRELAUNCH_PROVENANCE.txt"

export HDF5_TRAJ_SOURCE_DIR="${SOURCE_ROOT}"
export OUTPUT_REPLAYED_DEMOS_DIR="${OUT_ROOT}/replayed_demos"
export OUTPUT_REPLAYED_VIDEOS_DIR="${OUT_VIDEOS}"
export REPLAYED_DEMOS_DIR="${OUT_ROOT}/replayed_demos"
export TABERO_FFMPEG_BIN="${FFMPEG}"
export USE_TABERO_TASKS=0

cd "${TABERO_ROOT}"
set +e
"${PYTHON_BIN}" scripts/tools/replay_demos_with_camera.py \
  --task Isaac-Libero-Franka-Replay-Camera-Tactile-v0 \
  --task_suite libero_10 \
  --task_id 5 \
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
sim_status=${PIPESTATUS[0]}
set -e
echo "${sim_status}" > "${OUT_ROOT}/SIMULATOR_EXIT_CODE.txt"
echo "Replay process ended with status ${sim_status}; accept only after strict batch QA." >&2
exit "${sim_status}"
