#!/usr/bin/env bash
set -euo pipefail

ASSEMBLED=/media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_REPLAY5_ASSEMBLED_20260902_113000
OUTPUT_BASE=/media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_LEROBOT_7DPF_V21_20260902_120000
OUTPUT_DATASET=${OUTPUT_BASE}/activeforcing_e3_task5_5demo_7dpf
CONVERTER=/media/volume/newdata/exouser/Tabero_e3lh/benchmarks/common/convert_all_libero_to_tabero.py
PYTHON_BIN=/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python

if [[ ! -f "${ASSEMBLED}/ASSEMBLED_5DEMO_MANIFEST.json" ]]; then
  echo "FAIL_CLOSED: five-demo assembly manifest missing" >&2
  exit 2
fi
if [[ -e "${OUTPUT_DATASET}" ]]; then
  echo "FAIL_CLOSED: output dataset already exists and may not be overwritten" >&2
  exit 3
fi

export PYTHONNOUSERSITE=1
export JAX_PLATFORMS=cpu
export PYTHONPATH=/media/volume/newdata/exouser/tabero/uv-cache/archive-v0/FiR5so0BgJNw-JER:/media/volume/newdata/exouser/softvtbench/openpi-venv/lib/python3.11/site-packages:/media/volume/newdata/exouser/Tabero_e3lh

exec "${PYTHON_BIN}" "${CONVERTER}" \
  --task-suites libero_10 \
  --task-ids 5 \
  --single-source \
  --data-root "${ASSEMBLED}" \
  --hdf5-folder "${ASSEMBLED}/replayed_demos" \
  --video-dir "${ASSEMBLED}/video_datasets" \
  --output-dir "${OUTPUT_BASE}" \
  --repo-name activeforcing_e3_task5_5demo_7dpf
