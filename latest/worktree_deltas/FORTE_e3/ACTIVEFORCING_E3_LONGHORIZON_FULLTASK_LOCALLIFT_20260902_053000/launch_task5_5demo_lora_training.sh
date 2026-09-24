#!/usr/bin/env bash
set -euo pipefail

if [[ "${E3_PI0_TRAINING_GO:-}" != "YES" ]]; then
  echo "FAIL_CLOSED: explicit E3_PI0_TRAINING_GO=YES is required" >&2
  exit 2
fi

OPENPI=/home/exouser/FORTE/agent_lanes/Tabero_VTLA_e3_onboarding_20260902
ARTIFACT=/home/exouser/FORTE_e3/ACTIVEFORCING_E3_LONGHORIZON_FULLTASK_LOCALLIFT_20260902_053000
PYTHON=/media/volume/newdata/exouser/tabero/Tabero-VTLA/.venv/bin/python
LEROBOT_SITE=/media/volume/newdata/exouser/tabero/uv-cache/archive-v0/FiR5so0BgJNw-JER
OPENPI_SITE=/media/volume/newdata/exouser/softvtbench/openpi-venv/lib/python3.11/site-packages
CONFIG=pi0_lora_tacfield_e3_task5_5demo_7dpf
EXP=task5_5demo_7dpf_20260902_120000
DATASET_BASE=/media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_LEROBOT_7DPF_V21_20260902_120000
DATASET=${DATASET_BASE}/activeforcing_e3_task5_5demo_7dpf
QA=${DATASET}/E3_TASK5_5DEMO_DATASET_QA.json
ASSETS=/media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_7DPF_ASSETS_20260902_113000
NORM=${ASSETS}/${CONFIG}/activeforcing_e3_task5_5demo_7dpf/norm_stats.json
CHECKPOINT=/media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_7DPF_CHECKPOINTS_20260902_113000/${CONFIG}/${EXP}
BASE=/media/volume/newdata/exouser/tabero/models/pi0_lora_tacfield_tabero/checkpoints/pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999

if [[ "$(git -C "${OPENPI}" rev-parse HEAD)" != "31049447d685cb36ddaeddda4f1d62fec0bc6392" ]]; then
  echo "FAIL_CLOSED: isolated OpenPI onboarding commit changed" >&2
  exit 3
fi
if [[ "$(sha256sum "${OPENPI}/src/openpi/training/config.py" | awk '{print $1}')" != "296b01a51c985583bab688402808296315bd449ddd2aa343f4f5458ba1d1295e" ]]; then
  echo "FAIL_CLOSED: frozen onboarding config hash mismatch" >&2
  exit 3
fi
if [[ "$(sha256sum "${OPENPI}/src/openpi/policies/libero_policy.py" | awk '{print $1}')" != "f5eb0161b831c1f4a65b763f24183f5333a3efe54ae0537088a9450fdd54781a" ]]; then
  echo "FAIL_CLOSED: frozen onboarding policy transform hash mismatch" >&2
  exit 3
fi
if [[ -e "${CHECKPOINT}" ]]; then
  echo "FAIL_CLOSED: checkpoint output exists and may not be overwritten: ${CHECKPOINT}" >&2
  exit 3
fi
if [[ "$(sha256sum "${BASE}/_CHECKPOINT_METADATA" | awk '{print $1}')" != "f8519027dab6cf8eb113771fbc04d78042a0fb2b38bf770ce4c07866d2b6e022" ]]; then
  echo "FAIL_CLOSED: frozen base checkpoint metadata hash mismatch" >&2
  exit 3
fi
if [[ "$(sha256sum "${BASE}/params/_METADATA" | awk '{print $1}')" != "75842cbbd11926877a6d751bb9203814988f7c010ba889b995e3f628e173a458" ]]; then
  echo "FAIL_CLOSED: frozen base parameter metadata hash mismatch" >&2
  exit 3
fi
if [[ "$(sha256sum "${BASE}/params/manifest.ocdbt" | awk '{print $1}')" != "7007bb67166d1f4a845cdc04054a4ca17e9b9f04810048cb1442daca831fa201" ]]; then
  echo "FAIL_CLOSED: frozen base parameter manifest hash mismatch" >&2
  exit 3
fi
"${PYTHON}" - "${QA}" "${NORM}" <<'PY'
import json
import pathlib
import sys
qa_path, norm_path = map(pathlib.Path, sys.argv[1:])
if not qa_path.is_file() or not norm_path.is_file():
    raise SystemExit("FAIL_CLOSED: dataset QA or norm stats missing")
qa = json.loads(qa_path.read_text())
if qa.get("status") != "PASS" or qa.get("errors") != [] or qa.get("selected_demo_ids") != [1, 2, 11, 12, 19]:
    raise SystemExit("FAIL_CLOSED: five-demo dataset gate did not pass")
outer = json.loads(norm_path.read_text())
stats = outer.get("norm_stats", {})
expected_dims = {"state": 7, "actions": 13, "tactile_prefix": 396}
if set(stats) != set(expected_dims):
    raise SystemExit("FAIL_CLOSED: normalization schema mismatch")
for name, dim in expected_dims.items():
    if any(len(stats[name].get(field, [])) != dim for field in ("mean", "std", "q01", "q99")):
        raise SystemExit(f"FAIL_CLOSED: normalization dimension mismatch for {name}")
PY

gpu_line=$(nvidia-smi --query-gpu=utilization.gpu,memory.free --format=csv,noheader,nounits | head -n 1)
gpu_util=${gpu_line%%,*}; gpu_util=${gpu_util// /}
gpu_free=${gpu_line##*,}; gpu_free=${gpu_free// /}
if (( gpu_util >= 30 || gpu_free <= 28672 )); then
  echo "FAIL_CLOSED: LoRA training requires a dedicated window (util<30%, free>28GiB); util=${gpu_util}% free=${gpu_free}MiB" >&2
  exit 4
fi

export PYTHONNOUSERSITE=1
export PYTHONPATH=${OPENPI}/src:${LEROBOT_SITE}:${OPENPI_SITE}
export HF_LEROBOT_HOME=${DATASET_BASE}
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70
cd "${OPENPI}"
"${PYTHON}" "${ARTIFACT}/validate_task5_5demo_lora_preflight.py"
exec "${PYTHON}" scripts/train.py "${CONFIG}" --exp-name="${EXP}"
