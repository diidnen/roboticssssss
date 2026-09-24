#!/usr/bin/env bash
set -euo pipefail

if [[ "${E3_CONVERSION_GO:-}" != "YES" ]]; then
  echo "FAIL_CLOSED: set E3_CONVERSION_GO=YES only after all five independent replay gates pass" >&2
  exit 2
fi

ARTIFACT=/home/exouser/FORTE_e3/ACTIVEFORCING_E3_LONGHORIZON_FULLTASK_LOCALLIFT_20260902_053000
TABERO=/media/volume/newdata/exouser/Tabero_e3lh
PYTHON=/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python
LEROBOT_SITE=/media/volume/newdata/exouser/tabero/uv-cache/archive-v0/FiR5so0BgJNw-JER
OPENPI_SITE=/media/volume/newdata/exouser/softvtbench/openpi-venv/lib/python3.11/site-packages
ASSEMBLER=${ARTIFACT}/assemble_task5_replayed_five.py
CONVERTER_WRAPPER=${ARTIFACT}/convert_task5_5demo_7dpf_v21.sh
DATASET_VALIDATOR=${ARTIFACT}/validate_task5_onboarding_dataset.py
ASSEMBLED=/media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_REPLAY5_ASSEMBLED_20260902_113000
DATASET_BASE=/media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_LEROBOT_7DPF_V21_20260902_120000
DATASET=${DATASET_BASE}/activeforcing_e3_task5_5demo_7dpf

if [[ "$(git -C "${TABERO}" rev-parse HEAD)" != "f20281944f9771aa32c65745313a193a10665ad1" ]]; then
  echo "FAIL_CLOSED: isolated Tabero conversion commit changed" >&2
  exit 3
fi

"${PYTHON}" - "${ARTIFACT}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
expected = {
    1: "TASK5_PI0_ONBOARDING_ID1_RETRY_GATE.json",
    2: "TASK5_PI0_ONBOARDING_DEMO2_GATE.json",
    11: "TASK5_PI0_ONBOARDING_DEMO11_GATE.json",
    12: "TASK5_PI0_ONBOARDING_DEMO12_GATE.json",
    19: "TASK5_PI0_ONBOARDING_DEMO19_GATE.json",
}
for demo_id, name in expected.items():
    path = root / name
    if not path.is_file():
        raise SystemExit(f"FAIL_CLOSED: missing independent gate {path}")
    gate = json.loads(path.read_text())
    if demo_id == 1:
        passed = gate.get("status") == "ID1_REAL_TACTILE_7DPF_SMOKE_PASS" and gate.get("source_demo_id") == 1
    else:
        passed = (
            gate.get("gate_pass") is True
            and gate.get("errors") == []
            and gate.get("source_demo_id") == demo_id
            and gate.get("status") == f"DEMO{demo_id}_REAL_TACTILE_7DPF_GATE_PASS"
        )
    if not passed:
        raise SystemExit(f"FAIL_CLOSED: independent gate did not pass for demo {demo_id}")
PY

if [[ -e "${ASSEMBLED}" || -e "${DATASET}" ]]; then
  echo "FAIL_CLOSED: assembly or dataset output already exists; never overwrite" >&2
  exit 4
fi

export PYTHONNOUSERSITE=1
export JAX_PLATFORMS=cpu
"${PYTHON}" "${ASSEMBLER}"
bash "${CONVERTER_WRAPPER}"
PYTHONPATH="${LEROBOT_SITE}:${OPENPI_SITE}:${TABERO}" \
  "${PYTHON}" "${DATASET_VALIDATOR}" "${ASSEMBLED}" "${DATASET}"

if [[ ! -f "${DATASET}/E3_TASK5_5DEMO_DATASET_QA.json" ]]; then
  echo "FAIL_CLOSED: final dataset QA was not written" >&2
  exit 5
fi
echo "STRICT_5DEMO_ASSEMBLY_CONVERSION_DATASET_GATE_PASS"
