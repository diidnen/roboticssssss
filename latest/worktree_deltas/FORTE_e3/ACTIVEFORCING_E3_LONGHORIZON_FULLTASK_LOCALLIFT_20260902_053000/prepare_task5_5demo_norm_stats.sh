#!/usr/bin/env bash
set -euo pipefail

if [[ "${E3_NORM_STATS_GO:-}" != "YES" ]]; then
  echo "FAIL_CLOSED: set E3_NORM_STATS_GO=YES only after strict five-demo dataset QA" >&2
  exit 2
fi

OPENPI=/home/exouser/FORTE/agent_lanes/Tabero_VTLA_e3_onboarding_20260902
ARTIFACT=/home/exouser/FORTE_e3/ACTIVEFORCING_E3_LONGHORIZON_FULLTASK_LOCALLIFT_20260902_053000
PYTHON=/media/volume/newdata/exouser/tabero/Tabero-VTLA/.venv/bin/python
LEROBOT_SITE=/media/volume/newdata/exouser/tabero/uv-cache/archive-v0/FiR5so0BgJNw-JER
OPENPI_SITE=/media/volume/newdata/exouser/softvtbench/openpi-venv/lib/python3.11/site-packages
CONFIG=pi0_lora_tacfield_e3_task5_5demo_7dpf
DATASET_BASE=/media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_LEROBOT_7DPF_V21_20260902_120000
DATASET=${DATASET_BASE}/activeforcing_e3_task5_5demo_7dpf
QA=${DATASET}/E3_TASK5_5DEMO_DATASET_QA.json
ASSETS=/media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_7DPF_ASSETS_20260902_113000
NORM=${ASSETS}/${CONFIG}/activeforcing_e3_task5_5demo_7dpf/norm_stats.json

if [[ "$(git -C "${OPENPI}" rev-parse HEAD)" != "31049447d685cb36ddaeddda4f1d62fec0bc6392" ]]; then
  echo "FAIL_CLOSED: isolated OpenPI onboarding commit changed" >&2
  exit 3
fi
if [[ "$(sha256sum "${OPENPI}/src/openpi/training/config.py" | awk '{print $1}')" != "296b01a51c985583bab688402808296315bd449ddd2aa343f4f5458ba1d1295e" ]]; then
  echo "FAIL_CLOSED: frozen onboarding config hash mismatch" >&2
  exit 3
fi
if [[ -e "${ASSETS}" ]]; then
  echo "FAIL_CLOSED: assets output already exists and may not be overwritten: ${ASSETS}" >&2
  exit 3
fi
"${PYTHON}" - "${QA}" <<'PY'
import json
import pathlib
import sys
path = pathlib.Path(sys.argv[1])
if not path.is_file():
    raise SystemExit("FAIL_CLOSED: strict dataset QA missing")
qa = json.loads(path.read_text())
if qa.get("status") != "PASS" or qa.get("errors") != [] or qa.get("selected_demo_ids") != [1, 2, 11, 12, 19]:
    raise SystemExit("FAIL_CLOSED: strict dataset QA did not pass")
PY

export JAX_PLATFORMS=cpu
export PYTHONNOUSERSITE=1
export PYTHONPATH=${OPENPI}/src:${LEROBOT_SITE}:${OPENPI_SITE}
export HF_LEROBOT_HOME=${DATASET_BASE}
cd "${OPENPI}"
"${PYTHON}" "${ARTIFACT}/validate_task5_5demo_lora_preflight.py"
"${PYTHON}" scripts/compute_norm_stats.py --config-name "${CONFIG}"
if [[ ! -f "${NORM}" ]]; then
  echo "FAIL_CLOSED: normalization statistics missing after computation" >&2
  exit 4
fi
"${PYTHON}" - "${NORM}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
outer = json.loads(path.read_text())
if set(outer) != {"norm_stats"}:
    raise SystemExit(f"FAIL_CLOSED: normalization envelope keys {sorted(outer)}")
data = outer["norm_stats"]
expected_dims = {"state": 7, "actions": 13, "tactile_prefix": 396}
if set(data) != set(expected_dims):
    raise SystemExit(f"FAIL_CLOSED: normalization keys {sorted(data)} != {sorted(expected_dims)}")
for name, dim in expected_dims.items():
    if set(data[name]) != {"mean", "std", "q01", "q99"}:
        raise SystemExit(f"FAIL_CLOSED: incomplete statistics for {name}")
    if any(len(data[name][field]) != dim for field in ("mean", "std", "q01", "q99")):
        raise SystemExit(f"FAIL_CLOSED: wrong normalization dimension for {name}")
PY
sha256sum "${NORM}"
