#!/usr/bin/env bash
set -euo pipefail

GATE="${1:?fresh second-silent authorization JSON required}"
RUN_ROOT="${2:?fresh timestamped server run root required}"
if [[ "${E3_ROOT_GO:-}" != "YES" || "${E3_CORE_COORDINATOR_GO:-}" != "YES" || "${E3_FINAL_GATE_GO:-}" != "YES" ]]; then
  echo "FAIL_CLOSED: root, core coordinator, and V2 final-gate GO tokens are required" >&2
  exit 2
fi

ART=/home/exouser/FORTE_e3/ACTIVEFORCING_E3_LONGHORIZON_FULLTASK_LOCALLIFT_20260902_053000
OPENPI=/home/exouser/FORTE/agent_lanes/Tabero_VTLA_e3_onboarding_20260902
PYTHON=/media/volume/newdata/exouser/tabero/Tabero-VTLA/.venv/bin/python
DYNAMIC_VALIDATOR=/home/exouser/E3_E6_E7_LANES/E3_FULLTASK_VS_LOCALLIFT/validate_e3_coordinator_final_gate_dynamic.py
E5_ROOT=/home/exouser/FORTE/analysis/results/ACTIVEFORCING_E5_FRESH_UTILITY_E2E_20260902_053006
CANDIDATE=/media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_5DEMO_CANDIDATE_LOCK_20260902/CANDIDATE_LOCK.json
PORT=18883

"${PYTHON}" "${DYNAMIC_VALIDATOR}" "${GATE}" E3_TASK5_CANDIDATE_SERVER_START --e5-root "${E5_ROOT}"
[[ -f "${CANDIDATE}" ]] || { echo "FAIL_CLOSED: candidate lock missing" >&2; exit 3; }
[[ ! -e "${RUN_ROOT}" ]] || { echo "FAIL_CLOSED: server run root already exists" >&2; exit 3; }
if pgrep -af "${E5_ROOT}/run_e5_fresh_utility\\.py|${E5_ROOT}/core_gpu_scheduler\\.py" >/dev/null; then
  echo "FAIL_CLOSED: current E5 worker/scheduler detected after dynamic attestation" >&2
  exit 4
fi
if pgrep -af 'serve_task5_onboarded_candidate_pi0.py|--port 18883' >/dev/null; then
  echo "FAIL_CLOSED: duplicate candidate server detected" >&2
  exit 4
fi
if ss -ltn | awk '{print $4}' | grep -Eq '(^|:)18883$'; then
  echo "FAIL_CLOSED: port 18883 already bound" >&2
  exit 4
fi
gpu_line=$(nvidia-smi --query-gpu=utilization.gpu,memory.free --format=csv,noheader,nounits | head -n 1)
gpu_util=${gpu_line%%,*}; gpu_util=${gpu_util// /}
gpu_free=${gpu_line##*,}; gpu_free=${gpu_free// /}
if (( gpu_util >= 30 || gpu_free <= 28672 )); then
  echo "FAIL_CLOSED: dedicated candidate-server gate requires util<30 and free>28672MiB; util=${gpu_util} free=${gpu_free}" >&2
  exit 5
fi

mkdir -p "${RUN_ROOT}"
cp "${GATE}" "${RUN_ROOT}/COORDINATOR_FINAL_GATE_DYNAMIC.json"
cp "${CANDIDATE}" "${RUN_ROOT}/CANDIDATE_LOCK.json"
export PYTHONNOUSERSITE=1
export PYTHONPATH=${OPENPI}/src
cd "${OPENPI}"
exec "${PYTHON}" -u "${ART}/serve_task5_onboarded_candidate_pi0.py" \
  --port "${PORT}" --candidate-lock "${CANDIDATE}"
