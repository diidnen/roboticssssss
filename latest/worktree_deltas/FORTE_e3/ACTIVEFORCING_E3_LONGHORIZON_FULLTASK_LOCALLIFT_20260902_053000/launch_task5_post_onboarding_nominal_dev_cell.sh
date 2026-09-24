#!/usr/bin/env bash
set -euo pipefail

GATE="${1:?fresh per-cell second-silent authorization JSON required}"
OUT_ROOT="${2:?fresh nominal DEV output root required}"
ROOT_SEED="${3:?one frozen nominal DEV root required}"
if [[ "${E3_ROOT_GO:-}" != "YES" || "${E3_CORE_COORDINATOR_GO:-}" != "YES" || "${E3_FINAL_GATE_GO:-}" != "YES" ]]; then
  echo "FAIL_CLOSED: root, core coordinator, and V2 final-gate GO tokens are required" >&2
  exit 2
fi
case "${ROOT_SEED}" in
  7600|7601|7602|7603|7604) ;;
  *) echo "FAIL_CLOSED: root outside preregistered nominal DEV set" >&2; exit 2 ;;
esac

ART=/home/exouser/FORTE_e3/ACTIVEFORCING_E3_LONGHORIZON_FULLTASK_LOCALLIFT_20260902_053000
TABERO=/media/volume/newdata/exouser/Tabero_e3lh
ISAAC_PY=/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python
DYNAMIC_VALIDATOR=/home/exouser/E3_E6_E7_LANES/E3_FULLTASK_VS_LOCALLIFT/validate_e3_coordinator_final_gate_dynamic.py
E5_ROOT=/home/exouser/FORTE/analysis/results/ACTIVEFORCING_E5_FRESH_UTILITY_E2E_20260902_053006
OPENPI_CLIENT=${TABERO}/benchmarks/openpi/openpi-client/src
WARP_CORE=/media/volume/newdata/exouser/tabero/env_isaaclab51/lib/python3.11/site-packages/isaacsim/extscache/omni.warp.core-1.8.2+lx64
RAW_DATA=/media/volume/newdata/exouser/tabero/data/Isaaclab_Libero
CANDIDATE=/media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_5DEMO_CANDIDATE_LOCK_20260902/CANDIDATE_LOCK.json
CELL=${OUT_ROOT}/NOMINAL_DEV_root${ROOT_SEED}_mu0.6_F8N

"${ISAAC_PY}" "${DYNAMIC_VALIDATOR}" "${GATE}" E3_TASK5_NOMINAL_DEV_CELL --root-seed "${ROOT_SEED}" --e5-root "${E5_ROOT}"
[[ -f "${CANDIDATE}" ]] || { echo "FAIL_CLOSED: candidate lock missing" >&2; exit 3; }
[[ -d "${RAW_DATA}" ]] || { echo "FAIL_CLOSED: immutable Isaac data missing" >&2; exit 3; }
[[ ! -e "${CELL}" ]] || { echo "FAIL_CLOSED: nominal DEV cell already exists" >&2; exit 3; }
if pgrep -af "${E5_ROOT}/run_e5_fresh_utility\\.py|${E5_ROOT}/core_gpu_scheduler\\.py" >/dev/null; then
  echo "FAIL_CLOSED: current E5 worker/scheduler detected after dynamic attestation" >&2
  exit 4
fi
if pgrep -af "NOMINAL_DEV_root${ROOT_SEED}|--seed ${ROOT_SEED}.*E3_POST_ONBOARDING_NOMINAL_DEV" >/dev/null; then
  echo "FAIL_CLOSED: duplicate nominal DEV cell detected" >&2
  exit 4
fi
if ! ss -ltn | awk '{print $4}' | grep -Eq '(^|:)18883$'; then
  echo "FAIL_CLOSED: locked candidate server port 18883 is not listening" >&2
  exit 4
fi
gpu_line=$(nvidia-smi --query-gpu=utilization.gpu,memory.free --format=csv,noheader,nounits | head -n 1)
gpu_util=${gpu_line%%,*}; gpu_util=${gpu_util// /}
gpu_free=${gpu_line##*,}; gpu_free=${gpu_free// /}
if (( gpu_util >= 50 || gpu_free <= 18432 )); then
  echo "FAIL_CLOSED: per-cell gate requires util<50 and free>18432MiB; util=${gpu_util} free=${gpu_free}" >&2
  exit 5
fi

fingerprint=$("${ISAAC_PY}" - "${CANDIDATE}" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
assert d["status"] == "CANDIDATE_LOCKED_FOR_NOMINAL_DEV_NOT_FINAL_FREEZE"
assert d["selected_step"] == 999
print(d["checkpoint_tree_sha256"])
PY
)
lock_sha=$(sha256sum "${CANDIDATE}" | awk '{print $1}')
gate_sha=$(sha256sum "${GATE}" | awk '{print $1}')
mkdir -p "${CELL}"
cp "${GATE}" "${CELL}/COORDINATOR_FINAL_GATE_DYNAMIC.json"
cp "${CANDIDATE}" "${CELL}/CANDIDATE_LOCK.json"

E3_OBJECT_NAME=black_book_1 E3_FORCE_N=8 \
  E3_POLICY_FINGERPRINT="${fingerprint}" E3_CANDIDATE_LOCK_SHA256="${lock_sha}" \
  E3_COORDINATOR_DYNAMIC_GATE_SHA256="${gate_sha}" \
  TABERO_ROOT="${TABERO}" PYTHONNOUSERSITE=1 PYTHONPATH="${WARP_CORE}:${TABERO}:${OPENPI_CLIENT}" \
  HDF5_TRAJ_SOURCE_DIR="${RAW_DATA}/assembled_hdf5" \
  LIBERO_CONFIG_DIR="${TABERO}/benchmarks/datasets/libero/config" \
  LIBERO_ASSETS_DATA_DIR="${RAW_DATA}/USD" OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
  "${ISAAC_PY}" -u "${ART}/run_e3_post_onboarding_nominal_wrapper.py" \
    --server-host 127.0.0.1 --server-port 18883 \
    --task-suite libero_10 --task-id 5 \
    --language-instruction "pick up the book and place it in the back compartment of the caddy" \
    --control-mode tactile --task Isaac-Libero-Franka-Hybrid-Tactile-v0 \
    --num-total-experiments 1 --max-inference-steps 50 --replan-steps 10 \
    --seed "${ROOT_SEED}" --no-randomize-light \
    --b5-output-dir "${CELL}" --b5-method E3_POST_ONBOARDING_NOMINAL_DEV --b5-friction 0.6
