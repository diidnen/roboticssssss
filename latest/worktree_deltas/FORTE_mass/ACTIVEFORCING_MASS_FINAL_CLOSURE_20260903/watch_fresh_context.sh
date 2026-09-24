#!/usr/bin/env bash
set -u

ROOT_DIR=/home/exouser/FORTE_mass/ACTIVEFORCING_MASS_FINAL_CLOSURE_20260903
OUT="$1"
ROOT_ID="$2"
BAND="$3"
mkdir -p "$OUT"

protected_present() {
  local runner_pid="${RUNNER_PID:-0}"
  ps -eo pid=,args= | awk -v self="$$" -v runner="$runner_pid" '
    $1 != self && $1 != runner &&
    ($0 ~ /e6_second_query_pilot\.py --worker/ ||
     $0 ~ /ACTIVEFORCING_E5_FRESH_UTILITY_E2E_20260902_053006\/run_e5_fresh_utility\.py/ ||
     $0 ~ /run_selected_force_followup\.py --selection E6_POLICY_SELECTION_CONTEXT_EXPANSION\.csv --execute/ ||
     $0 ~ /activeforcing_e7_offgrid_rollout\.py --worker/ ||
     $0 ~ /run_e5_fresh_utility\.py --worker/) { found=1 }
    END { exit(found ? 0 : 1) }
  '
}

clear_ticks=0
while [ "$clear_ticks" -lt 6 ]; do
  if protected_present; then
    clear_ticks=0
    echo "waiting for protected Isaac worker before context ${ROOT_ID}_${BAND}" >&2
  else
    clear_ticks=$((clear_ticks + 1))
    echo "protected window clear ${clear_ticks}/6 for context ${ROOT_ID}_${BAND}" >&2
  fi
  sleep 5
done

export PYTHONNOUSERSITE=1
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export TABERO_ROOT=/home/exouser/Tabero
export HDF5_TRAJ_SOURCE_DIR=/home/exouser/Tabero/benchmarks/datasets/libero/assembled_hdf5
export LIBERO_ASSETS_DATA_DIR=/home/exouser/Tabero/benchmarks/datasets/libero/USD
export LIBERO_CONFIG_DIR=/home/exouser/Tabero/benchmarks/datasets/libero/config

"/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python" -u "$ROOT_DIR/run_mass_fresh_e2e_pi0.py" \
  --out "$OUT" --roots "$ROOT_ID" --bands "$BAND" \
  --scheduler-compat-marker run_e5_fresh_utility.py > "$OUT/run.log" 2>&1 &
RUNNER_PID=$!
echo "agent-owned runner pid=${RUNNER_PID} context=${ROOT_ID}_${BAND}" >&2

while kill -0 "$RUNNER_PID" 2>/dev/null; do
  if protected_present; then
    echo "protected Isaac worker detected; terminating only agent-owned runner pid=${RUNNER_PID}" >&2
    kill -TERM "$RUNNER_PID" 2>/dev/null || true
    sleep 3
    kill -KILL "$RUNNER_PID" 2>/dev/null || true
    wait "$RUNNER_PID" 2>/dev/null || true
    exit 99
  fi
  sleep 5
done
wait "$RUNNER_PID"
