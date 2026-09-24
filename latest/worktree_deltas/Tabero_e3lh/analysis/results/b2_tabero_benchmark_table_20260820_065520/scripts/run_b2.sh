#!/usr/bin/env bash
# B2 orchestrator. Isolated from D2: wait only, never kill.
set -euo pipefail

B2=/home/exouser/Tabero/analysis/results/b2_tabero_benchmark_table_20260820_065520
PY=/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python
COLLECT="$B2/scripts/b2_collect.py"
ANALYZE="$B2/scripts/b2_analyze.py"
WRAP="$B2/scripts/b2_wrapup.py"
LOG="$B2/logs/runner.log"
D2_PID="${D2_PID:-966250}"

mkdir -p "$B2/logs" "$B2/plots"
exec >>"$LOG" 2>&1

echo "==== B2 runner start $(date -Is) pid=$$ ===="
echo "will wait for D2 pid $D2_PID; will not kill it"

wait_for_d2() {
  while kill -0 "$D2_PID" 2>/dev/null; do
    echo "$(date -Is) waiting for D2 pid $D2_PID (GPU isolation)"
    sleep 45
  done
  # Extra: do not start while another isaaclab51 Tabero process still owns GPU.
  local n=0
  while true; do
    local hit
    hit=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null \
      | grep -F "env_isaaclab51/bin/python" || true)
    if [ -z "$hit" ]; then
      echo "$(date -Is) isaaclab51 GPU slot free"
      break
    fi
    echo "$(date -Is) isaaclab51 still on GPU: $hit"
    n=$((n+1))
    if [ "$n" -gt 120 ]; then
      echo "timeout waiting for isaaclab51 GPU slot; continuing if D2 is gone"
      break
    fi
    sleep 30
  done
}

run_task() {
  local task="$1" csv="$2" phase="$3" nseeds="$4" seed0="$5" forces="$6" mus="$7" logname="$8"
  echo "==== collect task=$task phase=$phase n=$nseeds seed0=$seed0 F=$forces mu=$mus $(date -Is) ===="
  B2_OUT="$B2" \
  B2_TASK_ID="$task" \
  B2_CSV="$csv" \
  B2_PHASE="$phase" \
  B2_N_SEEDS="$nseeds" \
  B2_SEED0="$seed0" \
  B2_FORCES="$forces" \
  B2_MUS="$mus" \
  B2_LOG_NAME="$logname" \
  OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
  "$PY" -u "$COLLECT"
}

wait_for_d2

# Phase A: task 1 then task 7. Integer force grid, N=5.
run_task 1 TASK1_ORACLE_SCAN.csv A 5 0 "3,4,5,6,8" "0.2,0.5,1.0" b2_task1_phaseA.log
run_task 7 TASK7_ORACLE_SCAN.csv A 5 0 "3,4,5,6,8" "0.2,0.5,1.0" b2_task7_phaseA.log

python3 "$ANALYZE" || true
python3 "$WRAP" --stage phaseA || true
echo "PHASE_A_ISAAC_DONE $(date -Is)"

# Expand transition cells to N=20 matched seeds (seed 0-19) without expanding all-fail/all-succeed.
python3 "$B2/scripts/b2_plan_expand.py" | tee "$B2/logs/expand_plan.txt"
if [ -f "$B2/logs/expand_cmds.sh" ]; then
  bash "$B2/logs/expand_cmds.sh"
fi

python3 "$ANALYZE" || true
python3 "$WRAP" --stage phaseA_expand || true
echo "PHASE_A_EXPAND_DONE $(date -Is)"

# Phase B cheap scan for remaining official tasks 0,2,3,5,6,8,9
# (tasks 1 and 7 already have denser Phase A coverage; still write a cheap extract)
for t in 0 2 3 5 6 8 9; do
  run_task "$t" "ALL_TASK_CHEAP_SCAN.csv" B 3 0 "4,5,6" "0.2,0.5,1.0" "b2_cheap_task${t}.log"
done

python3 "$ANALYZE" || true
python3 "$WRAP" --stage phaseB || true
echo "PHASE_B_CHEAP_DONE $(date -Is)"

# Expand only cheap-scan POSITIVE tasks to N=15 (seeds 0-14) on 4,5,6 (+8 if needed)
python3 "$B2/scripts/b2_plan_expand.py" --positive-only | tee "$B2/logs/expand_positive_plan.txt"
if [ -f "$B2/logs/expand_positive_cmds.sh" ]; then
  bash "$B2/logs/expand_positive_cmds.sh"
fi

python3 "$ANALYZE" || true
python3 "$WRAP" --stage final
echo "==== B2 runner complete $(date -Is) ===="
