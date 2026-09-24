#!/bin/bash
set -euo pipefail

BASE=/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911
HERE="$BASE/experiments/af_dump_fixed_force_5_15_20260914"
REPO="$BASE/RoboTwin"
DAS=/media/volume/dasdas/exouser/af_dump_fixed_force_5_15_20260914
POLICY_PY="$REPO/XPolicyLab/policy/Pi_0/openpi/.venv/bin/python"
QUEUE_PY="$BASE/venv_robotwin/bin/python3"

mkdir -p "$HERE/server" "$DAS"
for name in smoke_raw smoke_records smoke_kept main_raw main_records main_kept; do
  mkdir -p "$DAS/$name"
  if [[ -e "$HERE/$name" && ! -L "$HERE/$name" ]]; then
    if [[ -d "$HERE/$name" && -z "$(ls -A "$HERE/$name")" ]]; then
      rmdir "$HERE/$name"
    else
      echo "Refuse to replace nonempty local path: $HERE/$name" >&2
      exit 2
    fi
  fi
  ln -sfn "$DAS/$name" "$HERE/$name"
done

if pgrep -f "$HERE/run_fixed_sweep_queue.py" >/dev/null; then
  echo "Fixed-force sweep queue already live" >&2
  exit 1
fi

if ! ss -ltn | grep -q ':6001'; then
  setsid env PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 \
    "$POLICY_PY" "$REPO/XPolicyLab/setup_policy_server.py" \
      --config_path "$REPO/XPolicyLab/policy/Pi_0/deploy.yml" \
      --host localhost --port 6001 --protocol ws \
      --overrides bench_name=RoboTwin task_name=handover_block \
      ckpt_name="$BASE/checkpoints/pi0_robotwin_30000/30000" \
      env_cfg_type=arx_x5 seed=0 policy_name=Pi_0 action_type=joint \
      action_dim=14 train_config_name=pi0_base_aloha_full_sim_arx-x5_seed_0 \
      repo_id=arx_x5_sim \
    > "$HERE/server/server.log" 2>&1 < /dev/null &
  policy_pid=$!
  echo "{\"pid\": $policy_pid, \"port\": 6001, \"owner\": \"af_dump_fixed_force_5_15_20260914\"}" > "$HERE/server/PID.json"
  "$REPO/XPolicyLab/utils/wait_for_policy_server.sh" localhost 6001 "$policy_pid" "pi0 fixed-force sweep" 360
fi

setsid env PYTHONUNBUFFERED=1 "$QUEUE_PY" -u "$HERE/run_fixed_sweep_queue.py" \
  > "$HERE/queue.log" 2>&1 < /dev/null &
queue_pid=$!
echo "{\"pid\": $queue_pid, \"started_utc\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" > "$HERE/QUEUE_PID.json"
sleep 2
ps -p "$queue_pid" -o pid,etimes,cmd
tail -n 30 "$HERE/queue.log" || true
