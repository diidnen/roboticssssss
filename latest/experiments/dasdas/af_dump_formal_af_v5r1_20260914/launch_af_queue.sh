#!/bin/bash
set -euo pipefail
BASE=/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911
HERE=/media/volume/dasdas/exouser/af_dump_formal_af_v5r1_20260914
REPO="$BASE/RoboTwin"
POLICY_PY="$REPO/XPolicyLab/policy/Pi_0/openpi/.venv/bin/python"
QUEUE_PY="$BASE/venv_robotwin/bin/python3"

mkdir -p "$HERE/server"
if pgrep -f 'af_dump_formal_af_v5r1_20260914/run_af_queue' >/dev/null; then
  echo "AF-v5r1 queue already live" >&2
  exit 1
fi

if ! ss -ltn | grep -q ':6001'; then
  echo "Starting frozen π0 policy server on :6001"
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
  echo "{\"pid\": $policy_pid, \"port\": 6001, \"owner\": \"af_dump_formal_af_v5r1_20260914\"}" > "$HERE/server/PID.json"
  "$REPO/XPolicyLab/utils/wait_for_policy_server.sh" localhost 6001 "$policy_pid" "pi0 AF-v5r1 formal" 360
else
  echo "π0 already listening on :6001"
fi

echo "Starting detached AF-v5r1 queue"
setsid env PYTHONUNBUFFERED=1 \
  "$QUEUE_PY" -u "$HERE/run_af_queue.py" \
  > "$HERE/queue.log" 2>&1 < /dev/null &
echo "af_v5r1_queue_pid=$!"
echo "{\"pid\": $!, \"started_utc\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" > "$HERE/QUEUE_PID.json"
sleep 2
pgrep -af 'run_af_queue|setup_policy_server' || true
tail -n 40 "$HERE/queue.log" || true
