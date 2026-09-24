#!/bin/bash
# Detach Stage I from the Cursor tool shell so a session SIGTERM cannot
# again kill an in-progress matched context. Restarts the same frozen π0
# server that Stage I was already using.
set -euo pipefail

BASE=/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911
HERE="$BASE/experiments/af_motion_diversity_study_20260913"
REPO="$BASE/RoboTwin"
POLICY_PY="$REPO/XPolicyLab/policy/Pi_0/openpi/.venv/bin/python"
STUDY_PY="$BASE/venv_robotwin/bin/python3"

if pgrep -f 'af_motion_diversity_study_20260913/run_stage_i' >/dev/null; then
  echo "Stage I runner already live" >&2
  exit 1
fi

if ! ss -ltn | grep -q ':6001'; then
  echo "Starting frozen π0 policy server on :6001"
  setsid env PYTHONUNBUFFERED=1 \
    "$POLICY_PY" "$REPO/XPolicyLab/setup_policy_server.py" \
      --config_path "$REPO/XPolicyLab/policy/Pi_0/deploy.yml" \
      --host localhost --port 6001 --protocol ws \
      --overrides bench_name=RoboTwin task_name=handover_block \
      ckpt_name="$BASE/checkpoints/pi0_robotwin_30000/30000" \
      env_cfg_type=arx_x5 seed=0 policy_name=Pi_0 action_type=joint \
      action_dim=14 train_config_name=pi0_base_aloha_full_sim_arx-x5_seed_0 \
      repo_id=arx_x5_sim \
    > "$HERE/pi0_server_resume_v3.log" 2>&1 < /dev/null &
  policy_pid=$!
  "$REPO/XPolicyLab/utils/wait_for_policy_server.sh" localhost 6001 "$policy_pid" "pi0 Stage-I resume" 360
else
  echo "π0 already listening on :6001"
fi

echo "Starting detached Stage I resume v3"
setsid env PYTHONUNBUFFERED=1 \
  "$STUDY_PY" -u "$HERE/run_stage_i_resume_v3.py" \
  > "$HERE/stage_i_resume_v3.log" 2>&1 < /dev/null &
echo "resume_v3_pid=$!"
sleep 2
pgrep -af 'run_stage_i_resume_v3|setup_policy_server' || true
