#!/usr/bin/env bash
set -u

LOG=/home/exouser/activeforcing_table_push_20260910.log
BLOCKING_LAUNCHER_PID=254337
SESSION_ID=01a08b07-a395-7303-8791-204970166424

printf '\n[SUPERVISOR] Gate A is queued behind the existing No-Query task1/5/6 launcher (PID %s). No unrelated job will be interrupted.\n' "$BLOCKING_LAUNCHER_PID" | tee -a "$LOG"
while kill -0 "$BLOCKING_LAUNCHER_PID" 2>/dev/null; do
  sleep 20
done

printf '\n[SUPERVISOR] The blocking No-Query launcher exited. Resuming the same ActiveForcing session now.\n' | tee -a "$LOG"
cd /home/exouser/Tabero || exit 70
codex exec resume --dangerously-bypass-approvals-and-sandbox "$SESSION_ID" \
  "The pre-existing No-Query task1/5/6 launcher has now exited naturally. Resume this table-push ActiveForcing work from the preserved Gate A infrastructure failure. Re-check current GPU ownership and capacity without stopping unrelated processes. Remove only stale server-ready/PID markers owned by this ActiveForcing namespace, start one official frozen pi0_libero server using the correct OpenPI environment, verify a real inference response before rollout, and rerun the paired root-0 calibration. Preserve the prior infrastructure failure but do not count it as a policy rollout. Continue autonomously under every original gate, no-overclaim, split, and storage constraint. Stream evidence-first progress to the same log." \
  2>&1 | tee -a "$LOG"
