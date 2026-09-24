#!/usr/bin/env bash
set -u

LOG=/home/exouser/activeforcing_table_push_20260910.log
BLOCKING_QUEUE_PID=272852
SESSION_ID=01a08b81-a8ad-7062-a0cd-6711aece72cc

printf '\n[SUPERVISOR V3] Waiting for the pre-existing noquery_established_v2 queue (PID %s) to exit naturally; no unrelated process will be stopped.\n' "$BLOCKING_QUEUE_PID" | tee -a "$LOG"
while kill -0 "$BLOCKING_QUEUE_PID" 2>/dev/null; do
  sleep 20
done

printf '\n[SUPERVISOR V3] No-Query queue exited. Resuming the same V3 research session at exact-state Gate R1.\n' | tee -a "$LOG"
cd /home/exouser/Tabero || exit 70
codex exec resume --dangerously-bypass-approvals-and-sandbox "$SESSION_ID" \
  "The pre-existing noquery_established_v2 queue has exited naturally. Resume the user-authorized V3 fix from its immutable artifacts. Treat every prior V3 policy-server attempt as infrastructure-only: verify current GPU ownership; delete only stale V3-owned PID/readiness files; use the known correct Tabero-VTLA OpenPI environment from V2; start and hold one V3-owned official frozen pi0_libero server; require a real inference smoke response before the canonical prefix. Then execute Gate R1 exactly as frozen: one saved canonical prefix, exact replay/state/controller/contact/observation parity, clipping-free one-step branches, and post-env.step wrench measurement. Continue to R2/A3 only on explicit gate passes. Do not stop unrelated jobs, alter V1/V2, weaken gates, or start 216 branches prematurely. Stream concise evidence-first progress to the same log." \
  2>&1 | tee -a "$LOG"
