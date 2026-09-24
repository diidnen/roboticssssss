#!/usr/bin/env bash
set -o pipefail
cd /home/exouser/Tabero || exit 2
codex exec resume --dangerously-bypass-approvals-and-sandbox 01a08b81-a8ad-7062-a0cd-6711aece72cc - \
  < /home/exouser/activeforcing_table_push_v3_r2_prompt.md \
  2>&1 | tee -a /home/exouser/activeforcing_table_push_20260910.log
