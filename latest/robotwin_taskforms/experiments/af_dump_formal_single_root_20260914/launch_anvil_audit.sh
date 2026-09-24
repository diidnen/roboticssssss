#!/usr/bin/env bash
set -euo pipefail

ANVIL_ROOT=/anvil/projects/x-cis250966/tabero-transfer/jetstream-activeforcing-20260912/formal_single_root_20260914
ssh -i /home/exouser/.ssh/codex_anvil_migration_20260912 \
  -o BatchMode=yes \
  -o ServerAliveInterval=30 \
  x-csong7@anvil.rcac.purdue.edu \
  "cd '${ANVIL_ROOT}/audit' && rm -f ANVIL_AUDIT_EXIT_STATUS.txt && module load python/3.9.5 && (python3 anvil_formal_trace_audit.py FINAL_INDEPENDENT_AUDIT.json --output ANVIL_FORMAL_TRACE_AUDIT.json > anvil_trace_audit.log 2>&1; rc=\$?; printf '%s\n' \"\$rc\" > ANVIL_AUDIT_EXIT_STATUS.txt; exit \"\$rc\")"
