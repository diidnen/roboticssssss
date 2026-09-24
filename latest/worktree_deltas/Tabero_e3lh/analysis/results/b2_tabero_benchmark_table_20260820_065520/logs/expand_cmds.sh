#!/usr/bin/env bash
set -euo pipefail
B2=/home/exouser/Tabero/analysis/results/b2_tabero_benchmark_table_20260820_065520
PY=/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python
echo "expand task=1 F=4,6 mu=0.2,0.5 seed0=5 n=15"
B2_OUT="$B2" B2_TASK_ID=1 B2_CSV=TASK1_ORACLE_SCAN.csv B2_PHASE=EXPAND B2_N_SEEDS=15 B2_SEED0=5 B2_FORCES=4,6 B2_MUS=0.2,0.5 B2_LOG_NAME=b2_expand_t1.log OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y "$PY" -u "$B2/scripts/b2_collect.py"
