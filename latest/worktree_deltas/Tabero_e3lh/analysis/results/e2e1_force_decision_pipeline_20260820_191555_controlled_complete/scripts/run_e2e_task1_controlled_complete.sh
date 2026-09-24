#!/usr/bin/env bash
set -euo pipefail

OUT=/home/exouser/Tabero/analysis/results/e2e1_force_decision_pipeline_20260820_191555_controlled_complete
PY=/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python

export E2E_OUT="$OUT"
export E2E_TASK_ID=1
export E2E_N_SEEDS=20
export E2E_SEED0=50
export E2E_MUS=0.2,0.5,1.0
export E2E_METHODS=fixed_low,fixed_robust,oracle,deligrasp_style,forte_gt,provisional_ours
export E2E_LOG_NAME=e2e_task1_controlled_complete.log
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export TERM=xterm-256color
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

"$PY" -u "$OUT/scripts/e2e_eval.py"
