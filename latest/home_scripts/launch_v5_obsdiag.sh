#!/usr/bin/env bash
set -e
export MUJOCO_GL=egl
export LIBERO_CONFIG_PATH=/home/exouser/Tabero/analysis/results/pi0_nontransport_20260910/libero_config
export PYTHONPATH=/media/volume/newdata/exouser/flowdagger_e960a/bundle/e959c_bootstrap_resolved_r0_20260817T082827Z/flowdagger/flowdagger_pi05/openpi/third_party/libero:/home/exouser/Tabero/analysis/results/pi0_nontransport_20260910/client_venv/lib/python3.10/site-packages
V=/media/volume/newdata/exouser/flowdagger_e960a/e965_sirius_retrospective_correction_efficiency/env/bin/python
S=/media/volume/data/exouser/activeforcing_table_push_v5_20260910/code/r2h_branch_obsdiag.py
"$V" "$S" 55 untouched 0 serial
"$V" "$S" 55 disabled 0 serial
