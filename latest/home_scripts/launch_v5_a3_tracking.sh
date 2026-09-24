#!/usr/bin/env bash
set -u
V=/media/volume/newdata/exouser/flowdagger_e960a/e965_sirius_retrospective_correction_efficiency/env/bin/python
P=/media/volume/newdata/exouser/flowdagger_e960a/bundle/e959c_bootstrap_resolved_r0_20260817T082827Z/flowdagger/flowdagger_pi05/openpi/third_party/libero:/home/exouser/Tabero/analysis/results/pi0_nontransport_20260910/client_venv/lib/python3.10/site-packages
O=/media/volume/data/exouser/activeforcing_table_push_v5_20260910
export PYTHONPATH="$P:$O/code"
export MUJOCO_GL=egl
export LIBERO_CONFIG_PATH=/home/exouser/Tabero/analysis/results/pi0_nontransport_20260910/libero_config
failed=0
for target in 19.5 20.5 21.5; do
  for mode in feedback baseline; do
    "$V" "$O/code/a3_tracking_branch.py" "$target" "$mode" >> "$O/A3_TRACKING_RUN.log" 2>&1 &
    while [ "$(jobs -pr | wc -l)" -ge 3 ]; do
      wait -n || failed=1
    done
  done
done
wait || failed=1
if [ "$failed" -ne 0 ]; then
  exit 1
fi
"$V" "$O/code/aggregate_a3_tracking.py" >> "$O/A3_TRACKING_RUN.log" 2>&1
