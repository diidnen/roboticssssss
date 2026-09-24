#!/usr/bin/env bash
set -u
V=/media/volume/newdata/exouser/flowdagger_e960a/e965_sirius_retrospective_correction_efficiency/env/bin/python
P=/media/volume/newdata/exouser/flowdagger_e960a/bundle/e959c_bootstrap_resolved_r0_20260817T082827Z/flowdagger/flowdagger_pi05/openpi/third_party/libero:/home/exouser/Tabero/analysis/results/pi0_nontransport_20260910/client_venv/lib/python3.10/site-packages
O=/media/volume/data/exouser/activeforcing_table_push_v5_20260910
export PYTHONPATH="$P"
export MUJOCO_GL=egl
export LIBERO_CONFIG_PATH=/home/exouser/Tabero/analysis/results/pi0_nontransport_20260910/libero_config
failed=0
for step in 55 57; do
  for spec in "untouched 0" "disabled 0" "hybrid 10" "hybrid 15" "hybrid 20" "hybrid 25" "hybrid 30"; do
    set -- $spec
    "$V" "$O/code/r2h_branch.py" "$step" "$1" "$2" >> "$O/R2H_RUN.log" 2>&1 &
    while [ "$(jobs -pr | wc -l)" -ge 3 ]; do
      wait -n || failed=1
    done
  done
done
wait || failed=1
if [ "$failed" -ne 0 ]; then
  exit 1
fi
"$V" "$O/code/aggregate_r2h.py" >> "$O/R2H_RUN.log" 2>&1
