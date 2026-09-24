#!/usr/bin/env bash
set -euo pipefail

run_dir=/home/exouser/FORTE/activeforcing_shared_physical_transfer_20260901_094722
runner=/home/exouser/FORTE/run_shared_physical_transfer.py

for target in 0 1 5 6; do
  for fold in 0 1 2; do
    for seed in 0 1 2; do
      for estimator in LearnedProbe ExplicitSysID GT; do
        printf '%s\t%s\t%s\t%s\n' "$target" "$fold" "$seed" "$estimator"
      done
    done
  done
done | xargs -P 12 -n 4 bash -c '
  env MPLCONFIGDIR=/tmp/mpl-shared-transfer OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    python3 /home/exouser/FORTE/run_shared_physical_transfer.py \
    --out /home/exouser/FORTE/activeforcing_shared_physical_transfer_20260901_094722 \
    shard --target "$0" --fold "$1" --seed "$2" --estimator "$3"
'
