#!/usr/bin/env bash
set -e
export PYTHONPATH=/home/exouser/SoftVTBench/openpi/upstream/src
export XLA_PYTHON_CLIENT_PREALLOCATE=false
exec /media/volume/newdata/exouser/softvtbench/openpi-venv/bin/python -u /media/volume/data/exouser/activeforcing_table_push_v5_20260910/code/seeded_pi0_server.py
