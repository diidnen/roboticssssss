# P1 simplified collector launch template (template only; not executed by preflight)
P1_LANE=/home/exouser/E3_E6_E7_LANES/E3_FULLTASK_VS_LOCALLIFT
B5=/media/volume/newdata/exouser/Tabero_e3lh/analysis/results/b5_tabero_neutral_20260822_040652/scripts/b5_tabero_neutral_client.py
PY=/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python
TABERO=/media/volume/newdata/exouser/Tabero_e3lh
RAW=/media/volume/newdata/exouser/tabero/data/Isaaclab_Libero
HDF5_SOURCE_DIR=/media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_SOURCE_20260902_103300/assembled_hdf5
HDF5_ROOT_MANIFEST=/home/exouser/Tabero/E3_HDF5_ROOT_EPISODE_MANIFEST.json
SERVER=18883
ROOT_ID=<PRE_REGISTERED_ROOT_ID>
FORCE_N=<PRE_REGISTERED_FORCE_VALUE_N>
FMIN_N=<PRE_REGISTERED_FORCE_MIN_N>
FMAX_N=<PRE_REGISTERED_FORCE_MAX_N>
UTILITY_SHA256=<PRE_REGISTERED_UTILITY_CONTRACT_SHA256>
PI0_CKPT_SHA256=<LOCKED_PI0_CHECKPOINT_TREE_SHA256>
PI0_SERVER_SHA256=<LOCKED_PI0_SERVER_SOURCE_SHA256>
PLAN_SHA256=<SHA256_OF_FROZEN_P1_PLAN>
OUT=/media/volume/newdata/exouser/activeforcing_e3/P1_SIMPLIFIED_ROOT${ROOT_ID}_F${FORCE_N}N

test -n "${FMIN_N}" -a -n "${FMAX_N}" -a -n "${UTILITY_SHA256}"
test "${FMIN_N}" != "<PRE_REGISTERED_FORCE_MIN_N>"
test "${FMAX_N}" != "<PRE_REGISTERED_FORCE_MAX_N>"
test "${UTILITY_SHA256}" != "<PRE_REGISTERED_UTILITY_CONTRACT_SHA256>"
test "${FORCE_N}" -ge "${FMIN_N}" -a "${FORCE_N}" -le "${FMAX_N}"

E3_OBJECT_NAME=black_book_1 E3_FORCE_N="${FORCE_N}" P1_OBJECT_NAME=black_book_1 P1_TARGET_NAME=desk_caddy_1 P1_TASK_SUITE=libero_10 P1_TASK_ID=5 P1_QUERY_MODE=NONE P1_BRANCH_ID="libero10_t5_root${ROOT_ID}_F${FORCE_N}" P1_ROOT_ID="${ROOT_ID}" P1_OUTPUT_DIR="${OUT}" P1_SAVE_X_IMAGES=1 P1_B5_SOURCE_PATH="${B5}" P1_B5_SOURCE_SHA256="<B5_SOURCE_SHA256>" P1_WRAPPER_SHA256="<THIS_NEW_FILE_SHA256>" P1_PI0_CHECKPOINT_SHA256="${PI0_CKPT_SHA256}" P1_PI0_SERVER_SHA256="${PI0_SERVER_SHA256}" P1_PLAN_SHA256="${PLAN_SHA256}" P1_FMAX_N="${FMAX_N}" P1_UTILITY_SHA256="${UTILITY_SHA256}" TABERO_ROOT="${TABERO}" PYTHONNOUSERSITE=1 PYTHONPATH="${TABERO}:${TABERO}/benchmarks/openpi/openpi-client/src" HDF5_TRAJ_SOURCE_DIR="${HDF5_SOURCE_DIR}" P1_HDF5_EPISODE_INDEX="<PINNED_HDF5_EPISODE_INDEX>" P1_HDF5_ROOT_MANIFEST="${HDF5_ROOT_MANIFEST}" LIBERO_CONFIG_DIR="${TABERO}/benchmarks/datasets/libero/config" LIBERO_ASSETS_DATA_DIR="${RAW}/USD" OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y "${PY}" -u /home/exouser/E3_E6_E7_LANES/E3_FULLTASK_VS_LOCALLIFT/p1_simplified_runtime_collector.py   --execute-transformed-source   --server-host 127.0.0.1 --server-port "${SERVER}"   --task-suite libero_10 --task-id 5   --language-instruction "pick up the book and place it in the back compartment of the caddy"   --control-mode tactile --task Isaac-Libero-Franka-Hybrid-Tactile-v0   --num-total-experiments 1 --max-inference-steps 50 --replan-steps 10   --seed "${ROOT_ID}" --no-randomize-light   --b5-output-dir "${OUT}" --b5-method P1_SIMPLIFIED_DIRECT --b5-friction <AUDIT_ONLY_MU>

# Run this command once per pre-registered force, with the same ROOT_ID.
# The collector accepts no result-derived force, Fmax, or Utility changes.
