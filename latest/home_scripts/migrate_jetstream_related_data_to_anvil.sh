#!/usr/bin/env bash
set -euo pipefail

DEST_HOST="x-csong7@anvil.rcac.purdue.edu"
DEST_DIR="/anvil/projects/x-cis250966/tabero-transfer/jetstream-activeforcing-20260912"
KEY="/home/exouser/.ssh/codex_anvil_migration_20260912"
SSH=(ssh -i "$KEY" -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes "$DEST_HOST")
LOG="/home/exouser/related_data_migration_20260912.log"

exec >>"$LOG" 2>&1
echo "RELATED_DATA_START $(date -u +%FT%TZ)"

transfer_tree() {
  local label="$1"
  local source_root="$2"
  shift 2
  local archive="jetstream_${label}_20260912.tar.zst"
  local partial="$DEST_DIR/${archive}.partial"
  local final="$DEST_DIR/$archive"

  echo "ARCHIVE_START label=$label time=$(date -u +%FT%TZ)"
  "${SSH[@]}" "mkdir -p '$DEST_DIR' && rm -f '$partial'"
  tar -C "$source_root" -cf - -- "$@" \
    | zstd -T2 -3 \
    | "${SSH[@]}" "cat > '$partial'"
  "${SSH[@]}" "mv '$partial' '$final' && zstd -t '$final' && (cd '$DEST_DIR' && sha256sum '$archive' > '$archive.sha256')"
  local result
  result=$("${SSH[@]}" "stat -c '%s' '$final'; cat '$final.sha256'")
  echo "ARCHIVE_COMPLETE label=$label time=$(date -u +%FT%TZ)"
  echo "$result"
}

data_items=(
  activeforcing_knob_damping_v1_20260911
  activeforcing_knob_damping_v2_20260911
  activeforcing_table_push_20260910
  activeforcing_table_push_v2_20260910
  activeforcing_table_push_v3_20260910
  activeforcing_table_push_v4_20260910
  activeforcing_table_push_v5_20260910
  activeforcing_table_push_v6_20260910
  activeforcing_table_push_v7_20260911
  activeforcing_table_push_v8_20260911
  activeforcing_table_push_v8b_20260911
  activeforcing_table_push_v8c_20260911
  activeforcing_table_push_v9_20260911
  af_strict_noquery_paired_20260910
  online_vla_activeforcing_20260910
  pi0_nontransport_20260910
  pi0_table_friction_20260910
  TASK_FORM_EXPERIMENT_STATUS_20260911.json
)

newdata_dependency_items=(
  exouser/tabero/data/Isaaclab_Libero
  exouser/tabero/data/Tactile_Manipulation_Dataset
  exouser/flowdagger_e960a/e965_sirius_retrospective_correction_efficiency
)

# The data-volume experiment archive is independent from the newdata archive, so
# stream both in parallel to reduce shutdown risk. Each result is atomically
# renamed only after the source pipeline and remote integrity test succeed.
transfer_tree "sibling_activeforcing_experiments" "/media/volume/data/exouser" "${data_items[@]}" &
pid_data=$!
transfer_tree "newdata_tabero_dependencies" "/media/volume/newdata" "${newdata_dependency_items[@]}" &
pid_deps=$!
transfer_tree "activeforcing_disk_archive_20260902" "/media/volume/newdata/exouser" "ACTIVEFORCING_DISK_ARCHIVE_20260902" &
pid_archive=$!

status=0
wait "$pid_data" || status=1
wait "$pid_deps" || status=1
wait "$pid_archive" || status=1

if [[ "$status" -ne 0 ]]; then
  echo "RELATED_DATA_FAILED $(date -u +%FT%TZ)"
  exit 1
fi

"${SSH[@]}" "cd '$DEST_DIR' && cat jetstream_sibling_activeforcing_experiments_20260912.tar.zst.sha256 jetstream_newdata_tabero_dependencies_20260912.tar.zst.sha256 jetstream_activeforcing_disk_archive_20260902_20260912.tar.zst.sha256 > RELATED_DATA_COMPLETE.sha256"
echo "RELATED_DATA_COMPLETE $(date -u +%FT%TZ)"
