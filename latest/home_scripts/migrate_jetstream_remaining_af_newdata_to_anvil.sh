#!/usr/bin/env bash
set -euo pipefail

DEST_HOST="x-csong7@anvil.rcac.purdue.edu"
DEST_DIR="/anvil/projects/x-cis250966/tabero-transfer/jetstream-activeforcing-20260912"
KEY="/home/exouser/.ssh/codex_anvil_migration_20260912"
SSH=(ssh -i "$KEY" -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes "$DEST_HOST")
LOG="/home/exouser/remaining_af_newdata_migration_20260912.log"

exec >>"$LOG" 2>&1
echo "REMAINING_AF_NEWDATA_START $(date -u +%FT%TZ)"

transfer_tree() {
  local label="$1"
  shift
  local archive="jetstream_${label}_20260912.tar.zst"
  local partial="$DEST_DIR/${archive}.partial"
  local final="$DEST_DIR/$archive"

  echo "ARCHIVE_START label=$label time=$(date -u +%FT%TZ)"
  "${SSH[@]}" "mkdir -p '$DEST_DIR' && rm -f '$partial'"
  tar -C /media/volume/newdata/exouser -cf - -- "$@" \
    | zstd -T2 -3 \
    | "${SSH[@]}" "cat > '$partial'"
  "${SSH[@]}" "mv '$partial' '$final' && zstd -t '$final' && (cd '$DEST_DIR' && sha256sum '$archive' > '$archive.sha256')"
  local result
  result=$("${SSH[@]}" "stat -c '%s' '$final'; cat '$final.sha256'")
  echo "ARCHIVE_COMPLETE label=$label time=$(date -u +%FT%TZ)"
  echo "$result"
}

transfer_tree "newdata_activeforcing_e3" activeforcing_e3 &
pid_e3=$!
transfer_tree "newdata_online_vla_activeforcing_20260907" online_vla_activeforcing_20260907 &
pid_online=$!
transfer_tree "newdata_pi0_libero_activeforcing_20260910" pi0_libero_activeforcing_20260910 &
pid_pi0=$!
transfer_tree "newdata_activeforcing_misc" \
  ACTIVEFORCING_E5_QUARANTINE_20260903 \
  activeforcing_core_closure_archive_20260902 \
  activeforcing_e5_shards_20260902 \
  activeforcing_locked_test_20260902_120000 \
  activeforcing_locked_test_20260902_120000_root03 \
  task_form_coverage_20260910 \
  Tabero_e3lh &
pid_misc=$!

status=0
wait "$pid_e3" || status=1
wait "$pid_online" || status=1
wait "$pid_pi0" || status=1
wait "$pid_misc" || status=1

if [[ "$status" -ne 0 ]]; then
  echo "REMAINING_AF_NEWDATA_FAILED $(date -u +%FT%TZ)"
  exit 1
fi

"${SSH[@]}" "cd '$DEST_DIR' && cat jetstream_newdata_activeforcing_e3_20260912.tar.zst.sha256 jetstream_newdata_online_vla_activeforcing_20260907_20260912.tar.zst.sha256 jetstream_newdata_pi0_libero_activeforcing_20260910_20260912.tar.zst.sha256 jetstream_newdata_activeforcing_misc_20260912.tar.zst.sha256 > REMAINING_AF_NEWDATA_COMPLETE.sha256"
echo "REMAINING_AF_NEWDATA_COMPLETE $(date -u +%FT%TZ)"
