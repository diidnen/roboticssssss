#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE=/home/exouser
DEST_HOST=x-csong7@anvil.rcac.purdue.edu
DEST=/anvil/projects/x-cis250966/tabero-transfer/jetstream-activeforcing-20260912
KEY=/home/exouser/.ssh/codex_anvil_migration_20260912
LABEL=jetstream_home_project_nondot_20260912
FINAL="$DEST/$LABEL.tar.zst"
PARTIAL="$FINAL.partial"
SCOPE="$SOURCE/PROJECT_HOME_MIGRATION_SCOPE_20260912.txt"
SSH=(ssh -i "$KEY" -o BatchMode=yes -o StrictHostKeyChecking=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=20 "$DEST_HOST")

timestamp() {
    date -u +%Y-%m-%dT%H:%M:%SZ
}

{
    echo "schema=JETSTREAM_HOME_PROJECT_MIGRATION_V1"
    echo "captured_at=$(timestamp)"
    echo "source=$SOURCE"
    echo "destination=$DEST_HOST:$DEST"
    echo "included=all_nonhidden_top_level_entries_plus_.git_plus_.libero"
    echo "excluded=.ssh,.codex,.bash_history,hidden_credentials,hidden_caches"
    echo "top_level_entries_begin"
    find "$SOURCE" -mindepth 1 -maxdepth 1 -printf '%f\n' | sort
    echo "top_level_entries_end"
} > "$SCOPE"

printf '%s HOME_PROJECT_ARCHIVE_START\n' "$(timestamp)"
"${SSH[@]}" "mkdir -p '$DEST' && rm -f '$PARTIAL'"
cd "$SOURCE"
tar --ignore-failed-read --warning=no-file-changed --warning=no-file-removed \
    --sparse -cf - .git .libero -- * \
    | zstd -T2 -1 \
    | "${SSH[@]}" "umask 002; cat > '$PARTIAL' && mv '$PARTIAL' '$FINAL'"
"${SSH[@]}" "zstd -q -t '$FINAL' && sha256sum '$FINAL' > '${FINAL}.sha256' && cp '${FINAL}.sha256' '$DEST/HOME_PROJECT_COMPLETE.sha256'"
"${SSH[@]}" "stat -c 'HOME_PROJECT_ARCHIVE_DONE bytes=%s path=%n' '$FINAL'; cat '${FINAL}.sha256'"
printf '%s HOME_PROJECT_MIGRATION_COMPLETE\n' "$(timestamp)"
