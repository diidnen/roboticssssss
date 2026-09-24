#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_PARENT=/media/volume/data/exouser
SOURCE_NAME=activeforcing_robotwin_taskforms_20260911
SOURCE_ROOT="$SOURCE_PARENT/$SOURCE_NAME"
DEST_HOST=x-csong7@anvil.rcac.purdue.edu
DEST_ROOT=/anvil/projects/x-cis250966/tabero-transfer/jetstream-activeforcing-20260912
KEY=/home/exouser/.ssh/codex_anvil_migration_20260912
STATE_FILE="$SOURCE_ROOT/evidence/MIGRATION_SOURCE_STATE_20260912.txt"
SSH=(ssh -i "$KEY" -o BatchMode=yes -o StrictHostKeyChecking=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=20 "$DEST_HOST")

timestamp() {
    date -u +%Y-%m-%dT%H:%M:%SZ
}

log() {
    printf '%s %s\n' "$(timestamp)" "$*"
}

remote() {
    "${SSH[@]}" "$@"
}

write_source_state() {
    {
        echo "schema=JETSTREAM_ACTIVEFORCING_MIGRATION_V1"
        echo "captured_at=$(timestamp)"
        echo "source_host=$(hostname)"
        echo "source_root=$SOURCE_ROOT"
        echo "destination=$DEST_HOST:$DEST_ROOT"
        echo "copy_semantics=non_destructive_live_snapshot"
        echo "source_bytes=$(du -sb "$SOURCE_ROOT" | awk '{print $1}')"
        echo "source_files=$(find "$SOURCE_ROOT" -xdev -type f | wc -l)"
        echo "robotwin_head=$(git -C "$SOURCE_ROOT/RoboTwin" rev-parse HEAD)"
        echo "xpolicylab_head=$(git -C "$SOURCE_ROOT/RoboTwin/XPolicyLab" rev-parse HEAD)"
        echo "github_tabero_head=4695d0ed44fab8f0a1406adc1135cceb146f326f"
        echo "dump_pipeline_status_begin"
        cat "$SOURCE_ROOT/experiments/af_dump_bin_bigbin_forcegrid324_v1/pipeline_status.json" 2>/dev/null || true
        echo "dump_pipeline_status_end"
        echo "robotwin_status_begin"
        git -C "$SOURCE_ROOT/RoboTwin" status --short
        echo "robotwin_status_end"
        echo "xpolicylab_status_begin"
        git -C "$SOURCE_ROOT/RoboTwin/XPolicyLab" status --short
        echo "xpolicylab_status_end"
    } > "$STATE_FILE"
}

send_archive() {
    local label=$1
    shift
    local final="$DEST_ROOT/${label}.tar.zst"
    local partial="${final}.partial"
    log "ARCHIVE_START label=$label"
    remote "mkdir -p '$DEST_ROOT' && rm -f '$partial'"
    tar --ignore-failed-read --warning=no-file-changed --warning=no-file-removed \
        --sparse -C "$SOURCE_ROOT" -cf - "$@" \
        | zstd -T2 -1 \
        | remote "umask 002; cat > '$partial' && mv '$partial' '$final'"
    remote "zstd -q -t '$final' && sha256sum '$final' > '${final}.sha256'"
    remote "stat -c 'ARCHIVE_DONE label=$label bytes=%s path=%n' '$final'; cat '${final}.sha256'"
}

send_full_archive() {
    local label=activeforcing_robotwin_taskforms_20260911_full_live_20260912
    local final="$DEST_ROOT/${label}.tar.zst"
    local partial="${final}.partial"
    log "FULL_ARCHIVE_START source=$SOURCE_ROOT"
    remote "mkdir -p '$DEST_ROOT' && rm -f '$partial'"
    tar --ignore-failed-read --warning=no-file-changed --warning=no-file-removed \
        --sparse -C "$SOURCE_PARENT" -cf - "$SOURCE_NAME" \
        | zstd -T2 -1 \
        | remote "umask 002; cat > '$partial' && mv '$partial' '$final'"
    remote "zstd -q -t '$final' && sha256sum '$final' > '${final}.sha256'"
    remote "stat -c 'FULL_ARCHIVE_DONE bytes=%s path=%n' '$final'; cat '${final}.sha256'"
}

log "MIGRATION_START"
write_source_state
remote "mkdir -p '$DEST_ROOT'"
send_archive activeforcing_critical_20260912 \
    evidence \
    experiments \
    RoboTwin/.git \
    RoboTwin/envs \
    RoboTwin/scripts \
    RoboTwin/XPolicyLab/.git \
    RoboTwin/XPolicyLab/policy/Pi_0/model.py \
    RoboTwin/XPolicyLab/policy/Pi_0/openpi/scripts/process_data.py \
    RoboTwin/XPolicyLab/policy/Pi_0/openpi/src/openpi/policies/policy.py \
    RoboTwin/XPolicyLab/policy/Pi_0/openpi/src/openpi/training/config.py
send_full_archive
remote "cp '$DEST_ROOT/activeforcing_robotwin_taskforms_20260911_full_live_20260912.tar.zst.sha256' '$DEST_ROOT/MIGRATION_PRIMARY_COMPLETE.sha256'"
log "MIGRATION_COMPLETE"
