#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE=/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911
DEST_HOST=x-csong7@anvil.rcac.purdue.edu
DEST=/anvil/projects/x-cis250966/tabero-transfer/jetstream-activeforcing-20260912
KEY=/home/exouser/.ssh/codex_anvil_migration_20260912
STATUS="$SOURCE/experiments/af_dump_bin_bigbin_forcegrid324_v1/pipeline_status.json"
LOG=/home/exouser/activeforcing_delta_sync_20260912.log
SSH=(ssh -i "$KEY" -o BatchMode=yes -o StrictHostKeyChecking=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=20 "$DEST_HOST")

timestamp() {
    date -u +%Y-%m-%dT%H:%M:%SZ
}

log() {
    printf '%s %s\n' "$(timestamp)" "$*" | tee -a "$LOG"
}

pipeline_status() {
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status","unknown"))' "$STATUS" 2>/dev/null || echo unknown
}

pipeline_rows() {
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("development_rows","unknown"))' "$STATUS" 2>/dev/null || echo unknown
}

sync_delta() {
    local final="$DEST/activeforcing_live_delta_latest.tar.zst"
    local partial="${final}.partial"
    "${SSH[@]}" "mkdir -p '$DEST' && rm -f '$partial'"
    tar --ignore-failed-read --warning=no-file-changed --warning=no-file-removed \
        --sparse -C "$SOURCE" -cf - \
        evidence \
        experiments \
        RoboTwin/scripts \
        RoboTwin/eval_result \
        RoboTwin/XPolicyLab/policy/Pi_0/model.py \
        RoboTwin/XPolicyLab/policy/Pi_0/openpi/scripts/process_data.py \
        RoboTwin/XPolicyLab/policy/Pi_0/openpi/src/openpi/policies/policy.py \
        RoboTwin/XPolicyLab/policy/Pi_0/openpi/src/openpi/training/config.py \
        | zstd -T1 -1 \
        | "${SSH[@]}" "umask 002; cat > '$partial' && mv '$partial' '$final'"
    "${SSH[@]}" "zstd -q -t '$final' && sha256sum '$final' > '${final}.sha256'"
    log "DELTA_VERIFIED rows=$(pipeline_rows) status=$(pipeline_status)"
}

log "DELTA_SYNC_START"
while true; do
    sync_delta
    state=$(pipeline_status)
    if [[ "$state" == "complete" ]]; then
        "${SSH[@]}" "cp '$DEST/activeforcing_live_delta_latest.tar.zst' '$DEST/activeforcing_final_delta_20260912.tar.zst' && cd '$DEST' && sha256sum activeforcing_final_delta_20260912.tar.zst > activeforcing_final_delta_20260912.tar.zst.sha256 && sha256sum -c activeforcing_final_delta_20260912.tar.zst.sha256"
        log "FINAL_DELTA_COMPLETE rows=$(pipeline_rows)"
        exit 0
    fi
    if [[ "$state" == "blocked" ]]; then
        log "DELTA_SYNC_STOPPED_PIPELINE_BLOCKED rows=$(pipeline_rows)"
        exit 2
    fi
    sleep 300
done
