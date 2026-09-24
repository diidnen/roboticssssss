#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT=/home/exouser/FORTE
ARCHIVE_ROOT=/media/volume/newdata/exouser/activeforcing_core_closure_archive_20260902

# These are completed, older read-only closure snapshots.  Current experiment
# lanes, MASS, git worktrees, and the latest closure snapshots are excluded.
SNAPSHOTS=(
  activeforcing_full_claim_closure_20260902_094258
  activeforcing_full_claim_closure_20260902_094356
  activeforcing_full_claim_closure_20260902_100709
  activeforcing_full_claim_closure_20260902_102314
  activeforcing_full_claim_closure_20260902_102346
  activeforcing_full_claim_closure_20260902_104952
  activeforcing_full_claim_closure_20260902_105113
  activeforcing_full_claim_closure_20260902_105238
  activeforcing_full_claim_closure_20260902_105406
)

mkdir -p "${ARCHIVE_ROOT}"

for name in "${SNAPSHOTS[@]}"; do
  source_path="${SOURCE_ROOT}/${name}"
  archive_path="${ARCHIVE_ROOT}/${name}"
  if [[ -L "${source_path}" && -d "${archive_path}" ]]; then
    continue
  fi
  if [[ ! -d "${source_path}" || -L "${source_path}" ]]; then
    echo "FAIL_CLOSED: source is not a real directory: ${source_path}" >&2
    exit 2
  fi
  if [[ -e "${archive_path}" || -L "${archive_path}" ]]; then
    echo "FAIL_CLOSED: archive target already exists: ${archive_path}" >&2
    exit 3
  fi
  mv "${source_path}" "${archive_path}"
  ln -s "${archive_path}" "${source_path}"
done

for name in "${SNAPSHOTS[@]}"; do
  source_path="${SOURCE_ROOT}/${name}"
  archive_path="${ARCHIVE_ROOT}/${name}"
  [[ -L "${source_path}" && "$(readlink -f "${source_path}")" == "${archive_path}" && -d "${archive_path}" ]] || {
    echo "FAIL_CLOSED: post-migration path check failed for ${name}" >&2
    exit 4
  }
done

df -h "${SOURCE_ROOT}" "${ARCHIVE_ROOT}"
