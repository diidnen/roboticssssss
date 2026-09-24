# Recovery quarantine manifest

Created 2026-09-02T14:23Z. No material user data was deleted or moved. This manifest is the authoritative exclusion index until each item is independently revalidated.

## Excluded interrupted/partial evidence

- E5 rejected shard: `/home/exouser/FORTE/analysis/results/ACTIVEFORCING_E5_FRESH_UTILITY_E2E_20260902_113527_task1_offset03` — `FULL_STATUS` was `PARTIAL`.
- E5 rejected shard: `/home/exouser/FORTE/analysis/results/ACTIVEFORCING_E5_FRESH_UTILITY_E2E_20260902_065500_retry7` — runner hash mismatch.
- Mass in-flight material under `.../M3_TASK2_FORMAL_STRUCTURED_20260902_130700_RESUME/` for `TRAIN root8103 MID` after the last complete 10-row context commit — telemetry/query files exist, but no complete atomic context commit is present.
- The first E3 reserve task2 attempt with exit code 137 — explicitly `RESOURCE_ABORT_NO_SCIENTIFIC_OUTCOME`; output rows are zero and it is not reusable evidence.
- Any file or directory with `PARTIAL`, `partial`, `interrupted`, `oom`, `retry` (unless independently accepted by a PASS shard gate), `RESOURCE_ABORT`, or `NOT_RUN` status is excluded from formal tables by default.
- E2 CPU fusion attempt directories that stopped at argument normalization or NumPy-boolean serialization — diagnostic startup failures only; no model or fusion result was emitted.
- E3 downstream command-chain artifacts with the ID1 producer-QA hash mismatch or non-existent norm path — not launch-authorizing; training/DEV remain blocked pending coordinator repair.

## Accepted exception

E5 `ACTIVEFORCING_E5_FRESH_UTILITY_E2E_20260902_132755_task1_offset03` is not quarantined because its independent `FULL_STATUS.json` and `SHARD_QA.json` both pass and agree on the frozen Expected-Utility lineage.

## Re-entry rule

Quarantined material can re-enter only after a fresh validator proves atomic completeness, matching frozen protocol/runner/config hashes, reset/state parity, and no duplicate accepted tuple.
