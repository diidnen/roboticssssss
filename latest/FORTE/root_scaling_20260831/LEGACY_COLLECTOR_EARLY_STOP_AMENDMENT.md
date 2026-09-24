# Legacy collector early-stop amendment

Amendment time: 2026-08-31T15:45:20Z (UTC)

## Decision

The inherited task0 collector originally targeted 62 additions / 620 branches (`root12`–`root73`). The current S50 preregistration uses only `root12`–`root55`. At the atomic context boundary after `root63`, the authoritative audit showed:

- required S50 roots complete: 44/44;
- required S50 branches complete: 440/440;
- required parity and context-commit checks: PASS;
- completed legacy-excluded roots: `root56`–`root63` (8);
- remaining legacy-excluded roots not started: `root64`–`root73` (10).

The old collector was therefore stopped at `GRACEFUL_STOP_AFTER_CONTEXT`, after the current excluded context had emitted `CONTEXT_COMPLETE=PASS`. No branch was killed mid-execution and no partial excluded context was counted as a scientific row.

## Reason

> All preregistered S50 scientific roots were already complete. Remaining collector targets belong only to the legacy N80 plan and are preregistered-excluded from the current S6/S15/S30/S50 experiment.

This is infrastructure-only deletion of work. It is not scientific early stopping and was not based on any TEST result, learning-curve result, metric, or model comparison.

## Scientific invariants

This amendment does not change:

- S6/S15/S30/S50 root membership;
- untouched TEST roots or TEST semantics;
- force grid, repeat count, branch mapping, outcome labels, success semantics, or learning-curve gates;
- PCA, normalization, TRAIN, metrics, or model selection inputs;
- scientific retry count (remains zero).

`root56`–`root73` remain explicitly excluded from PCA, normalization, TRAIN, S50, metrics, and model selection, even when roots `root56`–`root63` were physically collected before the stop.

## Supervisor gate amendment

The supervisor now accepts the legacy collection gate only when all preregistered S50 roots, required branches, context commits, and parity checks pass, together with `S50_REQUIRED_ROOTS_QA.json=PASS`. It no longer waits for the legacy original target `62/620`.

The recorded counts are:

```text
collected_total = 52
included_in_S50 = 44
excluded_legacy_collected = 8
excluded_legacy_remaining = 10
```

The amendment was made before any authoritative TEST outcome or learning-curve result was viewed or accepted. A stale supervisor nevertheless briefly initialized a task0 TEST worker before this post-collection freeze was installed; it was terminated before any authoritative TEST context/branch/parity row or TEST commit, and all 13 partial files were moved to `collection_test_ABORTED_20260831T154500Z` and marked non-authoritative in `TEST0_COLLECTION_ABORTED_BEFORE_COMMIT.json`.
