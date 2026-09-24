# ActiveForcing final data audit

Status: **PASS_WITH_FORCE_GRID_GAP**

## Authoritative archive

- Source: `/home/exouser/FORTE/gnp_style_continuous_20260830_125107/CONTINUOUS_TRAIN_SUCCESS_DATA.csv`
- SHA-256: `80bb811e79796a9c292517ae37d2e224710a228ab8afd1e5be38ab9c2e5857c9`
- Rows: **720**; valid **720**; state parity **720**
- Tasks: `[0, 1, 5, 6]`; task counts: `{'0': 180, '1': 180, '5': 180, '6': 180}`
- Contexts: **72**; task-specific root families: **24**
- Each context: **5** force cells × **2** repeats; all contexts have one strict pre-probe hash, branch snapshot hash, and nominal-motion hash.

The 720 rows are post-query force branches. Therefore the offline no-information row is named **Query-Ignored / No-Physical-Information**. It is not reported as a true zero-query rollout.

## Frozen-method compatibility

The current frozen Direct selector uses the 0.25 N grid `[3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0]` and threshold 0.5 with max-force fallback. The authoritative 720 archive contains five continuous, context-specific stratum draws per context. Its observed force supports are recorded in `ACTIVEFORCING_FINAL_DATA_GAPS.json`; no nearest-force replacement is used.

Task1 retains the lineage caveat: 140/180 terminal labels are reconstructed. Exact label sensitivity is retained from the existing transfer artifact.

## Split integrity

The accompanying JSON assigns each task-specific root family to exactly one of three root-heldout folds. All sibling force branches, friction contexts, and repeats remain together.
