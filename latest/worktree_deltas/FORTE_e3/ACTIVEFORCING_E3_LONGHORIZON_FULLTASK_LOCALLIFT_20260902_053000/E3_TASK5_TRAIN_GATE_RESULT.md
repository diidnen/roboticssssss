# E3 task5 TRAIN force-physics gate result

## Verdict

`SUPPORT_SCREEN_INCOMPLETE_PRIOR_FMAX_BOUNDARY_WITHDRAWN`

The preregistered 1--5 N TRAIN grid is complete (15/15 cells) and internally consistent, but it contains no full-task success. Its branch labels remain valid. The earlier claim that this exhausted an authoritative `Fmax=5 N` domain is withdrawn: the 5 N value belongs to E1 `libero_object/task5` (tomato sauce -> basket), not this `libero_10/task5` (book -> caddy). Numeric task IDs are suite-local.

## Data-quality checks

- Frozen OpenPI/project-local π0 backend and checkpoint: see `PI0_BACKEND_FREEZE.md`.
- Force grid: `1, 2, 3, 4, 5 N`; friction grid: `0.2, 0.6, 1.0`; TRAIN root: `7400`.
- Complete cells: 15; missing/extra: 0/0; analyzer errors: 0.
- Query state reached: 15/15.
- One root-state hash across all cells: `a26a8457835421558a23e34b54c04a9f3f6815b6f4cad87a84fcda6e9bb6c816`.
- Regimes: `LOCAL_FAILURE=10`, `LOCAL_SUCCESS_DOWNSTREAM_FAILURE=5`, `FULL_TASK_SUCCESS=0`.
- No three-regime friction context within the screened 1--5 N range and no full-task positive in that screen.

## Scientific interpretation

The pilot fixes the original LocalLift label-degeneracy problem for this long-horizon task: it contains both lift negatives and lift positives, and the five lift-positive branches all fail downstream. It does not yet produce a scientifically usable FullTask-vs-LocalLift classifier comparison because FullTask remains all-negative in the screened rows.

The separate nominal 8 N qualification succeeded, proving the frozen pi0 can execute the semantic task under a robust force. That point is not a Utility datapoint and cannot define or tune Fmax. It establishes a pre-outcome reason to screen 6/7/8 N on the same TRAIN root purely for label support.

## Required stop

- Do not start task5 DEV or train/evaluate a FullTask classifier while the screened labels are all-negative.
- Do not transfer the E1 task5 Fmax, infer a new Fmax from outcomes, modify the Utility hash/selector/tie-break, or use TEST.
- Preserve all 15 rows unchanged. Run the frozen TRAIN-only same-root extension at 6/7/8 N for `root=7400, friction=0.6`; re-run the three-regime gate. If it still lacks a FullTask positive, proceed to the pre-registered reserve long-horizon candidates.
- Task1 remains a useful short-horizon diagnostic only and cannot satisfy long-horizon completion.

Machine-readable evidence: `E3_TASK5_TRAIN_REGIME_GATE.json`. Cell-level evidence: `E3_TASK5_TRAIN_REGIME_TABLE.csv`. Correction lineage: `E3_TASK5_FMAX_IDENTITY_COLLISION_AUDIT.md`.
