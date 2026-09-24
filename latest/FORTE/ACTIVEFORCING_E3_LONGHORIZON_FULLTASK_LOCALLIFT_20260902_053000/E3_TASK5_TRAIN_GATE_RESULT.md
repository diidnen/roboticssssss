# E3 task5 TRAIN force-physics gate result

## Verdict

`FAIL_CLOSED_NO_IN_DOMAIN_FULL_TASK_SUCCESS`

The preregistered TRAIN grid is complete (15/15 cells) and internally consistent, but it contains no full-task success at or below the authoritative task5 `Fmax=5 N`. DEV confirmation is therefore prohibited by the preregistered stopping rule.

## Data-quality checks

- Frozen OpenPI/project-local π0 backend and checkpoint: see `PI0_BACKEND_FREEZE.md`.
- Force grid: `1, 2, 3, 4, 5 N`; friction grid: `0.2, 0.6, 1.0`; TRAIN root: `7400`.
- Complete cells: 15; missing/extra: 0/0; analyzer errors: 0.
- Query state reached: 15/15.
- One root-state hash across all cells: `a26a8457835421558a23e34b54c04a9f3f6815b6f4cad87a84fcda6e9bb6c816`.
- Regimes: `LOCAL_FAILURE=10`, `LOCAL_SUCCESS_DOWNSTREAM_FAILURE=5`, `FULL_TASK_SUCCESS=0`.
- No three-regime friction context and no in-domain full-task positive.

## Scientific interpretation

The pilot fixes the original LocalLift label-degeneracy problem for this long-horizon task: it contains both lift negatives and lift positives, and the five lift-positive branches all fail downstream. It does not, however, produce a scientifically usable FullTask-vs-LocalLift classifier comparison because FullTask remains all-negative inside the frozen Utility support.

The nominal 8 N qualification succeeded, proving the frozen π0 can execute the semantic task under a robust force. That point is outside the authoritative task5 Utility domain and cannot be relabeled as an in-domain positive, used to change `Fmax`, or used to tune Utility.

## Required stop

- Do not start task5 DEV.
- Do not change `Fmax=5 N`, Utility hash, selector, or low-force tie-break.
- Do not train/evaluate a FullTask classifier on the all-negative in-domain labels.
- Preserve this result as a domain-boundary finding, then use the protocol's current-task-first fallback: a fresh, same-root low-force extension for an existing task whose authoritative support already contains downstream failures and full successes. Task1 is the preregistration candidate because `Fmax=6 N` and the pre-existing audit records 59 downstream failures plus 121 full successes; selection is based on label-support physics, not ActiveForcing wins.

Machine-readable evidence: `E3_TASK5_TRAIN_REGIME_GATE.json`. Cell-level evidence: `E3_TASK5_TRAIN_REGIME_TABLE.csv`.
