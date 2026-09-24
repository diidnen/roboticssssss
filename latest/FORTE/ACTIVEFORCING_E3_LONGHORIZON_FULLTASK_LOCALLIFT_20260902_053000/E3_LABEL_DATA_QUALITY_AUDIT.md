# E3 label data-quality audit

## Dataset, grain, and intended use

- Source: `/home/exouser/FORTE/activeforcing_final_closure_20260902_034923/E3_BRANCH_LABEL_AUDIT.csv`
- Grain: one physical branch at `(task, root_id, context_id, force_N, repeat)`.
- Intended use: determine whether the archive supports a scientific LocalLift versus FullTask comparison. It is not a zero-query archive: every branch follows the P4-B query.

## Checks and evidence

| Check | Evidence | Verdict |
|---|---:|---|
| Row/schema count | 720 rows, 10 columns | pass |
| Branch-key uniqueness | 720/720 unique `branch_id`; 720/720 unique composite grain | pass |
| Root/context shape | 24 roots x 30 branches; 72 contexts x 10 branches | pass |
| Task balance | task0/1/5/6 each 180 rows | pass |
| Friction-band balance | LOW/MID/HIGH each 240 rows | pass |
| Repeat balance | repeat1/repeat2 each 360 rows | pass |
| Failure-reason consistency | all 145 failures are `full_task_failure`; successes have no failure reason | pass |
| LocalLift label support | 720 positive, 0 negative | **critical failure** |
| FullTask label support | 575 success, 145 failure | pass for descriptive FullTask analysis |

Per-task LocalLift negatives are zero. FullTask failures after local success are task0=42, task1=59, task5=38, task6=6. There are no contradictions of FullTask success with LocalLift failure because LocalLift is universally positive.

## Finding and analytical risk

**Critical, high confidence:** the existing archive cannot identify or evaluate a LocalLift classifier. A fitted model would see only one class, and any apparent comparison against FullTask would conflate label degeneracy with scientific method performance. The balanced branch structure and valid keys do not repair this missing outcome regime.

Likely cause: the archived force ranges (task0/5 roughly 3--5 N, task1 roughly 4--6 N, task6 roughly 3--4 N) start above the local lift-failure boundary for these short basket tasks, while downstream failures still occur.

## Remediation and stable automated gates

1. Do not train or report a scientific LocalLift classifier from these 720 labels.
2. Qualify a task selected from historical semantics, not ActiveForcing outcomes, with frozen authoritative OpenPI and robust 8 N only for nominal-capability screening.
3. On TRAIN/DEV roots, pre-register a force/physics range that produces all three labels: local failure, local success plus downstream failure, and full-task success.
4. Before fitting, enforce: unique composite grain; no sibling-root split; nonzero support for every required regime overall and per reported split; no future-outcome fields in identifier features.

No temporal trend check is possible because this branch audit contains no acquisition timestamp. The source path and exact branch keys remain the inspectable evidence trail.
