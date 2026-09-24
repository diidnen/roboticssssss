# E2 Physical Identification Final

**Status: E2_PHYSICAL_IDENTIFICATION_COMPLETE_SCIENTIFIC_NEGATIVE_FOR_UNAVAILABLE_FUSION**  
Generated: `2026-09-03T02:10:43.541593+00:00`. This is a CPU/offline closure from frozen artifacts. No rollout was recollected and no checkpoint, E5 process, scheduler, or protected campaign was touched.

## Bottom line

The strongest valid primary result is the root-heldout P4-B physical-history estimator. At the context grain (72 contexts, 24 root families, 4 tasks, 720 force branches), the frozen ensemble-mean point estimate obtains **MAE 0.0655**, **RMSE 0.0893**, median AE **0.0532**, bias **-0.0165**, Spearman **0.9039**, pairwise ranking **0.9861** over **72** within-root friction pairs, and LOW/MID/HIGH accuracy **0.9167**.

The independent three-member PhysicalBelief diagnostic is weaker on its DEV root-heldout split (24 contexts, 8 roots): MAE **0.1431**, RMSE **0.2016**, Spearman **0.7043**, pairwise ranking **0.8750**. Its nominal 68/90/95% empirical coverages are **0.6667 / 0.8750 / 0.8750**, with Gaussian NLL **-0.3180**, uncertainty/error Spearman **-0.0557**, and AURC **0.1289**. Brier is **NA** because no archived calibrated band-probability vector exists. These are diagnostic uncertainty statistics, not a calibration claim.

## Paper-ready comparison

The CSV is the machine-readable table: [TABLE_E2_PHYSICAL_IDENTIFICATION.csv](./TABLE_E2_PHYSICAL_IDENTIFICATION.csv). `MAE`–`low_mid_high_accuracy` use the `metric_aggregation` column. Downstream columns are archived same-controller proxies only when a matched Direct + candidate set + Expected Utility artifact exists.

| variant | split / priority | μ metrics | uncertainty / visual diagnostic | downstream proxy |
|---|---|---|---|---|
| PRIOR | E1 720 archive / secondary | not an identifier | — | SR 0.9444, agreement 0.3750, force 4.477 N, under 0.0417, utility 0.0487 |
| VISION ONLY | root-heldout / primary | not a μ estimator | NLL 0.2310, Brier 0.0657, frontier MAE 0.2453 N | unavailable |
| PHYSICAL HISTORY ONLY | P4-B root-heldout / primary | MAE 0.0655, RMSE 0.0893, rank 0.9861 | point OOF; no predictive band | SR 0.9375, agreement 0.8194, force 4.202 N, under 0.0486, utility 0.0879 |
| PHYSICAL HISTORY ONLY (3-member) | DEV root-heldout / primary | MAE 0.1431, RMSE 0.2016, rank 0.8750 | coverage 68/90/95 = 0.6667/0.8750/0.8750; AURC 0.1289 | unavailable |
| VISION + PHYSICAL | root-heldout / primary | not a μ estimator | NLL 0.6301, Brier 0.0967, frontier MAE 0.3459 N | unavailable |
| EXPLICIT SYSID | LOTO task/object-heldout / secondary | MAE 0.1810, RMSE 0.2181, median AE 0.1496, rank 0.8611 | point only; no validated predictive uncertainty | B60 SR 0.8958, agreement 0.5370, force 4.252 N, under 0.0903, utility 0.0273 |
| Phys2Real-style fusion | not available / primary | blocked | no frozen fusion artifact | blocked |
| GT physics | privileged LOTO diagnostic / secondary | exact GT diagnostic | oracle, non-deployable | B60 SR 0.9236, agreement 1.0000, force 4.004 N, under 0.0625, utility 0.1058 |

## Validation findings

- P4-B contains 216 seed-level rows plus 72 derived `ENSEMBLE_MEAN` rows. The primary point metrics above use the 72 derived rows once per context; the previous lane artifact mixed context-level MAE with an all-row RMSE. The corrected context-pooled RMSE is **0.0893**; the member-only 216-row RMSE is **0.1089**, while including the derived rows gives **0.1044**.
- Explicit SysID's frozen source table reports task-balanced macro metrics: MAE **0.18097324543506949**, RMSE **0.2181250286036145**, Spearman **0.7297401261724618**, pair ranking **0.8611111111111112**. The raw pooled audit is MAE **0.1810**, RMSE **0.2261**, median AE **0.1363**, Spearman **0.6389**; both are preserved in `VALIDATION_AUDIT.json`, and the table labels the registered macro choice.
- Vision-only and vision-plus-physical rows are valid root-heldout feasibility/frontier diagnostics. Their NLL/Brier/frontier scores must not be relabeled as friction μ identification, and no same-policy downstream utility run is present.
- LOTO changes task and object family together. It is secondary evidence and cannot by itself establish pure semantic task transfer.
- The PhysicalBelief manifest reports `test_contexts_loaded = 0`; no original TEST rows were used. The manifest's train-fitted interval scale is recorded in the provenance, but the archived row intervals use the source `total_variance` convention; no calibration claim is made.
- No frozen Phys2Real-style visual-prior + interaction-uncertainty fusion checkpoint/prediction/downstream artifact was found. This is a scientific blocker for that requested comparison, not a reason to invent a value.

## Reproducibility and exact artifacts

- Runner: [run_e2_final.py](./run_e2_final.py)
- Table: [TABLE_E2_PHYSICAL_IDENTIFICATION.csv](./TABLE_E2_PHYSICAL_IDENTIFICATION.csv)
- Task audit: [E2_EXPLICIT_SYSID_TASK_AUDIT.csv](./E2_EXPLICIT_SYSID_TASK_AUDIT.csv)
- Validation: [VALIDATION_AUDIT.json](./VALIDATION_AUDIT.json)
- Provenance: [PROVENANCE.json](./PROVENANCE.json)
- Status: [STATUS.md](./STATUS.md)

Overall E2 status is complete for the strongest valid existing evidence, with a scientific negative/blocked result for commensurate visual μ comparison and Phys2Real-style fusion.
