# Existing generalization audit

The completed main online-VLA evidence is **96 matched contexts / 384 branches / 8 roots**, obtained by pooling two separately frozen 48-context campaigns. The older `final_main_results_v1` is only the first half. Row uniqueness and four-method matching were checked from `confirmatory_v1/POOLED8_BRANCH_RESULTS.csv`.

## A. Root generalization — supported

Feasibility TRAIN roots are 5100,5101,5102,5106 (431 valid rows, 48 contexts); VAL root 6100 and historical TEST root 6103 contain the same task IDs. Main evaluation roots 170040–170047 are disjoint, as are friction-study roots 170052–170053. Fresh-root exposure manifests were frozen before those experiments. This is new reset/context evidence within the same task family, not unseen tasks.

## B. Exact-held-out friction generalization — supported

The completed continuous-friction study contains 4 known tasks × 6 exact-unseen in-support values × 2 fresh roots = 48 contexts, with AF, GT-Physics and Fixed-4 (144 branches). Every test friction was checked absent from both recorded belief and feasibility TRAIN values for that task, and strictly inside their recorded min/max support. Training-row hashes in the archived support audit were rechecked. The variable is object-side material static/dynamic friction, not a measured effective contact-pair coefficient. There is no mass intervention and no new task identity in this evidence.

## C. Object generalization — not demonstrated by the current frozen model's formal evaluations

TRAIN, the pooled main experiment and continuous-friction experiment all manipulate alphabet soup, cream cheese, tomato sauce and butter. Different roots/frictions do not make those unseen objects. The configured Tabero task list includes additional objects, but configured evaluation coverage is not held-out AF evidence.

## D. Task generalization — historical experiment exists; current frozen model evidence does not

An archived August 30 `task_demand_loto_generalization_20260830_044334` experiment explicitly holds each of tasks 0,1,5,6 out of that fold's training and trains different task-demand models. Its `TASK_DEMAND_LOTO_PROTOCOL.json`, training manifest, checkpoint manifest and results are present. Its final decision is `T1_REMAINS_STRUCTURALLY_OUT_OF_DISTRIBUTION`, with both feasibility and joint task-demand go gates false. Thus it would be inaccurate to say no task-held-out work was ever attempted anywhere. It is a different historical architecture/data/runtime, and it does not supply a held-out task result for the authoritative September 6 phase-free ensemble. All four task identities occur in that ensemble's TRAIN set.

## E. Task-form generalization — not previously demonstrated

Both the current evaluations and archived four-task LOTO work remain within lift/transport/place. Scripted task-demand phase descriptors in the historical experiment explicitly cover branch_hold/lift/transit/over_basket/place. Archived generic LIBERO tasks or demonstrations in other projects do not demonstrate transfer by the current frozen AF ensemble.

## Existing-family reference results, never new-form results

| Method | Existing form-A success | Measured squeeze |
|---|---:|---:|
| ACTIVEFORCING | 69/96 (71.88%) | 3.519 N |
| FIXED_3 | 37/96 (38.54%) | 1.762 N |
| FIXED_4 | 65/96 (67.71%) | 3.443 N |
| FIXED_5 | 71/96 (73.96%) | 4.689 N |

The squeeze statistic is the archived mean over non-release observations, averaged by branch; it is not the selected setpoint and not the bilateral-contact-conditional statistic. Existing-family paired comparisons are not evidence about new task forms.

## Sources and reproducibility

- `/home/exouser/FORTE/analysis/results/current_fulltask_feasibility_baseline_v1_20260906/TRAINING_PROTOCOL.json` and `current_matched_stage1_648_v1_20260906/STAGE_MANIFEST.json`.
- `/media/volume/newdata/exouser/online_vla_activeforcing_20260907/confirmatory_v1/{POOLED8_BRANCH_RESULTS.csv,FINAL_CONFIRMATORY_STATUS.json,FINAL_FRESH_ROOT_PLAN.json}`.
- `/media/volume/newdata/exouser/online_vla_activeforcing_20260907/final_continuous_friction_generalization_v1/{TRAINING_FRICTION_SUPPORT_AUDIT.json,DEV_PLAN.json,ROOT_NONEXPOSURE_AUDIT.json,FINAL_CONTINUOUS_FRICTION_RESULTS.json}`.
- `/media/volume/newdata/exouser/Tabero_e3lh/analysis/results/task_demand_loto_generalization_20260830_044334/`.

`audit_generalization.py` reproduces counts and exact-value overlap checks; `evidence/GENERALIZATION_VERIFICATION.json` contains the per-context audit. Historical LOTO existence must be distinguished from a supported positive generalization claim.
