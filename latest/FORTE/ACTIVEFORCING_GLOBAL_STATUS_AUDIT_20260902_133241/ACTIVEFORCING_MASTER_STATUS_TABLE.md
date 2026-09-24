# ActiveForcing master status table

Snapshot basis: `2026-09-02T13:38:14Z` process state and status artifacts read through that cutoff. Percentages are milestone-weighted, not confidence estimates. E0-E8 use the experiment numbering in the audit request; the older closure bundle used a different numbering convention and is not used for the row names below.

| Experiment | Progress | What is actually complete | What remains | Current truth | Running | GPU wait/use | Next action |
|---|---:|---|---|---|---|---|---|
| E0 | 75% | Frozen pi0 checkpoint 49999, preprocessing/action contract, setpoint-only adaptation, action-side-channel invariance | matched controller/reset hashes and live off-grid interface certification | PARTIAL | pi0 service only | no | certify before locked E7 |
| E1 | 80% | 720 branches = 144 paired episodes x 5 forces; Utility DEV benchmark and controls | native default, faithful FORTE/Tabero, fresh locked closure | PARTIAL; DEV COMPLETE | no | no | freeze DEV table; close with fresh evidence |
| task6 diagnosis | 100% | Direct low-boundary overconfidence primary; Utility secondary; global Fmax only 1/5 rescue | separate Boundary hypothesis only | COMPLETE_NEGATIVE; METHOD_PROMOTION=NO | no | no | freeze |
| FORTE | 25% | reproduction audit | faithful sensor/actuator adapter and paired rollout | IMPLEMENTATION_BLOCKED | no | later | adapter first |
| Tabero | 50% | native executor/checkpoint; smoke | exact paired tuples/results | DATA_BLOCKED | no | yes | queue later |
| Boundary Direct | 14% | CPU Phase 1 audit only | preflight, new data, augmented set, 3-seed train, Utility compare, fresh validation | DEPENDENCY_BLOCKED; DATA_COVERAGE_HYPOTHESIS=NOT_YET_TESTED | no | yes | force preflight first |
| E2 | 55% | point physical-history OOF, ExplicitSysID and learned LOTO summaries, partial 3-seed matrix | complete modality/LOTO/calibration/posterior/downstream matrix | PARTIAL | no | no | reconcile missing folds |
| Phys2Real fusion | 0% | no executed closure found | all four stages | NOT_STARTED | no | no | freeze contract first |
| E3 | 35% | label audit; task search; 17 TRAIN cells; failed six-cell DEV gate; demo 1/2/11/12 PASS | demo19 then conversion/norm/LoRA/DEV/formal data/two-model comparison | WAITING_ONBOARDING + RESOURCE_BLOCKED | no | yes | demo19 after resource recheck |
| E4 | 80% | all 108 DEV shards, LOTO and B=0/10/20/30/60 | only locked Utility confirmation if retained | COMPLETE_NEGATIVE_DEV_ONLY | no | no | freeze conclusion |
| E5 | 47% scientific; 33.3% collection | 20/60 tuples, 100/300 rollouts accepted | 40 tuples, 200 rollouts, balance, missing required arms, final report | IN_PROGRESS | scheduler + task1 worker | using | natural end then QA |
| E6 | 30% | belief infrastructure and legacy rho disagreement diagnostic | Utility recompute, real second query, five-arm evaluation | DIAGNOSTIC_ONLY | no | yes | obtain genuine second-query evidence |
| E7 | 60% | implementation, offline Direct gate, generator QA | 8 exact-float + 144 matched real rollouts | OFFLINE_GATE_PASS; REAL_ROLLOUT_PENDING | no | yes | frozen 8-rollout block |
| Mass | 44% | query identifiability, task2 qualification, broader-task negative screen | finish 180 formal branches, QA, estimator/Direct/Utility/E2E | IN_PROGRESS | PID 2080255 | using | natural end and validate |
| Joint physics | 10% | collectors/analyzers compile | 3x3 data through E2E | DEPENDENCY_BLOCKED | no | yes | wait for Mass |
| Force calibration | 60% | historical TRAIN-only contact/load-bearing mapping and resolution | current live preflight and locked parity | PARTIAL_HISTORICAL_ONLY | no | yes | minimal preflight |

## Milestone arithmetic

- E7 follows the requested five equal milestones: implementation, offline Direct gate, candidate QA, real small block, matched real rollout/final verdict. It is `3/5 = 60%`.
- E5 uses five 20-point milestones: V2 runner/freeze (20), engineering/DEV gates (20), collection scaled by `20/60` (6.7), complete required-arm balance (0), broad locked/final closure (0). Rounded: `47%`. Its raw collection completion is separately `20/60 = 33.3%`.
- Boundary uses seven gates; only the old-data CPU audit is complete: `1/7 = 14%`.
- E3 credits task/label recovery, candidate qualification, and 4/5 demo replay, but not conversion, training, post-training DEV, formal collection, matched model training, or final validation.
- E8 paper progress treats Mass and Joint as separate rows. For the single E8 block in the global average, their mean is used.

## E1 verified table

These are TRAIN/root-heldout-OOF DEV results, not sealed TEST:

| Method | n episodes | SR | selected force | realized Utility |
|---|---:|---:|---:|---:|
| ActiveForcing | 144 | 93.06% | 4.0628 N | 0.1056 |
| Query-Ignored | 144 | 93.75% | 4.4667 N | 0.0418 |
| Fixed-Max | 144 | 98.61% | 4.8329 N | 0.0189 |
| GT-friction Utility | 144 | 95.14% | 4.0327 N | 0.1374 |

The often-repeated “720 paired branches” means 720 unique candidate branches, organized as 144 paired episodes with five candidates each. It is not 720 independent episodes.

## E2 sub-status

| Component | Truth |
|---|---|
| Vision-only | no complete all-task final E2 row found; older visual-context work is diagnostic |
| Physical-only | 3 folds x 3 seeds root-heldout checkpoints exist; LOTO and final reporting are incomplete |
| Vision+Physical | root-heldout checkpoints exist; LOTO checkpoint grid is incomplete and no final metric table exists |
| Explicit SysID | DEV LOTO complete: MAE 0.1810, RMSE 0.2181, Spearman 0.7297, pair ranking 0.8611 |
| LearnedProbe LOTO | DEV complete but negative: MAE 0.5149, RMSE 0.5665, Spearman 0.5258, pair ranking 0.7917 |
| Probe-PhysicalHistory point | strongest current root-heldout OOF: MAE 0.06547, RMSE 0.10438, Spearman 0.9018 |
| Ensemble/posterior | members/checkpoints reusable; sigma/posterior semantics remain diagnostic and calibration gate is not final |
| 5-seed | not found as complete authoritative matrix |
| Phys2Real fusion | no executed visual-prior/interaction/fusion/downstream chain found |

## E3 correction against stale status

The gate template still says `locked_test_coverage=17/60` and the old blocker file says replay not started. Both are stale. The latest accepted evidence is demo IDs `1,2,11,12`; ID12 passed QA at `13:27:38Z`. Only ID19 remains. No conversion, normalization, LoRA training, formal long-horizon collection, or matched LocalLift-vs-FullTask training has occurred. `E3_LONGHORIZON_READY_FOR_E2E.json` does not exist.

## E5 correction against optimistic closure snapshots

The canonical coverage auditor says `completion_status=INCOMPLETE`, `accepted_unique_tuples=20`, `accepted_unique_rollouts=100`, `missing_tuples=40`. Four one-tuple-per-task locked shards pass shard QA, but they do not turn the planned 60-tuple campaign into a balanced final test. The current five-arm runner contains Native Default, Fixed-Max, No-Query Training Prior, ActiveForcing 1Q, and GT. It does **not** contain the complete required set of Success-Only, same-transition Query-Ignored, Point, Posterior, and re-query arms.

## Force interface

The historical TRAIN-only calibration covers 720 trajectories and 360 native commands. Contact/load-bearing median MAE is 0.1922 N, median bias -0.1243 N, and taskwise effective resolution is 0.04-0.16 N over task-specific upper ranges. This is not the live minimal preflight: the current controller has not yet passed the required 1.0 N stability, 0.25 N adjacent separation, MAE <=0.4 N, no saturation, and peak <=8 N gate.
