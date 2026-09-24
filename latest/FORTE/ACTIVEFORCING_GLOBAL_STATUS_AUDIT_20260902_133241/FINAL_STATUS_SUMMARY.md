# ActiveForcing global experiment status — final summary

Final state: `ACTIVEFORCING_GLOBAL_EXPERIMENT_STATUS_RECOVERED`

Snapshot cutoff: `2026-09-02T13:38:14Z`. The project is live and external schedulers may change state after this timestamp; PID and coverage statements below are point-in-time truths, not predictions.

## Bottom line

The project is **not globally complete**. The current final scientific selector is `EXPECTED_UTILITY`, not `MIN_RELIABLE_RHO`. The most mature core evidence is E1/E4 DEV work and task6 diagnosis. The principal unfinished submission blockers are E5, E3, E7, Mass, E6, Boundary and joint physics. Milestone-weighted E0-E8 completion is approximately **54%**.

## 1. What is running now?

Four ActiveForcing-related processes were live at the cutoff:

1. frozen pi0 server PID `33931`, 8640 MiB;
2. Mass task2 formal collector PID `2080255`, 7564 MiB;
3. E5/E3 resource scheduler PID `2096169`, CPU/wait-orchestration only;
4. E5 task1 offset03 worker PID `2099454`, 6880 MiB.

The A100 was 95% utilized with 23,181 MiB allocated. No E3 worker, E7 worker, Boundary job, training job, joint job, or external-baseline rollout was active.

## 2. What has really finished?

- task6 mechanism diagnosis: complete; Direct boundary overconfidence is primary, Utility is secondary, global Fmax is minor, estimator/pi0 are not primary.
- global-force-scale study: complete negative for promotion; `METHOD_PROMOTION=NO`.
- E1 current-Utility paired DEV reconstruction: complete at the DEV level, 144 episodes/720 candidate branches.
- E4 archived transfer study: all 108 DEV shards complete; negative conclusion is `SUBSTANTIAL_TASK_SPECIFIC_DATA_REQUIRED`.
- old Joint/DecisionAligned, WM residual, Probe-conditioned WM, predictive verifier and 60-rollout sample-efficiency questions: complete negative/diagnostic and should be frozen.

None of those DEV/negative completions automatically means paper-ready locked closure.

## 3. What is only CPU/DEV complete but lacks simulator/final test?

- E1: DEV table complete; sealed fresh locked closure and faithful baselines missing.
- Boundary: only CPU Phase 1 complete; no new boundary branch exists.
- E2: existing OOF/LOTO summaries and partial checkpoint grids; modality/calibration/downstream matrix incomplete.
- E4: DEV complete; no final locked confirmation.
- E6: only legacy disagreement diagnostic and infrastructure; no scientific second-query test.
- E7: offline repaired gate complete; zero real 8-block and zero real 144-block rollouts.
- Force calibration: historical TRAIN-only certification, not current live preflight.

## 4. What is waiting for GPU?

E3 demo19/onboarding training, E7 exact-float validation, Boundary preflight/collection, E6 real second-query work, joint physics, Tabero paired baseline, and later Mass downstream E2E. E5 and Mass are not waiting; they are already using the GPU.

## 5. What is waiting on another experiment/output?

- E3 conversion/training waits on demo19 replay QA; formal comparison then waits on the onboarding checkpoint passing nominal DEV.
- Joint physics waits on valid Mass formal closure and a frozen mass axis.
- Boundary collection waits on live force-interface preflight.
- E7 144-rollout matrix waits on the 8-rollout exact-float block.
- E6 final evaluation waits on qualified second-query evidence.
- FORTE rollout waits on a faithful simulator adapter.

## 6. How complete is E5?

Canonical accepted coverage is **20/60 tuples = 100/300 rollouts**. Remaining: **40 tuples = 200 current-arm rollouts**. That is 33.3% raw collection completion and 47% milestone-weighted scientific progress. The task1 offset03 worker is still in flight and not counted. The current five-arm runner also lacks the complete authoritative E5 comparison set, especially same-transition Query-Ignored, Success-Only, Point, Posterior and re-query arms. There is no final broad `TABLE_E5` or final report.

Four locked shards (one LOW/root03 tuple per task) pass QA, but they are only four tuples and do not make the 60-tuple population balanced or complete.

## 7. Where is E3 blocked?

The old 720 LocalLift archive is unusable for the contrast because every LocalLift label is positive. The true long-horizon task5 TRAIN pilot has all three regimes, but its six-cell preregistered DEV gate failed with no FullTask success. Reserve task2, task8 and goal-task3 also fail nominally.

Onboarding is now **4/5 demos complete**: IDs 1, 2, 11 and 12 pass; only ID19 remains. Conversion, normalization, LoRA training and post-training DEV have not started. No formal long-horizon dataset and no matched LocalLift-vs-FullTask models exist. `E3_LONGHORIZON_READY_FOR_E2E.json` is absent.

## 8. Where is E7 blocked?

The new Utility repair supersedes the old negative. It has a PASS Direct gate, historical force-interface calibration, 168 held-out folds, distinct generator QA, and a 144-row offline planner matrix containing 104 off-grid decisions. Those are **not simulator rollouts**. The 8-request exact-float result file is empty/header-only, so the required 8-rollout block and subsequent 144-rollout matched block are both pending. Status: `OFFLINE_GATE_PASS / REAL_ROLLOUT_PENDING / SCIENTIFIC_VERDICT_PENDING`.

## 9. Is Mass complete?

No. The query-identifiability result is positive and task2 qualification passed, while broader-task qualification failed to find a second robust task. The formal task2 collector is currently running. At the cutoff, 70/180 branch rows were persisted. No formal Mass Direct, final Utility controller, fresh E2E, `MASS_FINAL_REPORT`, or `TABLE_MASS` exists. Disappearance of PID 2080255 later must not be treated as success without file QA.

## 10. Where is joint physics?

Only isolated 3x3 collector/analyzer implementation and compile checks exist. No 3x3 friction x mass data directory, joint estimator, joint Direct, joint controller, one-axis ablation, or joint E2E was found. It is `DEPENDENCY_BLOCKED` on Mass, about 10% complete. The old neural `Joint` is not joint-physics evidence.

## 11. Do FORTE/Tabero baselines have real results?

No faithful paired final results.

- FORTE: only reproduction audit. The archive lacks the six-channel analog tactile/SVR and Dynamixel impedance semantics required for faithful simulation.
- Tabero: the native 13D closed-loop runner and frozen checkpoint are recoverable and smoke evidence exists, but there is no exact E1 same-tuple rollout set without force override.

Therefore SR, mean measured force, peak force and realized Utility remain NA for both in the paired table.

## 12. Did Boundary-aware collection gather new data?

No. The augmented manifest records `720 old + 0 new`. `DATA_COVERAGE_HYPOTHESIS=NOT_YET_TESTED`. The mandatory current-controller force preflight is `FROZEN_NOT_RUN`, so no adaptive qualification, new collection, augmented retraining, 3-seed comparison or fresh validation has occurred.

## 13. What should never be resumed?

- hard-rho/minimum-passing E6/E7 and E5 variants as final methods;
- old nearest-grid continuous E7 validation;
- legacy Q2F E2E as current E5 evidence;
- old neural Joint as mass x friction evidence;
- old 720 LocalLift classifier plan;
- observed task0 roots 5174-5183 as sealed TEST;
- global-scale Utility promotion, WM residual/default, predictive verifier and 60-rollout sample-efficiency repeats.

Raw data/checkpoints may be reusable where explicitly allowed, but selector-dependent results stay `LEGACY_DIAGNOSTIC_ONLY`.

## 14. If only three tasks continue today

1. protect, finish and QA current **Mass formal collection**;
2. finish/repair **E5 V2**, starting with the in-flight shard and exact coverage/arm audit;
3. complete **E3 demo19 -> CPU conversion/norm -> onboarding training gate**.

E7's 8-rollout block is the first reserve.

## 15. Overall paper E0-E8 completion

Approximately **54%** by explicit milestone arithmetic:

`E0 75, E1 80, E2 55, E3 35, E4 80, E5 47, E6 30, E7 60, E8 mean(Mass 44, Joint 10)=27`; equal-weight mean across E0-E8 is 54.3%.

This percentage credits reusable DEV evidence but gives no completion credit for labels such as `PASS` or `COMPLETE` when required simulator/locked stages are absent.

## Authoritative protocol recovered

- version: `ACTIVEFORCING_FULL_CLAIM_V2_UTILITY`;
- final selector: `EXPECTED_UTILITY`;
- pi0: frozen `pi0_lora_tacfield_tabero`, checkpoint 49999, fixed norm/preprocessing and native action semantics;
- query: P4-B; Query-Ignored must share the physical transition and withhold evidence only;
- Direct: shared full-task feasibility;
- Utility: `p(success)*(1-F/Fmax) + (1-p(success))*(-1)`, lower-force tie break;
- posterior: marginalize Utility over belief members, not probability-threshold crossings;
- re-query: disagreement among member-wise Utility-optimal force decisions;
- continuous: exact float execution, never nearest-grid replay;
- mass/joint: fresh mass and 3x3 friction x mass evidence required.

The two copies of the authoritative protocol at `/home/exouser` and `/home/exouser/FORTE` have identical SHA-256 `4326fd66b637e060f54c2ad1fce1d74e3242e4d2c7679dcff98246b3d99d9e95`.

## Repository/worktree audit

- FORTE main: commit `7f88d0184c1617ed95e67502da96e60be07b3689`, branch `main`, dirty with many untracked experiment artifacts.
- Worktrees: E1 diagnosis (`deeecde`, clean), E3 (`afe8248`, modified status/report files), E6/E7 (`7f88d`, untracked closure artifacts), Mass (`7f88d`, untracked code/results), training farm (`7f88d`, untracked analysis), E1 final (`3cbbc4e`, clean).
- Tabero main is dirty with many untracked timestamped p6g0 result directories.
- No repository files were modified by the audit outside the new audit artifact directory.

## Data-quality conflicts resolved

1. E3's template `17/60` is stale; canonical E5 coverage is 20/60.
2. E3's old blocker says replay not started; latest gates show demos 1/2/11/12 passed.
3. E6's filename/status says complete negative, but its own text says no strategy was scientifically evaluated; final status is diagnostic only.
4. Old E7 negative is superseded by the new Utility repair, but the repair still has zero real rollout results.
5. A closure snapshot describes E5 as complete-with-scope-caveat from 21 contexts/102 rows; the preregistered population auditor remains authoritative and says 20/60 incomplete.

## Audit coverage

The scan covered `/home/exouser/FORTE`, `/home/exouser/Tabero`, `/home/exouser`, every `*activeforcing*`, `*FORTE*`, and `*Tabero*` worktree/result root visible there, plus the live `/media/volume/newdata/exouser/activeforcing_*` outputs referenced by manifests and processes. It inspected timestamped result directories, status/report/QA JSON/CSV/Markdown, manifests, launchers, scheduler logs, gate templates/results, Git branches/worktrees/status, and host processes/GPU/tmux/screen state. The concurrently generated `activeforcing_full_claim_closure_20260902_133310` was checked last; its own requirement audit says `NOT FULLY CLOSED` and adds no gate that changes the conclusions above.
