# ActiveForcing experiment dependency graph

This graph records actual gates found on disk. A line ending in `unlock` names the event that releases the next scientific step; it is not authorization to launch it.

## Live resource chain

```text
Frozen pi0 server PID 33931 (shared dependency, protected)
  + Mass formal collector PID 2080255 (protected)
  + E5 scheduler PID 2096169 -> E5 task1 worker PID 2099454
  -> current GPU occupancy at 13:30: 23,181/40,960 MiB, 89% util
  -> blocks any additional E3/E7/Boundary heavy worker
  -> unlock: current worker(s) naturally exit, output QA passes, then a fresh 10-second silence/recheck
```

## E3 long-horizon chain

```text
old 720 LocalLift labels degenerate (720/720 positive)
  -> true long-horizon candidate qualification
  -> task5 TRAIN support PASS, task5 DEV nominal gate FAIL
  -> reserve task2/task8/goal-task3 nominal failures
  -> five-demo onboarding fallback
  -> demo1 PASS -> demo2 PASS -> demo11 PASS -> demo12 PASS -> demo19 PENDING
  -> strict five-demo assembly/conversion
  -> CPU norm stats
  -> 1000-step LoRA from checkpoint 49999
  -> freeze step-999 checkpoint
  -> preregistered nominal FullTask DEV qualification
  -> formal long-horizon stage-labeled collection
  -> matched LocalLift-vs-FullTask Direct training
  -> same-Utility simulator validation
  -> E3_LONGHORIZON_READY_FOR_E2E.json
```

- Current blocker: `demo19` plus resource gate. Conversion, norm and training are not started.
- Actual unlock event: protected Mass/E5 worker state rechecked; no duplicate E3 worker; explicit valid final gate; demo19 replay QA PASS.
- The on-disk V2 template is not authorization. Its `17/60` E5 coverage is stale. No non-template `FINAL_GATE_PASS` file was found.
- The prior scheduler PID `2068026` is historical and absent in the final process snapshot; current scheduler is E5 PID `2096169`.

## E5 fresh Utility chain

```text
V2 runner + Utility hash + reset-to-end QA
  -> 60 preregistered tuples x 5 current arms = 300 rollouts
  -> canonical accepted count 20 tuples / 100 rollouts
  -> in-flight task1 offset03 shard (not counted until QA)
  -> missing 40 tuples / 200 rollouts
  -> balanced task/root/friction allocation
  -> add/execute protocol-required Success-Only, same-transition Query-Ignored,
     Point, Posterior, and re-query comparisons
  -> frozen aggregate + final report
```

- Current blocker: collection is incomplete and the current runner's five arms do not cover the authoritative eight-arm E5 requirement.
- Unlock: each worker must exit naturally and pass `FULL_STATUS` + `SHARD_QA`; coverage auditor must reach the preregistered target with no duplicates/invalid partials.
- `PASS` in `E5_COVERAGE_STATUS.json` means the coverage **audit ran successfully**, not that collection is complete; the same file says `completion_status=INCOMPLETE`.

## E7 continuous planning chain

```text
Utility repair implementation
  -> historical interface calibration
  -> 168 true holdout folds + monotone Direct gate PASS
  -> distinct generator QA + 144-row offline decision matrix PASS
  -> frozen 8-request exact-float tracking/parity block (0/8 executed)
  -> if and only if PASS: 144 matched real POINT/POSTERIOR x K{5,10} rollouts
  -> final scientific verdict
```

- Current blocker: GPU/resource priority and the live preflight/8-rollout gate.
- Unlock: current Mass/E5 heavy workers exit and the frozen 8-request block is authorized after fresh audit.
- The 144 rows currently present are offline planner decisions, not simulator rollouts. `E7_OFFGRID_DEV_RESULTS.csv` is header-only.

## Boundary-aware Direct chain

```text
old-720 CPU coverage audit COMPLETE
  -> minimal force-interface preflight FROZEN_NOT_RUN
  -> adaptive four-task qualification
  -> boundary acquisition
  -> augmented manifest (>0 new branches; currently 0)
  -> three-seed matched OLD vs OLD+BOUNDARY Direct training
  -> same frozen Utility comparison
  -> fresh-root validation and promotion decision
```

- Current blocker: dependency on preflight plus occupied GPU.
- Unlock: `MINIMAL_FORCE_INTERFACE_CALIBRATION_PASS.json` exists and validates 1.0 N, 0.25 N separation, MAE, saturation and peak criteria.
- `DATA_COVERAGE_HYPOTHESIS=NOT_YET_TESTED`; the current state must not be called a negative result.

## E6 re-query chain

```text
three-member friction belief checkpoints (reusable after provenance QA)
  -> member-wise Utility-optimal force recompute
  -> genuine second-query continuation and posterior update
  -> One Query / Always Two / Raw Uncertainty / Utility Consensus / Oracle
  -> matched DEV then locked evaluation
```

- Current blocker: no qualified second-query evidence. Duplicate first-query evidence is forbidden.
- Unlock: a real second-query transition with frozen action/state semantics and measurable uncertainty reduction.
- All rho-induced decision tables are `LEGACY_DIAGNOSTIC_ONLY`.

## E8a Mass to E8b Joint chain

```text
Mass P4-B identifiability PASS
  -> Mass task qualification: task2 PASS; breadth screen otherwise negative
  -> formal task2 180-branch collection IN PROGRESS
  -> formal data QA
  -> mass estimator / Direct / Expected Utility
  -> fresh mass E2E + MASS_FINAL_REPORT/TABLE_MASS
  -> freeze valid mass axis and release GPU
  -> joint 3x3 friction x mass pilot
  -> joint estimator + cross-confusion
  -> joint Direct/Utility + friction-only/mass-only ablations
  -> joint E2E
```

- Mass blocker: current collection is incomplete; later model/controller/E2E stages have not run.
- Joint blocker: depends on valid Mass closure and GPU release. Only collector/analyzer compilation is complete.

## E2 and Phys2Real chain

```text
point physical-history root-heldout OOF + ExplicitSysID/LOTO summaries
  -> complete Physical-only and Vision+Physical fold/seed grid
  -> Vision-only matched baseline
  -> calibrated interaction ensemble/posterior
  -> Phys2Real-style uncertainty fusion
  -> downstream same-Utility decision evaluation
```

- Current blocker: incomplete modality evidence and no executed fusion closure.
- Unlock: a frozen matched context/split contract with all modalities present; do not infer fusion success from existing point-estimator metrics.

## External baselines

```text
FORTE source/checkpoint audit
  -> faithful six-channel tactile + impedance simulator adapter
  -> same tuples -> force telemetry -> realized Utility table

Tabero native executor/checkpoint recovered
  -> exact same-tuple native 13D rollout without force overwrite
  -> force telemetry -> realized Utility table
```

- FORTE is implementation-blocked, not merely GPU-blocked.
- Tabero is data/resource-blocked; its existing smoke is not paired E1 evidence.

## E4

```text
LOTO + B{0,10,20,30,60} + nested acquisition + 108 DEV shards COMPLETE
  -> conclusion: SUBSTANTIAL_TASK_SPECIFIC_DATA_REQUIRED
  -> optional fresh locked confirmation only if paper retains this claim
```

E4 has no dependency that justifies resuming the existing DEV experiment. Freeze it unless the final paper explicitly requires a locked confirmation.
