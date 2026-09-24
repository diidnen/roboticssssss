# ActiveForcing P0/P1/P3 Closure Ledger

Owner: coordinator
Start: 2026-09-03 UTC
Numbering: P0 = canonical E5 Fresh Utility E2E; P1 = canonical E3 FullTask vs LocalLift; P3 = canonical E7 Continuous Posterior-Aware Planning.

## Scope locks

- In scope: P0, P1, P3 only.
- Explicitly out of scope: Decision-aware re-query, fixed-two-query, raw-uncertainty re-query, P2, and Joint friction × mass.
- Frozen: π0, evaluator, success criterion, force semantics, hidden-physics labels, train/dev/test split, root grouping, baseline definitions, and Direct/Utility scientific definitions.
- No existing formal process may be terminated, paused, reprioritized, or disturbed.

## Lane state at takeover

| Lane | Current state | Immediate next gate | Owner |
|---|---|---|---|
| P0 | Protected scheduler active; latest direct re-audit 40/60 tuples and 200/300 accepted rollouts, incomplete | Let protected Mass window clear; validate and continue missing tuples | coordinator/scheduler |
| P1 | Candidate checkpoint/onboarding lineage exists; nominal matched DEV not yet accepted | Fresh attestation, checkpoint integrity, then matched DEV if protocol/resource gates pass | P1 agent |
| P3 offline | Existing offline continuous evidence exists; formal 2×2 needs audit/aggregation | Verify legal matched Point/Posterior × Grid/Continuous cells | P3 agent |
| P3 integration | Exact-float retained gate is 4/8 and fail-closed | Trace parity; no 144-rollout matrix before 8/8 | P3 agent/coordinator |

## Live protected processes at baseline

- π0 server PID 2128032, port 18881.
- P0 scheduler PID 3494903.
- P0 worker PID 3663202 at audit time.

## Latest live re-audit

- UTC: 2026-09-03T07:36Z
- P0 scheduler PID 3494903 remains active; latest direct coverage is 40/60 tuples and 200/300 accepted rollouts. The authoritative coverage file is one scheduler audit behind because the scheduler is waiting on the protected Mass worker.
- Protected Mass worker: PID 3838385, `fresh_e2e_corrected_final_root8400_high_v1`.
- π0 server PID 2128032 remains unchanged.
- Coordinator-only post-clear watchers are waiting; no P1/P3 Isaac process has been launched.

## Follow-up audit and P1 collection readiness

- UTC: 2026-09-03T07:54Z
- P0 remains authoritative at 40/60 tuples and 200/300 accepted rollouts; the current protected Mass worker is operating on root 8401/MID and has not been disturbed.
- P1 CPU collector sidecar was added and syntax/import audited. Its compatibility check correctly rejects the legacy P5S0C manual branch driver because it is not a frozen-π0 nominal-VLA evaluator and lacks authoritative stage fields. No P1 manifest or outcome was fabricated.
- P1 post-P0 coordinator and P3 post-P1 coordinator remain alive and fail-closed; no candidate server, P1 Isaac cell, or P3 Isaac rollout has started.
- A second CPU audit confirms the existing B5 wrapper points at a missing source path in this worktree, lacks P1 canonical telemetry/official close/drop fields, and targets the wrong basket/object semantics. A contract-only adapter stub was added; it intentionally cannot produce a formal P1 manifest.
- The nominal wrapper path was corrected to the hash-matched read-only B5 source on `/media/volume/newdata`; wrapper compilation and source SHA audit pass. This repairs nominal qualification only and does not make the formal matched comparison admissible.
- At 08:16Z, task0 offset10 completed runner return 0, shard validator return 0, and atomic link. The live coverage file remains 40/60 until the scheduler's next audit because the protected Mass query-only worker is active.

The current resource snapshot is recorded separately in `RESOURCE_BASELINE.md`.

## CPU-only P3 data-quality and parity re-audit

- UTC: 2026-09-03T08:20Z
- The frozen P3 source table has 144 rows: 9 matched DEV contexts × 2 planning conditions × 4 frozen candidate generators × K={5,10}; the legal primary slice is 72 rows using FIXED_GRID and PROPOSAL_GUIDED. Manifest linkage has zero mismatches, TEST is sealed out, and realized outcomes are absent rather than imputed.
- The regenerated offline audit is `PASS_FOR_OFFLINE_2X2_EXTRACTION`. It remains prediction-only and does not support a realized-SR or realized-utility claim.
- The CPU parity trace still classifies the retained exact-float block as an engineering/runtime scene parity failure: 4/8 valid branches (task0 4/4, task1 0/4). Archived task1 branches were not substituted.
- Current P0 live coverage at the same audit: 41/60 tuples and 205/300 accepted rollouts; scheduler worker is running task1/offset10. P1 and P3 launch coordinators remain waiting.

## Live continuation audit

- UTC: 2026-09-03T08:35Z
- The protected Mass query-only worker completed naturally after writing its v2 protocol/query/row/decision artifacts. P0 immediately resumed under scheduler PID 3494903.
- Current direct coverage: 42/60 unique tuples and 210/300 accepted rollouts; 18 tuples / 90 rollouts remain, and completion status is `INCOMPLETE`.
- Current E5 worker: task5 / tuple offset10, PID 176191; it is active at approximately 194% CPU and GPU memory remains below the resource ceiling.
- P1 and P3 coordinators are still waiting; no P1/P3 GPU or Isaac process has been launched. The frozen π0 server remains PID 2128032.
- A fresh CPU-only P3 integration audit was executed at 08:37Z. It preserves the exact-float result as `4/8` with task-1 contact-edge runtime parity failure, confirms the 8-request small manifest and 144-request full manifest are structurally valid/no-TEST, and does not authorize the full matrix.
- P1 CPU semantic preflight (08:40Z) confirms the formal lane remains fail-closed: the available P4-B runtime is `libero_object/task5` tomato-sauce→basket, not the registered `libero_10/task5` black-book→desk-caddy task, and the current protocol leaves task-5 `Fmax_N` undefined while disabling the final force-selection claim. No branch manifest or outcome was generated.
- At 08:41Z, E5 task5/tuple10 completed with runner and shard-validator return code 0; coverage re-audited to 43/60 tuples and 215/300 rollouts. The scheduler immediately launched task6/tuple10 (PID 259531), with 17 tuples / 85 rollouts remaining.
- The CPU QA audit independently confirmed the 43/60 snapshot is structurally clean but `NOT_READY_INCOMPLETE`; 38 accepted shard entries have no duplicate tuple×method keys, lineage is consistent, and the aggregate contract correctly rejects incomplete input. P1 preflight and P3 integration audit are now recorded as fail-closed CPU evidence.

## P1 user-authorized simplified protocol amendment

- UTC: 2026-09-03T08:58Z
- P1 is now a new tuple-only collection contract, explicitly authorized by the user: one registered task `libero_10/task5` (`black_book_1 -> desk_caddy_1`), fixed force range/grid `[1,8]N` / `{1,...,8}N`, `Fmax=8N`, and Utility hash `555e11c038d9c896fe998ea15177e7d887afe6fe43751c4e2f91275b21f3ae1c`.
- The collection grain is one row per `(context_id, force_N, replicate_id)` containing the matched `(x,z,F,y_lift,y_full)`; old P4-B data and re-query branches are prohibited.
- `y_lift` is pre-registered as the frozen B5 local-lift predicate `object_dz >= 0.03m`; `y_full` remains the unchanged official success evaluator latch. Delayed failure is exactly `y_lift=1 && y_full=0`.
- CPU validation: protocol 5/5 tests PASS; runtime source transform/preflight PASS; no P1 outcome exists yet.
- A 9-context / 72-branch plan and a sequential post-nominal collection coordinator are prepared. The existing post-P0 coordinator will run this simplified lane before exiting, keeping the P3 watcher from overlapping GPU use.
