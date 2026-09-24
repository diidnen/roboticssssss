# E3 task5 post-onboarding runner and QA readiness

Status: **CPU_STATIC_QA_READY; EXECUTION_FORBIDDEN_PENDING_IDS12_19_CONVERSION_NORMALIZATION_TRAINING_AND_DUAL_COORDINATOR_SECOND_SILENT_GO**.

This package implements the preregistered `E3_POST_ONBOARDING_FULLTASK_VS_LOCALLIFT_V1` nominal DEV gate without running it. It preserves roots `7600..7604`, fixed robust diagnostic force 8 N, official FullTask pass threshold `>=3/5`, and query-state reach threshold `>=4/5`. Friction is frozen at `mu=0.6`, matching the earlier task5 nominal diagnostic convention; 8 N remains an engineering qualification point, not Fmax, Utility, or a safety limit.

## Candidate-lock and policy identity

`validate_and_lock_task5_onboarded_candidate.py` preselects terminal training step 999 before any nominal DEV outcome, verifies the exact isolated OpenPI commit/config/transform and normalization schema, and hashes every file in the selected checkpoint tree. Its output status is only `CANDIDATE_LOCKED_FOR_NOMINAL_DEV_NOT_FINAL_FREEZE`.

`serve_task5_onboarded_candidate_pi0.py` uses the same project `create_trained_policy` and websocket server path, but adds read-only identity metadata: task, config, step, checkpoint-tree digest, candidate-lock digest, and OpenPI commit. `run_e3_post_onboarding_nominal_wrapper.py` refuses inference unless every metadata field matches the locked candidate. It then records the same fingerprint in episode telemetry. The wrapper preserves nominal motion from pi0, changes only the two normal-force setpoint slots to the fixed 8 N diagnostic value, and keeps the low-level controller and official evaluator unchanged.

## Coordinator final-gate V2 and resource firewall

The original non-authorizing `E3_SECOND_SILENT_GATE_TEMPLATE.json` and 120-second validator remain immutable lineage from readiness V1, but are no longer accepted by either launcher. `E3_SECOND_SILENT_GATE_COORDINATOR_OVERRIDE_20260902.{json,md}` records the explicit operational override without changing scientific inputs or outcomes.

V2 requires external E5 PID `2059639` to end naturally, zero active E5 PIDs, at least 10 seconds of observed silence, an immediate final protected-process/resource/duplicate recheck, and explicit coordinator `FINAL_GATE_PASS` no older than 60 seconds. Core E5 PID `2059513` exit `-9` partial remains rejected and coverage remains `17/60`. Separate V2 authorization is required for candidate-server start and for every one-root nominal DEV cell; launchers also require the root, core-coordinator, and final-gate environment tokens.

The launch scripts additionally reject any live E5 process, port/server duplicate, reused output directory, or numeric GPU-gate failure. Server start is conservatively limited to utilization below 30% and free memory above 28 GiB. Each Isaac cell requires utilization below 50% and free memory above 18 GiB. Natural termination of the current E5 jobs is never itself authorization.

## Nominal DEV analysis and final promotion

`analyze_task5_post_onboarding_nominal_dev.py` requires exactly one complete episode/step pair for every root, exact config/step/lock fingerprints, 8 N and `mu=0.6` telemetry, distinct root-state hashes, empty telemetry-error logs, and the preregistered `3/5` FullTask plus `4/5` query gate. It writes new outputs only and never overwrites.

Only a passing nominal DEV gate can feed `promote_task5_onboarded_candidate_after_nominal_dev.py`. Promotion rehashes the checkpoint and normalization assets and refuses any byte change after candidate lock. The promoted manifest requires the exact same checkpoint/config/norm fingerprint for every later LocalLift, FullTask, fixed-force, GT-physics, and ActiveForcing comparison. A failed nominal DEV gate triggers the preregistered 10-demo nominal-only expansion; it cannot trigger checkpoint cherry-picking or TEST/force-method inspection.

No GPU process, demo12/19 replay, conversion, normalization, training, candidate lock, server, DEV rollout, or final freeze was executed while preparing this readiness package.

## CPU-only QA result

- All Python files compile with an isolated `/tmp` bytecode cache; both shell launchers pass `bash -n`.
- `preflight_task5_post_onboarding_readiness.py`: `E3_POST_ONBOARDING_READINESS_PREFLIGHT_PASS`.
- `test_task5_post_onboarding_readiness.py`: `E3_POST_ONBOARDING_ANALYZER_PROMOTION_UNIT_PASS` on a synthetic five-root fixture, including exact lock/gate fingerprint propagation and unchanged-byte promotion.
- The non-authorizing gate template is rejected; both launchers exit before output creation when dual GO tokens are absent.
