# Task6 failures are a low-boundary Direct-calibration failure, not a friction-identifier failure

## Verdict

`TASK6_PRIMARY_FAILURE_SOURCE = DIRECT`

Secondary source: `UTILITY`, specifically the intended force cost acting on an overconfident Direct probability at the lowest candidate. `IDENTIFIER`, `PI0`, `FORCE_RANGE`, `CONTROLLER`, and `LABEL/EVALUATOR` are not supported as the primary explanation by this archive.

Task6 ActiveForcing succeeds in 31/36 episodes (86.1111%), versus 34/36 (94.4444%) for Query-Ignored and 36/36 (100.0000%) for Fixed-Max. There are exactly 3 Query-Ignored-success -> Active-failure paired cases. In every one, substituting GT friction leaves the selector on the identical failing low-force branch. That rules out the identifier as the cause of those paired regressions. Direct assigns the failing branch p(success)=0.9394–0.9968; the next archived candidate, 3.277–3.345 N, succeeds.

Across all five Active task6 failures, the selected force is the context's lowest candidate (about 3.01–3.06 N); Fixed-Max succeeds 36/36. This is concentrated low-boundary under-force. It is not lack of force support: all paired rescue forces are below the current 4 N cap and the maximum archived candidate succeeds.

## Case-level evidence

The authoritative case table is `TASK6_FAILURE_CASE_AUDIT.csv`. Candidate vectors contain every force, Direct probability, Expected Utility, and observed branch label. The mutually exclusive primary attribution for all 3 paired regressions is `D.DIRECT_MODEL_MISCALIBRATION`; `E.UTILITY_FORCE_COST_TOO_AGGRESSIVE_CONDITIONAL_ON_OVERCONFIDENT_P` is secondary.

| case_id | gt_friction_mu | predicted_friction_mu | active_force_N | gt_friction_utility_force_N | query_ignored_force_N | frontier_N | active_p_success | active_expected_utility | active_failure_stage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pv_train_t6_r02_s5102_low_mu0.225446__repeat2 | 0.2254 | 0.2559 | 3.0193 | 3.0193 | 3.3178 | 3.3178 | 0.9968 | 0.2412 | full_task_rollout |
| pv_train_t6_r03_s5103_low_mu0.298364__repeat2 | 0.2984 | 0.2167 | 3.0272 | 3.0272 | 3.3447 | 3.3447 | 0.9394 | 0.1678 | full_task_rollout |
| pv_train_t6_r05_s5105_low_mu0.260231__repeat1 | 0.2602 | 0.4175 | 3.0178 | 3.0178 | 3.2774 | 3.2774 | 0.9876 | 0.2301 | full_task_rollout |

## Fmax=4 N audit

The frozen config and development protocol both set task6 Fmax=4 N. The config hash is `c5bc4f39861a84b9ba95d55cdb5f8633a8b004d205b3864a02799340653d9b10` (verification: `True`). Git history contains task6=4 N as `ROBUST_FORCE_BY_TASK` in Tabero commit `80ab3be`, while the candidate-generation source defines task6 support as `(3.0, 4.0)`. No inspected source labels 4 N as a physical or safety limit. Separate simulator sensor-visualization and damage thresholds exist elsewhere and do not establish this value as safety-critical.

Therefore, the defensible interpretation is: task6 Fmax is a historical robust-force / candidate-support endpoint reused for Utility normalization and task-specific configuration. Its provenance does **not** support calling it a physical safety limit. Candidate support is sufficient to rescue the audited failures; the issue is selection within support, not an unavailable >4 N action. No TEST outcome was used to change Fmax, and this report makes no Fmax change.

## Controller and evaluator checks

The raw controller telemetry reports large tracking MAE on failed low-force branches, but that statistic includes the post-contact-loss zero-force tail. It is consequence-contaminated: after the object is lost, measured contact force is zero by construction. Transient `measured_force_mean_N` also includes the branch-hold load spike (peaks near 20 N), so it is not a setpoint-tracking statistic. Successful task6 branches have stable steady-state tracking; the archive does not independently establish a low-level controller fault.

The paired higher-force branches succeed under the same state/repeat, so evaluator error is not needed to explain the regressions. Some component labels (`transport_retention`, `lost_in_transit`, `place_success`) have collector-specific semantics and should not be over-interpreted; the controlling binary label is the archived full-task outcome.

## Limits

These are grouped-root OOF TRAIN/development post-query matched branches, not sealed TEST and not native pi0 E2E. The diagnosis is complete for the E1 force-selection anomaly; it does not prove independent nominal pi0 task6 capability.

`TASK6_FAILURE_DIAGNOSIS_COMPLETE`
