# STATUS

TASK0_CONTEXT_MEMORIZATION_SUPPORTED

# GPU SIDECAR EXECUTION

GPU 0 (A100 40GB) was selected after an 8GB residual-memory safety gate. The original GPU processes were not killed, paused, restarted, or reconfigured. The task0 side-car completed without an OOM and exited before frozen evaluation, which ran on CPU.

# AUTHORITATIVE TASK0 DEV

90/90 branches passed the staging and post-commit audits: two contexts, nine forces per context, five repeats per force, exact snapshot parity, and aligned pre-probe RGB/visual features.

# SAME-OBJECT/TASK SCOPE

This is held-out-root evaluation within the same object/task distribution. It is not cross-object generalization.

# REAL FORCE FRONTIERS

| context_id | real_frontier_N | frontier_supported | rho | tested_force_min_N | tested_force_max_N |
|---|---|---|---|---|---|
| pv_dev_t0_r06_s5106_low_mu0.288234 | 4.25 | True | 0.8 | 3 | 5 |
| pv_dev_t0_r07_s5107_low_mu0.294781 | 3.75 | True | 0.8 | 3 | 5 |

# BASE FEAS

Probability MAE 0.1824; Brier 0.1113; NLL 0.4814; frontier MAE 0.400N; under-force 0.500; monotonic contexts 1.000.

# VISUAL RESIDUAL

Probability MAE 0.1802; Brier 0.1084; NLL 0.4661; frontier MAE 0.400N; under-force 0.500; monotonic contexts 1.000.

# FULL VISUAL FEAS

Probability MAE 0.1590; Brier 0.0904; NLL 0.3865; frontier MAE 0.300N; under-force 1.000; monotonic contexts 1.000.

# VISUAL JOINT

Probability MAE 0.2110; Brier 0.1446; NLL 1.1700; frontier MAE 0.450N; under-force 1.000; monotonic contexts 1.000.

# TRAIN VS DEV

TRAIN ordering is reported only as in-sample context; the classification below is determined by frozen held-out DEV rules and the preregistered context-heldout diagnostic.

# DOES VISUAL CONTEXT HELP?

NO under the frozen held-out decision rule. Full Visual improves probability MAE by 12.85% and frontier MAE by 0.10N versus Base, but worsens under-force from 0.50 to 1.00, does not benefit both DEV contexts, and receives no support from the preregistered root-heldout CV diagnostic.

# OFFSET OR x × F?

NEITHER mechanism is established on held-out evidence. Residual recovers only 9.20% of Full's probability-MAE gain; Full's point estimates exceed Residual, but the safety and multiple-context requirements for x × F support fail.

# DOES JOINT PROVIDE INDEPENDENT VALUE?

NO. Joint is 32.72% worse than Full Visual in probability MAE and 0.15N worse in frontier MAE, with no multi-context benefit.

# PROBABILITY

| model | aggregation | cells | physical_repeats | probability_MAE | Brier | NLL | signed_bias | calibration_intercept | calibration_slope |
|---|---|---|---|---|---|---|---|---|---|
| PROSPECTIVE_BASE_FEAS | ENSEMBLE | 18 | 90 | 0.182401 | 0.111298 | 0.481404 | 0.155573 | -3.37038e+07 | 2.29625e+07 |
| VISUAL_INTERCEPT_RESIDUAL | ENSEMBLE | 18 | 90 | 0.180243 | 0.108381 | 0.466076 | 0.150877 | -1.67548e+07 | 2.51713e+06 |
| VISUAL_CONTEXT_FULL_FEAS | ENSEMBLE | 18 | 90 | 0.158957 | 0.0903926 | 0.386491 | 0.157679 | -9.45102e+06 | 1.78024e+06 |
| VISUAL_CONTEXT_JOINT | ENSEMBLE | 18 | 90 | 0.210963 | 0.144605 | 1.16995 | 0.210962 | -8.74755e+07 | 8.12741e+06 |

# FRONTIER MAE

| model | DEV_contexts | valid_real_frontier_count | valid_real_frontier_coverage | finite_decision_count | finite_decision_coverage | frontier_MAE_N | under_force_count | under_force_rate | mean_under_force_magnitude_N | excess_force_count | mean_excess_force_N | within_0.25N_rate | within_0.50N_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PROSPECTIVE_BASE_FEAS | 2 | 2 | 1 | 2 | 1 | 0.4 | 1 | 0.5 | 0.65 | 1 | 0.15 | 0.5 | 0.5 |
| VISUAL_INTERCEPT_RESIDUAL | 2 | 2 | 1 | 2 | 1 | 0.4 | 1 | 0.5 | 0.65 | 1 | 0.15 | 0.5 | 0.5 |
| VISUAL_CONTEXT_FULL_FEAS | 2 | 2 | 1 | 2 | 1 | 0.3 | 2 | 1 | 0.3 | 0 | 0 | 0.5 | 0.5 |
| VISUAL_CONTEXT_JOINT | 2 | 2 | 1 | 2 | 1 | 0.45 | 2 | 1 | 0.45 | 0 | 0 | 0.5 | 0.5 |

# UNDER-FORCE

Selected backend: VISUAL_INTERCEPT_RESIDUAL. Gate status: FAIL.

# MONOTONICITY

| model | context_id | spearman_force_probability | adjacent_ordering_rate | nonmonotonic_steps | context_monotonic | safe_to_unsafe_reversals | dense_steps |
|---|---|---|---|---|---|---|---|
| PROSPECTIVE_BASE_FEAS | __AGGREGATE__ | 1 | 1 | 0 | 1.0 | 0 | 80 |
| VISUAL_INTERCEPT_RESIDUAL | __AGGREGATE__ | 1 | 1 | 0 | 1.0 | 0 | 80 |
| VISUAL_CONTEXT_FULL_FEAS | __AGGREGATE__ | 1 | 1 | 0 | 1.0 | 0 | 80 |
| VISUAL_CONTEXT_JOINT | __AGGREGATE__ | 0.991478 | 1 | 0 | 1.0 | 0 | 80 |

# SELECTED BACKEND

VISUAL_INTERCEPT_RESIDUAL

# TASK0 GT CONTINUOUS GATE

FAIL

# TASK0 PROBE

NOT REACHED: the task0 GT continuous gate failed, so no Probe evaluation was run.

# FINAL CLASSIFICATION

TASK0_CONTEXT_MEMORIZATION_SUPPORTED

# WHAT TASK0 SUPPORTS

Only the task0 held-out-root conclusions that satisfy the frozen comparison and gate rules.

# WHAT TASK0 DOES NOT SUPPORT

- Task0 only.
- Same object/task distribution.
- No cross-object claim.
- No unseen-task claim.
- No full multi-task conclusion until tasks 1/5/6 complete.
