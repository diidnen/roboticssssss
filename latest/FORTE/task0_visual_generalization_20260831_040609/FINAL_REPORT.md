# STATUS

TASK0_VISUAL_CONTEXT_TRAIN_MEMORIZATION

# SINGLE SCIENTIFIC QUESTION

Do frozen visual features generalize from 18 task0 TRAIN contexts to the two preregistered held-out physical/visual task0 DEV contexts, and do force interaction or Joint supervision add safe independent value?

# WHY TRAIN RESULTS ARE NOT ENOUGH

The task0 TRAIN run contains 18 visual contexts and a centered 17D PCA representation. Because all 10 force/repeat branches within a context share the same x, the effective visual sample size is 18, not 180; context memorization is a primary risk.

# FROZEN TASK0 MODELS

Twelve final epoch-80 checkpoints (four models × three seeds) were hashed before task0 DEV became available. No architecture, PCA, loss, lambda, seed, or checkpoint selection changed after DEV.

# HELD-OUT DEV POPULATION

Two preregistered task0 DEV contexts, 90 physical branches, nine forces from 3.00N to 5.00N, and five repeats per context-force cell.

# VISUAL LEAKAGE AUDIT

PASS. TRAIN roots 0–5 and DEV roots 6–7 are disjoint; PCA and normalization are TRAIN-only; visual x is captured at strict pre-probe step 190; DEV features are projection-only; no DEV outcome was used for feature, calibration, checkpoint, or architecture selection.

# BASE FEAS

Held-out probability MAE 0.182, Brier 0.111, NLL 0.481, frontier MAE 0.400N, under-force 0.500, monotonic-context fraction 1.000.

# VISUAL RESIDUAL

Held-out probability MAE 0.180, Brier 0.108, NLL 0.466, frontier MAE 0.400N, under-force 0.500, monotonic-context fraction 1.000.

# FULL VISUAL FEAS

Held-out probability MAE 0.159, Brier 0.090, NLL 0.386, frontier MAE 0.300N, under-force 1.000, monotonic-context fraction 1.000.

# VISUAL JOINT

Held-out probability MAE 0.211, Brier 0.145, NLL 1.170, frontier MAE 0.450N, under-force 1.000, monotonic-context fraction 1.000.

# TRAIN VS DEV GENERALIZATION

| Model | TRAIN BCE | DEV NLL | Gap | TRAIN rel. gain vs Base | DEV rel. gain vs Base |
|---|---:|---:|---:|---:|---:|
| PROSPECTIVE_BASE_FEAS | 0.144 | 0.482 | 0.338 | 0.000 | 0.000 |
| VISUAL_INTERCEPT_RESIDUAL | 0.136 | 0.471 | 0.335 | 0.057 | 0.023 |
| VISUAL_CONTEXT_FULL_FEAS | 0.095 | 0.401 | 0.306 | 0.342 | 0.169 |
| VISUAL_CONTEXT_JOINT | 0.061 | 1.741 | 1.680 | 0.577 | -2.610 |

# DOES VISUAL CONTEXT GENERALIZE?

NO / EVIDENCE LIMITED

# DOES VISUAL CONTEXT ONLY SHIFT THE CURVE?

NO

# DOES x × F INTERACTION MATTER?

NO

# DOES JOINT ADD INDEPENDENT VALUE?

NO / EVIDENCE LIMITED

# FRONTIER MAE

| model | DEV_contexts | valid_real_frontier_count | valid_real_frontier_coverage | finite_decision_count | finite_decision_coverage | frontier_MAE_N | under_force_count | under_force_rate | mean_under_force_magnitude_N | excess_force_count | mean_excess_force_N | within_0.25N_rate | within_0.50N_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PROSPECTIVE_BASE_FEAS | 2 | 2 | 1.0000 | 2 | 1.0000 | 0.4000 | 1 | 0.5000 | 0.6500 | 1 | 0.1500 | 0.5000 | 0.5000 |
| VISUAL_INTERCEPT_RESIDUAL | 2 | 2 | 1.0000 | 2 | 1.0000 | 0.4000 | 1 | 0.5000 | 0.6500 | 1 | 0.1500 | 0.5000 | 0.5000 |
| VISUAL_CONTEXT_FULL_FEAS | 2 | 2 | 1.0000 | 2 | 1.0000 | 0.3000 | 2 | 1.0000 | 0.3000 | 0 | 0.0000 | 0.5000 | 0.5000 |
| VISUAL_CONTEXT_JOINT | 2 | 2 | 1.0000 | 2 | 1.0000 | 0.4500 | 2 | 1.0000 | 0.4500 | 0 | 0.0000 | 0.5000 | 0.5000 |

# UNDER-FORCE

Safety-first selected backend: VISUAL_INTERCEPT_RESIDUAL. GT gate: FAIL.

# PROBABILITY QUALITY

| model | aggregation | cells | physical_repeats | probability_MAE | Brier | NLL | signed_bias | calibration_intercept | calibration_slope |
|---|---|---|---|---|---|---|---|---|---|
| PROSPECTIVE_BASE_FEAS | ENSEMBLE | 18 | 90 | 0.1824 | 0.1113 | 0.4814 | 0.1556 | -33703763.8519 | 22962470.7127 |
| VISUAL_INTERCEPT_RESIDUAL | ENSEMBLE | 18 | 90 | 0.1802 | 0.1084 | 0.4661 | 0.1509 | -16754754.5996 | 2517126.8488 |
| VISUAL_CONTEXT_FULL_FEAS | ENSEMBLE | 18 | 90 | 0.1590 | 0.0904 | 0.3865 | 0.1577 | -9451015.2567 | 1780238.8421 |
| VISUAL_CONTEXT_JOINT | ENSEMBLE | 18 | 90 | 0.2110 | 0.1446 | 1.1700 | 0.2110 | -87475462.8187 | 8127413.5698 |

# MONOTONICITY

| model | context_id | spearman_force_probability | adjacent_ordering_rate | nonmonotonic_steps | context_monotonic | safe_to_unsafe_reversals | dense_steps |
|---|---|---|---|---|---|---|---|
| PROSPECTIVE_BASE_FEAS | __AGGREGATE__ | 1.0000 | 1.0000 | 0 | 1.0000 | 0 | 80 |
| VISUAL_INTERCEPT_RESIDUAL | __AGGREGATE__ | 1.0000 | 1.0000 | 0 | 1.0000 | 0 | 80 |
| VISUAL_CONTEXT_FULL_FEAS | __AGGREGATE__ | 1.0000 | 1.0000 | 0 | 1.0000 | 0 | 80 |
| VISUAL_CONTEXT_JOINT | __AGGREGATE__ | 0.9915 | 1.0000 | 0 | 1.0000 | 0 | 80 |

# NEAREST-CONTEXT MEMORIZATION RISK

| context_id | nearest_TRAIN_context_id | nearest_TRAIN_root_id | nearest_TRAIN_distance | mean_TRAIN_distance | max_TRAIN_distance |
|---|---|---|---|---|---|
| pv_dev_t0_r06_s5106_low_mu0.288234 | pv_train_t0_r01_s5101_high_mu0.988466 | pv_train_t0_root01_s5101 | 4.2651 | 6.0637 | 7.4554 |
| pv_dev_t0_r07_s5107_low_mu0.294781 | pv_train_t0_r02_s5102_mid_mu0.571550 | pv_train_t0_root02_s5102 | 4.4621 | 6.8144 | 8.1905 |

Only two primary DEV contexts exist, so a distance-benefit correlation is not statistically interpretable. The preregistered supplementary 3-fold TRAIN root-heldout CV was therefore run.

# TASK0 GT GATE

FAIL

# PRIMARY TASK0 CLASSIFICATION

TASK0_VISUAL_CONTEXT_TRAIN_MEMORIZATION

# WHAT TASK0 NOW SUPPORTS

The report supports only the comparisons that meet every frozen held-out probability, frontier, safety, multi-context, and monotonicity rule above.

# WHAT TASK0 DOES NOT SUPPORT

- No multi-task claim while tasks 1/5/6 are incomplete.
- No cross-object or unseen-task claim.
- No final Probe claim.
- No fresh E2E claim.

# NEXT ACTION

Continue the unchanged preregistered tasks 1/5/6. Do not redesign from task0. If task0 visual generalization failed, quantify independent-context insufficiency before considering more force samples.
