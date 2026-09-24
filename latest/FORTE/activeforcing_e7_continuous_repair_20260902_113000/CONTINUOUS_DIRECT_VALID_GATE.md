# Continuous Direct Valid Gate

## Verdict: PASS

| check | value | pass |
|---|---|---|
| alternating_anchor_probability_mae | 0.0650 | True |
| leave_region_out_brier | 0.0473 | True |
| heldout_root_brier | 0.0378 | True |
| dev_gt_probability_mae | 0.1586 | True |
| dev_point_probability_mae | 0.1468 | True |
| dev_monotonic_context_rate | 1.0000 | True |
| dev_runtime_under_force_rate | 0.0781 | True |
| dev_point_under_force_rate_diagnostic | 0.1719 | DIAGNOSTIC |
| dev_runtime_resolution_aware_under_force_rate_diagnostic | 0.0000 | DIAGNOSTIC |
| interface_loadbearing_coverage | 0.9542 | True |
| interface_tracking_mae_median_N | 0.1922 | True |

The repaired Direct uses correctly normalized task-relative force `(F-Fmin)/(Fmax-Fmin)`, Newton/SLSQP-fitted task heads, physical context, and explicit derivative constraints that make success probability non-decreasing in force across the supported range. Full-task binary labels remain individual branches. Roots and force regions are held out as documented. Regularization, interaction, and failure-class loss weight are selected without DEV outcomes by mean GT-conditional/runtime-belief Brier score; TRAIN root-OOF planner under-force is retained as a separate diagnostic. This prevents a cost-sensitive model from winning merely by depressing probabilities and destroying calibration. The selected failure-class weight is 1.0. Full search evidence is in `CONTINUOUS_DIRECT_TRAIN_MODEL_SELECTION.csv`; final thresholds are frozen in `CONTINUOUS_DIRECT_VALID_GATE.json`.

The prior 3-member identifier omitted task/execution context and produced DEV physical MAE 0.1431, including a catastrophic task1 LOW inversion. The engineering repair uses only deployable P4-B probe summaries, fits a low-capacity task-conditioned ridge/affine calibration, and selects regularization/features by TRAIN leave-one-root-out error. Because the physical intervention was generated from separated LOW/MID/HIGH populations, the continuous probe estimate is then assigned to its nearest TRAIN-only band centroid; the posterior consists of the actual other-root TRAIN friction values in that inferred band. This prevents unsupported gap values from masquerading as physical states. Its DEV physical point MAE is 0.0417. Feature choices, band accuracy, and OOF errors are in `REPAIRED_PHYSICAL_BELIEF_TRAIN_SELECTION.csv`.

Planner authorization is granted. Posterior Expected Utility is the frozen runtime path; point belief is retained as an ablation. The primary under-force metric is the strict comparison against the observed DEV frontier. A frontier-resolution-aware rate remains a sensitivity diagnostic only and cannot authorize rollout.
