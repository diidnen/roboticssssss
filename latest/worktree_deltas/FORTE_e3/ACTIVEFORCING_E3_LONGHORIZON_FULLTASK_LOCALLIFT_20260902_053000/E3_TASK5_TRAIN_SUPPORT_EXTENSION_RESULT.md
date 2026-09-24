# E3 task5 TRAIN support-extension result

Verdict: `PASS_THREE_REGIME_LONG_HORIZON_TRAIN_SUPPORT`

The frozen TRAIN-only label-support extension stopped at the first FullTask success, 7 N. In the same paired `root7400`, `friction=0.6` context:

| Force | Regime |
|---:|---|
| 1 N | LOCAL_FAILURE |
| 2 N | LOCAL_FAILURE |
| 3 N | LOCAL_FAILURE |
| 4 N | LOCAL_FAILURE |
| 5 N | LOCAL_SUCCESS_DOWNSTREAM_FAILURE |
| 6 N | LOCAL_SUCCESS_DOWNSTREAM_FAILURE |
| 7 N | FULL_TASK_SUCCESS |

All seven branches reached query state and share reset-state SHA-256 `a26a8457835421558a23e34b54c04a9f3f6815b6f4cad87a84fcda6e9bb6c816`. There are no missing forces through the stopping point, no analyzer errors, and no unexpected force cells. The 7 N branch completed in 202 steps.

This is label-support evidence only. It does not define or infer a new-task `Fmax`, does not modify or evaluate Utility, does not evaluate ActiveForcing, and does not use TEST. The existing Utility hash remains `c5bc4f39861a84b9ba95d55cdb5f8633a8b004d205b3864a02799340653d9b10`; Expected Utility remains the selector and hard-rho remains prohibited.

The prior numeric task5 `Fmax=5 N` transfer was invalid because the E1 mapping names `libero_object/task5` (tomato sauce to basket), while this is `libero_10/task5` (book to caddy). The identity-collision audit and original pre-correction hashes are preserved separately.

Next gate: freeze and run DEV-only confirmation on two disjoint roots before classifier fitting or any broader E3 claim.
