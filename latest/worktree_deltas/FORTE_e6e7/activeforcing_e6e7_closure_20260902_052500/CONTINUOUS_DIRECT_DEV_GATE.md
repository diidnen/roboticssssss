# Continuous Direct DEV Gate

STATUS: **FAIL_WITH_ROOT_CAUSE**

The gate uses only root-held-out DEV contexts and archived repeated DEV outcomes. No TEST roots were loaded.

Posterior-aware Uniform K=10: probability MAE=0.3456, frontier MAE=0.6728 N, under-force=0.4444, held-out roots=7.

Failed checks: probability_calibration, heldout_root_performance, simulator_dev_rollout.

Root cause boundary: this lane has no new simulator validation for arbitrary continuous selected setpoints, and the archived continuous Direct line already documented reliability/decision failures. Planner tables are offline DEV diagnostics, not a validated real controller.
