# Active friction imagination successor v1

STATUS: ESTIMATOR_TRAINED_RUNTIME_PILOT_PENDING

This namespace is successor-method development. P5-S0-C/P5-S0-D artifacts are preserved.

- contexts: 144; roots: 48; branches: 576
- train/dev/test roots: 24/8/16
- estimator: projection 16 -> GRU 16 -> mu_hat, sigma_mu
- estimator checkpoint: /home/exouser/Tabero/analysis/results/active_friction_imagination_20260828_211106/FRICTION_GRU.pt
- offline decision audit: {'n': 72, 'split_counts': {'DEV': 24, 'TEST': 48}, 'selection_accuracy': 0.9722222222222222, 'mean_force_gap': 0.5069444444444444, 'under_force_rate': 0.027777777777777776}
- runtime Isaac snapshot-branch imagination and real full-task pilot: NOT yet executed by this module

The offline decision audit is not presented as runtime-imagination evidence; it is only a gate diagnostic against the frozen deterministic branch frontiers.
