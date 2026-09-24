# Force-interface safety and resolution audit

Status: `MINIMAL_PREFLIGHT_REQUIRED_BEFORE_BOUNDARY_COLLECTION`

The controller accepts arbitrary floating-point targets and does not clip the force command; each finger receives half the requested total force. The force servo uses a 0.4 N deadband and clips gripper displacement, not force. Prior project evidence executed 0.25 N commands at 4.25/4.50/4.75 N and a separate legacy task tracked a 1.0 N request at 1.196 N.

No inspected source certifies a project-wide safe minimum. Therefore 1.0 N is only an empirical lower guard, not a safety certificate, and the unlaunched 0.5 N assumption in `current4task_low_force_e3.py` is rejected.

Before any object rollout, a minimal interface preflight must verify:

1. Stable, nonsaturated tracking at 1.0 N in the frozen current controller.
2. Measured separation for adjacent 0.25 N commands around each planned bracket region; the separation criterion is preregistered as non-overlapping 95% intervals or a mean difference >= 0.20 N with tracking MAE <= 0.40 N.
3. No force spike above the existing project-wide 8 N action-support ceiling during preflight.

Until that JSON gate exists and passes, every row in `BOUNDARY_NEXT_QUERY_STATE.csv` has `candidate_is_launchable=0`.

Sources:

- `/home/exouser/Tabero/analysis/results/p4_contact_conditioned_probe_20260822_184213/scripts/p4_collect_probe.py` (SHA-256 `a1566334f9f79ad9d8491612f10d086386049295314f6d1fd62512be1867bca9`)
- `/home/exouser/Tabero/analysis/results/dev_fine_force_forensic_20260829_150000/CONTINUOUS_FORCE_CONTROLLER_AUDIT.json` (SHA-256 `32608e3c8c12dcc83b0ca05ff05640c7e5aa33f0856540060145cd62e8236161`)
- `/home/exouser/Tabero/analysis/results/f1r2_tabero_reactive_force_20260818_205953/FINAL_VERDICT.json` (SHA-256 `87a4f3c25effbfa381c9f150cf640064790e280dc53886d0c5072d3353b78813`)
