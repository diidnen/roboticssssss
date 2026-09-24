# ACTIVEFORCING E7 Continuous Planner Repair and Validation

## Current status: OFFLINE_GATE_PASS_ROLLOUT_PENDING

Legacy status is **LEGACY_INVALID/NEGATIVE_DIAGNOSTIC**, not final evidence. The two old generators were identical, force interpolation was not held out, complete-trajectory calibration was contaminated, and no planner-selected exact float was executed.

The repaired CPU evidence contains segmented interface calibration, three genuine root/force holdouts, a monotone TRAIN-selected Direct, distinct candidate generators, and Expected-Utility point/posterior planning. The authoritative Utility hash is `19c93ad7a57ff5823c53e0171cf297424256d1bbac62df79b7736f0c680415e1`. No hard-rho selector is used and sealed TEST was never read.

Direct gate: **PASS**. Exact-float Isaac authorization: **READY_FOR_ISAAC**. The 8-rollout tracking/parity block must pass before the frozen 144-rollout matched POINT/POSTERIOR × K=5/10 matrix in `E7_MATCHED_FULL_DEV_ROLLOUT_MANIFEST.json` can execute. The results CSV is empty by construction until real execution; nearest-grid substitution is forbidden.

This artifact does **not** claim `E7_CONTINUOUS_PLANNING_SCIENTIFICALLY_VALIDATED` and does not assign `E7_SUPPORTED/NEUTRAL/NEGATIVE` before a passed Direct gate plus paired real off-grid execution.
