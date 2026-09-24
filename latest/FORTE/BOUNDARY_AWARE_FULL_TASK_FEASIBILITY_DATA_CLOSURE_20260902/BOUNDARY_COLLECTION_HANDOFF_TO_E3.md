# Boundary collection handoff to E3

Status: `MECHANISM_READY_ON_OBSERVED_CURRENT_TASKS; LONG_HORIZON_STILL_REQUIRED`

Tasks [1, 6] already contain all three physical regimes in observed TRAIN/DEV data. This is mechanism evidence only; it does not replace genuinely long-horizon task-level validation. Task1 carries a reconstructed-label caveat, while task6 is the clean direct-label anchor.

## Unified boundary-seeking rule

For every task/root/physics context, classify branches as local failure, local-success/full-failure, or full success. If the current minimum still lifts, search downward by the calibrated coarse step. After a failure/success bracket exists, bisect to the empirically distinguishable force resolution. Replicate stochastic overlaps before interpolation. Stop at the safe lower guard and report censoring rather than inventing a frontier.

Required branch telemetry: task/root/physics, command/mean/peak force, repeat, initial/query state hashes, grasp/lift height/lift hold, retention/reorientation/placement/full success, drop time/position, failure stage/reason, and tracking error.

Do not change π0, P4-B, friction estimator, Direct architecture, Utility, or controller semantics. Do not spend rollouts in an obviously high-force all-success region.


E3-specific: locate and report both the 3 cm local-lift transition and the later retention/reorientation/placement transition. Reuse already scheduled task5 cells when identical in root, friction, force, repeat, and state hash; never duplicate them merely to satisfy a table shape.
