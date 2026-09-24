# Boundary collection handoff to Joint friction×mass

Status: `PROTOCOL_READY; NO_RUNNING_CAMPAIGN_MUTATION`

## Unified boundary-seeking rule

For every task/root/physics context, classify branches as local failure, local-success/full-failure, or full success. If the current minimum still lifts, search downward by the calibrated coarse step. After a failure/success bracket exists, bisect to the empirically distinguishable force resolution. Replicate stochastic overlaps before interpolation. Stop at the safe lower guard and report censoring rather than inventing a frontier.

Required branch telemetry: task/root/physics, command/mean/peak force, repeat, initial/query state hashes, grasp/lift height/lift hold, retention/reorientation/placement/full success, drop time/position, failure stage/reason, and tracking error.

Do not change π0, P4-B, friction estimator, Direct architecture, Utility, or controller semantics. Do not spend rollouts in an obviously high-force all-success region.


Joint-specific: the context key is task × root × friction × mass. Allocate matched budget to cells whose present outcomes do not bracket either transition; avoid multiplying high-force successes across the full Cartesian grid. Keep physics estimation and acquisition selection distinct in lineage.
