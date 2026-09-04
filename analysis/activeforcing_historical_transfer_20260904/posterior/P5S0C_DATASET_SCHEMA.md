# P5-S0-C Dataset Schema

Root group: same task, root seed, initial observable reset state, and task instruction with LOW/MID/HIGH hidden-friction variants.

Context: one root/friction variant, one P4-B probe trace, one post-probe simulator state hash, and task-specific force branches restored from that same state.

Model inputs may use task id, requested scalar force, deployable pre-probe static contact/proprio fields, and deployable dynamic probe telemetry. Hidden friction, object private state, future branch telemetry, and labels are excluded from model tensors.
