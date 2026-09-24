# Table-push ActiveForcing contract

Gate A changes only `model.geom_friction[table_collision][0]`, reading back `[mu,0.005,0.0001]`. The force variable is the summed, world-frame robot-on-plate contact force projected onto a causal direction from the current frozen pi0 action after stable contact. MuJoCo contact force is converted with `frame.T @ wrench[:3]`; it is negated when the plate is `geom1`, because MuJoCo reports the force on `geom2`.

`PushForceController` tracks a Newton target with EMA P+I control and anti-windup through a bounded residual parallel to that direction. It mutates only action x/y; pi0 z, rotation, and gripper dimensions are copied unchanged. The seed field is private, stripped before model transforms, and resets JAX policy RNG exactly once at the first inference of each episode.

Gate A candidate targets are preregistered as 15, 25, 35, and 45 N at mu=1.2 on roots 0--2, paired within root by seed. Outcomes require online frozen pi0; replay is audit-only. Gate B begins only after all Gate A conditions pass.
