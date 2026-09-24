# Force realize (commanded F ≈ actual squeeze)

New directory. Does not overwrite official 18/19, Fig.B, checkpoints, v4 labels,
or `af_dump_grasp_surface_friction_only_20260915`.

Change: remainder uses the **lift/Tabero original squeeze inner loop**
(`ForcePositionAction` EMA 0.2 / ff 0.9 / kp 0.0008 / deadzone 0.25 N).
Commanded F is a force reference, not a gripper actuator cap. Stock effort
limits are kept. Grasp-surface friction isolation is reused.

Coulomb target = grasp retention. Official dump is logged separately.

Seed 200014 frozen. μ = 0.425 / 0.575 / 0.85. Force grid = v4 20 points.
Coulomb target = grasp retention. Official dump success is logged separately
and is not required to be monotonic. Inner loop is the lift original squeeze
subsystem; F is a reference; stock gripper effort limits are kept.
