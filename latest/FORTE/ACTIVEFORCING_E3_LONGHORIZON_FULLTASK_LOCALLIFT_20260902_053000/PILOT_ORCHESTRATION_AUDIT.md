# E3 pilot orchestration audit

At 2026-09-02 06:24:44 UTC, E5 had exited and the GPU gate was open (`util=39%`, `free=25,706 MiB`). Two schedulers then started the same pre-registered TRAIN cell within four seconds:

- PID `508906`, output root `TASK5_FORCE_PHYSICS_PILOT_20260902_062444` (this agent's session).
- PID `508961`, output root `TASK5_FORCE_PHYSICS_PILOT_20260902_062448` (external/root scheduler).

The duplicate was detected before any result was written. This agent immediately stopped its own PID `508906` through its attached session and did not signal the externally owned PID or MASS. A read-only GPU check at 06:26:04 confirmed that PID `508906` had no residue and only PID `508961` remained. No outcome from the aborted lane is admissible.

Canonical pilot root from this point forward:

`/media/volume/newdata/exouser/activeforcing_e3/TASK5_FORCE_PHYSICS_PILOT_20260902_062448`

The abandoned `...062444/TRAIN_root7400_mu0.2_F1N` directory may contain only initialization residue. It must not be merged, treated as an episode, or overwritten. All resume and analysis commands must use the canonical `...062448` root.
