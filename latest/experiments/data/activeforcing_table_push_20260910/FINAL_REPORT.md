# ActiveForcing table-push status: Gate A falsified

Completed development evidence: the official native successful init-0 action trace replayed to native success (159 steps). The audit recorded 334 raw robot/plate contacts across 104 contact steps, three hand-check samples, contact identities, contact-frame wrenches, world-frame forces on the plate, and summed projection on a direction derived solely from frozen pi0 actions. Mean summed projected force was 18.925 N. `PushForceController` passed parity, XY-only mutation, and bounded-residual unit tests.

The initial seeded service failure and a second smoke failure are retained as infrastructure evidence, not outcomes. With the server held in a live terminal session, a real frozen pi0 task-5 inference succeeded and recorded its private seed stripping/reset. Four paired online root-0, `mu=1.2` branches then completed for targets 15/25/35/45 N with the same seed. They all had exact `[1.2,0.005,0.0001]` readback and all native task outcomes were false. Their active contact-phase EMA projected forces were respectively -14.071, -13.538, -7.969, and -19.347 N; every branch saturated the bounded 0.02 residual. The controller therefore did not demonstrate ordered, finite Newton tracking or force-dependent recovery/feasibility.

Gate A is falsified. No belief data, 216-branch collection, models, selector, or E2E evaluation were started.

Artifacts are all contained in this directory. Any resumption requires a new immutable controller/interface protocol rather than silently retuning this failed Gate-A contract.
