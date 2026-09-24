# Table-push ActiveForcing V3: fix the engineering confounds and rerun

The user explicitly authorizes implementing the fix and rerunning. Work autonomously from `/home/exouser/Tabero`. Preserve V1 and V2 exactly as negative engineering evidence. Create a new namespace `/media/volume/data/exouser/activeforcing_table_push_v3_20260910`; never overwrite V1/V2. Do not modify pi0 weights, resume Mass, or stop unrelated GPU jobs. Stream concise evidence-first progress to the current log.

## Corrected diagnosis that V3 must test, not assume

V2 did not establish that table-push force is intrinsically uncontrollable. It confounded the intervention with action saturation, contact/state drift, online replanning, and a one-step telemetry alignment error.

Audited facts from V2:

- Residual grid `[-.12,-.06,0,+.06,+.12]` produced steady means `[39.621,43.068,45.765,45.367,35.683] N`.
- For the first three values the response was ordered upward. `+.06` plateaued. The apparent reversal came mainly from `+.12`.
- At `+.12`, nominal y was about `.915`, 21/35 executed y actions hit the `+1` limit, actual last-window residual averaged only `.0853`, contact counts degraded from 4 to 2/1, and mean downward force weakened to about `-34.2 N` versus `-43.9 N` at zero residual. That cell did not isolate a larger force command.
- The V2 logger's `clipped` flag checked the residual cap but failed to flag final `nominal+residual` action clipping.
- Contact force was sampled before `env.step(executed)` but written on the same row as that action, creating a one-action causal offset.
- Five conditions replanned pi0 every five steps after intervention and compared their final ten steps, by which time plate/EEF/contact states differed.
- Identical seed was not exact pairing: pre-intervention action traces differed by as much as about `.0218`, and seed-reset first-chunk hashes were not all identical.

Write an immutable `V2_ENGINEERING_FORENSIC_AUDIT.json` reproducing these exact quantities directly from raw telemetry. Do not silently change V2's historical decision; state that its physical conclusion is unresolved because the experiment was confounded.

## Freeze V3 protocol before simulator intervention

Create `V3_PROTOCOL.json`, `METHOD_CONTRACT.md`, `STATUS.json`, `PROGRESS.jsonl`, an experiment card, unit tests, and explicit pass/fail rules before new simulator actions. The workflow has three hard gates.

### Gate R1 — exact-state, one-step counterfactual diagnosis

Goal: determine whether the existing OSC action channel has a locally monotonic effect when every other variable is fixed.

1. Obtain one canonical online official frozen `pi0_libero` prefix on LIBERO Goal task 5 (`push the plate to the front of the stove`), root 0, table friction `[1.2,.005,.0001]`, seed 1000. Official checkpoint path and tree SHA-256 remain:
   `/media/volume/newdata/exouser/pi0_libero_activeforcing_20260910/pi0_libero`
   `92b4ac0c5ed929c81677b750fd13b93aa1d30d1fed50fef06a3143dfda9df103`.
2. Save the canonical observation/action chunks and exact executed prefix. Select 2--3 preregistered stable-contact snapshots with adequate positive and negative action headroom. Runtime direction comes only from a fresh pi0 chunk as in V2.
3. Reconstruct each snapshot by replaying the exact same saved canonical prefix. Before admitting a branch, assert equality/tight tolerance for qpos, qvel, controller goal state, contact identities, pre-action wrench, observation hash, nominal action and direction. Record hashes/tolerances. If replay parity fails, stop and implement a full simulator+controller state snapshot/restore; do not continue with merely the same seed.
4. At each identical snapshot, branch over a symmetric residual grid that is guaranteed not to clip after addition to the fixed nominal action. Execute exactly one perturbed action with no pi0 re-inference and no other changed dimension. Explicitly assert `abs(nominal+residual)<1` with margin.
5. Measure robot-on-plate contact force after `env.step(executed)`, not before. Store pre-action and post-action wrench separately. If possible instrument internal MuJoCo control substeps; otherwise clearly call it end-of-control-step response. Compare force increments relative to zero residual at each snapshot.
6. Pass R1 only if the within-identical-state response has the predicted sign and useful monotonic ordering at at least two snapshots, with no clipping/contact loss and independently reconstructed raw contact sums. If it fails, stop with the OSC action channel falsified. Maximum: one canonical online prefix plus replay-only diagnostic branches; replay branches are not task outcomes.

### Gate R2 — genuine Cartesian-force augmentation

Only if R1 diagnoses saturation/state drift as the V2 cause, implement a task-space force augmentation below the normalized action interface. Inspect the exact installed robosuite `OperationalSpaceController` implementation. Do not edit the shared environment/package. Build a namespace-local reversible wrapper/subclass or instance adapter whose disabled mode is bitwise/strictly equivalent to the nominal controller.

The intended structure is a bounded planar Cartesian wrench contribution mapped through the live translational Jacobian, e.g. an additive joint torque `J_pos.T @ (direction * delta_force_N)`, applied after nominal OSC computation and before actuator-limit enforcement. Verify the exact coordinate frame, sign, torque compensation, Jacobian convention and actuator limits from source/runtime rather than blindly copying this example. Never label it Newtons until measured-contact tracking validates it.

Required tests/audits:

- zero-augmentation parity against untouched OSC over a short replay;
- force sign test with small `+/-` additions at an exact cloned contact state;
- joint-torque dimension, finite-value and actuator-clipping checks;
- pi0 action remains unchanged; only the explicitly logged low-level planar wrench augmentation is added;
- force is measured after the causal control step with raw MuJoCo contacts retained;
- no friction, plate/goal pose, or future outcome enters the controller.

Freeze a small safe feedforward grid only after the sign test. Run exact-state short response trials. Pass R2 only if measured aligned force is ordered, contact is retained, and actuator saturation does not dominate. Otherwise stop before closed-loop/task trials.

### Gate A3 — closed-loop Newton tracking and task recovery

Only after R2 passes, freeze a bounded P/PI contact-force controller, three Newton targets inside the empirically demonstrated range, and a separate no-feedback control. Use measured force from the previous completed control step, EMA filtering, ramp limiting and anti-windup. Do not use current-action/pre-step force as if it were post-action feedback.

1. Tracking: paired root-0, mu=1.2 short trials must show ordered realized force and materially lower tracking error than no feedback.
2. Outcome: at most six online full-task branches, root 0 first and root 1 only for replication, with exact canonical prefix parity before branching and identical episode seed. After branching, online frozen pi0 may legitimately replan from changed observations; log this divergence instead of treating it as calibration evidence.
3. Gate A3 passes only if tracking passes and at least one same-root high-friction comparison shows force-dependent native task recovery/feasibility. Preserve failures.

Only after an explicit `GATE_A3_PASS.json` may the prior active-query identifiability Gate B be attempted. The 216-branch dataset remains forbidden until both A3 and B pass. Never use action replay as full-task outcome evidence.

## Resources and environment

- All new artifacts go under `/media/volume/data/exouser/activeforcing_table_push_v3_20260910`; `/media/volume/newdata` is read-only and nearly full.
- Existing unrelated No-Query processes may still occupy GPU memory. Do not stop them. Complete CPU/code/audit work first, then verify GPU headroom immediately before starting the official pi0 server. If insufficient, record `WAITING_FOR_GPU` and wait/retry safely; do not create another false infrastructure failure.
- Use the correct OpenPI environment for the policy server and the established LIBERO simulator environment/client paths from V1/V2.
- Keep GPU service ownership explicit and clean up only V3-owned processes on stop.

Start now and continue autonomously through the cheapest allowed gate. Report current gate, strongest evidence, strongest counterevidence, decision, next falsifiable action and resource cost. A rigorous corrected result is more important than forcing a pass.
