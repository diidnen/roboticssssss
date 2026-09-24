# ActiveForcing table-push V2: redesign after the honest Gate-A falsifier

Resume the user-authorized table-push ActiveForcing study now that the pre-existing No-Query launcher has completed. Work autonomously in `/home/exouser/Tabero`. Do not restart Mass, alter pi0 weights, stop unrelated processes, or overwrite V1. Use a new immutable namespace `/media/volume/data/exouser/activeforcing_table_push_v2_20260910`. Continue streaming concise evidence-first progress.

## Preserve and explain V1

V1 at `/media/volume/data/exouser/activeforcing_table_push_20260910` is a valid negative engineering result and must remain immutable. Its Gate A requested 15/25/35/45 N, but all branches saturated the 0.02 XY residual and produced unordered signed EMA forces, so it correctly stopped. Do not relabel or exclude those outcomes.

A direct forensic read of V1 telemetry already reveals a specific testable bug in the interface design, not permission to assume V2 passes:

- V1 froze direction from the single nominal action at the third contact step, approximately `[-1, 0]` in all four branches.
- Yet over the active contact phase the frozen-pi0 nominal XY mean was approximately `[-0.18, +0.83]`, and the mean measured force-on-plate XY was approximately `[+14, +31] N` in representative branches.
- Thus the controller projected onto and added residual along an early lateral contact-establishment action rather than the ensuing forward push motion. This explains the negative projection and wrong-axis saturation, but it does not yet prove controllable force or task recovery.

Write `V1_FORENSIC_AUDIT.json` with exact calculations over all four V1 receipts/telemetry before any new rollout.

## Fixed scientific/runtime facts

- Official strict frozen checkpoint only: `/media/volume/newdata/exouser/pi0_libero_activeforcing_20260910/pi0_libero`, tree SHA-256 `92b4ac0c5ed929c81677b750fd13b93aa1d30d1fed50fef06a3143dfda9df103`. It is pi0_libero, not pi0.5 and not the Tabero LoRA already running on the GPU.
- Native task: LIBERO Goal task 5, `push the plate to the front of the stove`.
- Hidden factor: only table sliding friction `model.geom_friction[table_collision][0]`; retain/read back `[mu,0.005,0.0001]`.
- Selected action: measured horizontal robot-on-plate push force along a causal frozen-pi0 future-motion direction. It is neither gripper squeeze nor table-normal force.
- MuJoCo contact measurement and geom filtering must be independently audited as in V1. Store raw wrench, frame, sign, world force and summed projection.
- Policy is online frozen pi0 for outcome evidence. Exact replay is audit-only. Paired root/physics/force branches share an episode-level policy seed.
- Use `/media/volume/data`, not the nearly full `/media/volume/newdata`, for all new outputs.

## V2 direction estimator — causal and preregistered

At three consecutive robot/plate contact steps, discard any queued pre-contact action plan and request a fresh frozen-pi0 action chunk from the current observation. This chunk is available at runtime and contains no hidden simulator state. Before applying a force residual:

1. Compute the intended planar direction from a preregistered robust aggregate of the fresh future chunk, e.g. the normalized sum/median direction over the first 10--20 nondegenerate XY actions after rejecting only actions below a fixed small norm.
2. Store the entire raw chunk, selected indices, aggregate vector, direction, and hash.
3. Require a direction-persistence diagnostic: the aggregate must be nondegenerate and later nominal contact-phase actions must have a positive median dot product with the frozen direction. This later diagnostic is analysis only, never used to change execution.
4. Do not use plate pose, goal pose, friction, future outcome, successful replay displacement, or hand-authored world direction at runtime.

First validate this estimator offline on the successful audit trajectory and on the four V1 traces. Then unit-test it with synthetic chunks, including degenerate and conflicting motions.

## V2 force interface — calibrate before full-task branches

The V1 residual bound was only 0.02 normalized OSC translation (about 1 mm at the documented 0.05 m scale), so it provided no usable authority. Do not simply raise the bound and call it Newton control.

Create an immutable `V2_PROTOCOL.json`, `METHOD_CONTRACT.md`, `STATUS.json`, and append-only `PROGRESS.jsonl` before new simulator actions. Predeclare a two-stage V2 Gate A:

### A2.1 short contact calibration

- Use development root 0 and `mu=1.2` only, with the same episode seed.
- Reach stable contact online with frozen pi0, construct the fresh-chunk direction above, and run short bounded calibration segments rather than complete task outcomes.
- Characterize the measured aligned-force response to a small symmetric residual grid such as `[-0.12,-0.06,0,+0.06,+0.12]` normalized, with ramp-rate limiting and a hard absolute cap no larger than 0.15. Preserve pi0 z/rotation/gripper exactly. Do not call these residual values Newtons.
- Reset to the exact same root and paired policy seed for each segment/episode. Use enough steady contact samples to distinguish signal from impact transients; separately report initial impact and steady phase.
- Check contact retention, finite forces, ordering/correlation, clipping, and action-dimension invariants. A force target is allowed only within the empirically observed overlapping controllable range. If force versus residual is flat, reversed, dominated by impacts, or contact cannot be retained, stop V2 as falsified.
- Maximum five calibration episodes. These are interface calibration, not full-task success trials.

### A2.2 closed-loop tracking and recovery

Only if A2.1 shows a usable response, freeze three Newton targets spanning the demonstrated controllable steady-force range and freeze controller gains/cap/ramp. Then run at most six paired online full-task branches: root 0 first at `mu=1.2`, with the three targets and identical policy seed; use root 1 only if needed for replication.

Pass only if:

- realized aligned contact-phase force is ordered by frozen target and has materially lower tracking error than an open-loop/no-feedback control;
- residual changes only XY, stays within cap/ramp, and contact telemetry is finite;
- at least one same-root comparison shows force-dependent native full-task feasibility/recovery;
- checkpoint, task, seed handling and friction readback are exact.

If tracking works but all outcomes are identical, report that force control was validated but the task-recovery premise was not; do not proceed. If it passes, write an explicit `GATE_A2_PASS.json` and only then begin the existing active-query identifiability Gate B. Do not launch the 216 branches until Gate B also passes.

## Later gates if and only if allowed

If A2 passes, adapt the prior frozen Gate B/C/D plan without weakening it:

- identical micro-push physical transition for Active-Query and Query-Ignored;
- belief runtime inputs only commanded query, EEF proprioception, robot/plate wrench/contact and time; plate pose and simulator friction labels are analysis/training only;
- root-heldout identifiability pilot before collection;
- exactly 216 branches only after passes: roots 5--16 x frictions `[0.6,1.2,2.0]` x six frozen tracked Newton targets; TRAIN 5--10, DEV 11--13, LOCKED_TEST 14--16;
- full-task feasibility predicts native binary success conditioned on future pi0 motion, observable post-query state, friction posterior and candidate force;
- expected-utility selection only among executed targets, with lower-force tie break; no hidden physics/outcome at runtime.

Start now. Prefer a rigorous V2 falsifier today over a large invalid collection. If a resource/infrastructure failure occurs, preserve it separately and retry only after diagnosing it; do not treat it as a task outcome.
