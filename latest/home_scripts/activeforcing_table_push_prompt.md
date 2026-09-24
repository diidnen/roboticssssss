# Integrate ActiveForcing into the non-transport table-pushing task

Work autonomously on the remote project, using `/home/exouser/Tabero` and the existing official native LIBERO setup. This is a user-authorized method extension, not merely another robustness sweep. Build a table-push ActiveForcing pipeline structurally analogous to the existing transport pipeline: active physical query -> physical belief -> full-task feasibility branches -> expected-utility push-force selection -> held-out evaluation.

Do not train or alter pi0 weights. Do not resume Mass. Do not modify/delete old artifacts or unrelated jobs. Put every new artifact under `/media/volume/data/exouser/activeforcing_table_push_20260910`; `/media/volume/newdata` is read-only for this work and nearly full. Preserve every failed run.

## Authoritative existing evidence and resources

- Official strict frozen `pi0_libero` checkpoint (not pi0.5, pi0_base, or Tabero LoRA): `/media/volume/newdata/exouser/pi0_libero_activeforcing_20260910/pi0_libero`, tree SHA-256 `92b4ac0c5ed929c81677b750fd13b93aa1d30d1fed50fef06a3143dfda9df103`.
- Native task: LIBERO Goal task ID 5, exact language `push the plate to the front of the stove`.
- Completed frozen table-friction pilot: `/media/volume/data/exouser/pi0_table_friction_20260910`; result 5/5 at table sliding friction 0.6 and 0/5 at both 1.2 and 2.0. Use only as development evidence. Do not overwrite or silently count it as confirmation.
- Working LIBERO invocation pieces:
  - `LIBERO_CONFIG_PATH=/home/exouser/Tabero/analysis/results/pi0_nontransport_20260910/libero_config`
  - `MUJOCO_GL=egl`
  - LIBERO root `/media/volume/newdata/exouser/flowdagger_e960a/bundle/e959c_bootstrap_resolved_r0_20260817T082827Z/flowdagger/flowdagger_pi05/openpi/third_party/libero`
  - simulator Python `/media/volume/newdata/exouser/flowdagger_e960a/e965_sirius_retrospective_correction_efficiency/env/bin/python`
  - client-only site `/home/exouser/Tabero/analysis/results/pi0_nontransport_20260910/client_venv/lib/python3.10/site-packages`
  - OpenPI client `/media/volume/newdata/exouser/tabero/Tabero-VTLA/packages/openpi-client/src`
  - OpenPI source `/media/volume/newdata/exouser/tabero/Tabero-VTLA/src`
  - inference runtime shims `/home/exouser/Tabero/analysis/results/pi0_nontransport_20260910/runtime_shims`
- MuJoCo/robosuite audit already established: OSC_POSE has normalized 6D input [-1,1], translational output range +/-0.05 m, rotational +/-0.5 rad, and kp=150. Robot contact geoms are `gripper0_hand_collision`, `gripper0_finger1_collision`, `gripper0_finger1_pad_collision`, `gripper0_finger2_collision`, `gripper0_finger2_pad_collision`.
- Direct replay of the successful init-0 trajectory proved that `mujoco.mj_contactForce(model._model,data._data,contact_index,wrench)` works. Across robot/plate contact steps 55-158, measured force on the plate averaged about 22.33 N along the observed planar push direction and 37.93 N downward, with 104 contact steps. This is an audit starting point, not a calibrated controller result.
- Existing transport contracts to inspect and adapt, not blindly copy: `/home/exouser/FORTE/ACTIVEFORCING_AUTHORITATIVE_PROTOCOL.json`, `activeforcing_current_probe.py`, `current_contract_physical_belief.py`, `current_fulltask_feasibility_runtime.py`, and `analysis/results/current_runtime_branch_execution_v6_20260905/branch_execution.py`.

## Correct physical semantics

- Hidden physical factor: plate/table sliding friction `mu_table`, varied only through `model.geom_friction[table_collision][0]`; preserve torsional/rolling entries at `[0.005,0.0001]` and read back the full vector before execution.
- Selected action variable: horizontal robot-on-plate push-force setpoint `F_push` in measured Newtons, projected along a causal motion direction derived only from the frozen pi0 action chunk/current task motion. It is not gripper squeeze and not downward table-normal force.
- Use `mujoco.mj_contactForce`; convert each contact-frame wrench to world coordinates and apply the correct sign to obtain force on the plate. Sum only robot-contact-geom <-> `plate_1_*` contacts. Store raw contacts and the projected force calculation.
- A controller target may be called Newtons only if closed-loop measured tracking is validated. A normalized XY residual or displacement command alone must never be labeled Newtons.
- The low-level controller may add only a bounded planar residual parallel to the causal pi0-derived push direction after stable robot/plate contact. Keep pi0 z, rotation and gripper actions unchanged. Log nominal action, residual, executed action, clipping and measured force every step.
- Use a filtered/EMA measured push force and a documented anti-windup P or PI law. Never access hidden friction or future success inside the controller.

## Gate A — force-interface calibration and recovery pilot (mandatory before data collection)

Create `STATUS.json`, `PROGRESS.jsonl`, `METHOD_CONTRACT.md`, `FORCE_INTERFACE_AUDIT.json`, tests, and a development experiment card before policy rollouts.

1. Reconstruct the known successful init-0 trajectory without inference and independently verify contact-force sign, world projection, contact identities, units, and push-direction construction. Include hand-computed checks for at least three contact samples.
2. Build and unit-test `PushForceController`. Validate zero-residual parity when disabled and prove only action dimensions x/y change when enabled.
3. Make policy sampling paired and auditable. The JAX policy defaults to `jax.random.key(0)` and advances by split. Build an audited server wrapper that accepts a private episode-start seed outside the model input, resets `policy._rng=jax.random.key(seed)` exactly once at episode start, strips the private field before transforms, and records every reset/inference. Same root/friction candidates must use the same policy-noise seed. Never claim determinism without an exact replay test.
4. Run the cheapest development calibration on prior-exposed roots only (init indices 0-2). Use friction 1.2 first, then 2.0 only if useful. Choose a small preregistered candidate set around the observed 22 N baseline, such as 15/25/35/45 N, adjusting once only if pre-rollout controller calibration proves the range untrackable. Maximum 12 policy rollouts for Gate A.
5. Gate A passes only if all are true:
   - exact table friction readback and correct official checkpoint/task/root;
   - realized contact-phase push force is ordered by requested target with a meaningful effect and finite telemetry;
   - controller residual is bounded and does not alter z/rotation/gripper dimensions;
   - at least one same-root high-friction comparison shows a force-dependent feasibility change or task recovery;
   - no infrastructure fallback/action replay is used as outcome evidence.
If the controller cannot track or no force-dependent outcome appears, STOP before large collection and report the falsifier. Do not manufacture a belief/model dataset.

## Gate B — active micro-push query identifiability

Only after Gate A passes:

- Insert a fixed, short active micro-push at the first stable robot/plate contact before downstream branching. Pause/clear the queued pi0 actions, hold all non-planar dimensions, apply a small forward/back planar perturbation along the pi0-derived motion direction, then re-infer from the resulting observation.
- The physical query transition must be identical for Active-Query and Query-Ignored controls; only evidence availability differs.
- Belief inputs may contain only deployment-observable quantities: commanded micro-push, robot proprioceptive EEF pose/velocity, measured robot/plate wrench and contact indicators, and time. Direct plate pose/velocity and simulator `mu_table` are analysis/training labels only, never runtime belief features.
- Run an identifiability pilot on development roots 0-4 and frictions `[0.6,1.2,2.0]`, with exact prefix/query replay checks. Train-only labels may use known simulator friction. Require root-heldout ordering/separation above a frozen simple prior/control before claiming a usable belief. Preserve failures/contact loss.

## Gate C — 216-branch collection, only if A and B pass

Freeze exact force targets once from Gate A's validated tracking range, then do not change them. Target collection size is exactly:

`12 independent official init roots x 3 frictions x 6 force targets x 1 execution = 216 branches`.

- Use roots 5-16; friction `[0.6,1.2,2.0]`; six strictly increasing, tracked Newton targets frozen before collection.
- Freeze root split before collection: TRAIN roots 5-10, DEV 11-13, LOCKED_TEST 14-16. Sibling friction/force branches never cross splits.
- For every root/friction, reconstruct the identical online pi0 prefix and identical active query under a paired episode RNG seed. Save one canonical query/evidence record and exact parity checks across its six branches.
- From the post-query decision state, execute the full downstream task with online frozen pi0 plus only the push-force residual. Label `full_task_success_y` using native task success. Record failure stage, contact loss, target/realized force, tracking error, plate displacement (analysis only), pi0 requests, timings, actions and videos.
- Sequential execution unless a tested concurrency plan proves GPU/server and RNG isolation. Resume atomically; four states are `unstarted/active/valid/failed`. Never equate failed infrastructure with task failure.
- Status must report `valid/216`, failed infrastructure, active claims, ETA from empirical throughput, and no stale counts.

## Gate D — models and selector

Only after a structural data audit passes:

- Physical belief: a small 3-seed ensemble over query sequences, predicting positive `mu_table` support/uncertainty. Fit TRAIN only, select on DEV, open LOCKED_TEST once. Compare against no-query prior and query-ignored evidence.
- Full-task feasibility: model `p_theta(y_task=1 | future pi0 motion descriptor, post-query observable state, mu posterior node, F_push)`. Candidate force is an input; target is binary full-task success, not force regression. Use root-heldout splits and report NLL/Brier/AUROC where defined, calibration, monotonicity and boundary behavior.
- Selector: posterior expected utility over only the six executed force targets, with success reward decreasing in force and failure reward -1; lower-force tie break. Never use nearest-force substitution, hidden friction, empirical frontier or future outcome at runtime.
- Offline locked-test comparisons: Active-Query, Query-Ignored/prior, Fixed-Low, Fixed-Max, GT-friction diagnostic, and hindsight oracle diagnostic. Report paired success, selected/realized force, under-force, excess force, query cost and bootstrap uncertainty by root.

## Gate E — fresh E2E only after complete freeze

If Gates A-D pass and resources/time remain, write a separate immutable E2E protocol before touching roots 17-19. Run Active-Query, Query-Ignored and Fixed-Max on the same 3 roots x 3 frictions with paired policy seeds. No tuning on these results. Otherwise stop cleanly at the offline locked test and state what remains.

## Global safety and reporting

- Do not exceed Gate A before its pass record exists. Do not start 216 branches before both Gate A and Gate B pass records exist.
- Prefer completing a scientifically valid smaller gate today over launching an invalid large batch.
- All model/checkpoint/cache/log/video outputs go to `/media/volume/data/exouser/activeforcing_table_push_20260910`.
- Update progress in evidence-first form: current gate, strongest evidence, strongest counterevidence, decision, next falsifiable action, resource cost.
- Keep a concise final report and exact paths. Do not stop at a plan: begin Gate A now and, if it passes, continue automatically through the next allowed gate.
