# Post-grasp force adaptation protocol

## Final status

```text
FINAL_STATUS = POST_GRASP_4N_LIVE_VALIDATION_COMPLETE
TABERO_NATIVE_EXECUTION_PRESERVED = YES
OBJECT_SPECIFIC_FORCE_FEEDBACK = PASS (live object-filtered bilateral telemetry)
DOUBLE_FORCE_CONTROL = NO
HANDOFF_DEFINITION = first VLA semantic grasp-completed/post-grasp/pre-lift/lift-start row with bilateral target-object contact
4N_POST_GRASP_STATIC = FAIL (post-handoff contact instability)
4N_POST_GRASP_NORMAL_VLA_LIFT = NOT_RUN (static failed; protocol stop)
PURE_VERTICAL_DIAGNOSTIC = NOT_RUN
2N_4N_6N_SAME_STATE_FRONTIER = NOT_RUN
ABSOLUTE_NEWTON_SEMANTICS = PARTIAL (tracking not sustained at 4N for this VLA grasp)
PRIMARY_BLOCKER = post-handoff contact instability while reducing native grasp force toward 4N
READY_FOR_POST_GRASP_ACTIVEFORCING_EVAL = NO
READY_FOR_ACTIVEFORCING_VS_FIXEDMAX = NO
```

The scope is now explicitly post-grasp. Frozen VLA owns approach, grasp
acquisition, centering, and the arm trajectory. ActiveForcing owns only the
post-handoff `F_des`; the executor owns only physical force realization on the
target object.

## Code changes

[`analysis/tabero_true_physical_force_hybrid.py`](tabero_true_physical_force_hybrid.py)
now includes:

- semantic `find_post_grasp_handoff()`; no fixed handoff step;
- `HANDOFF_DEFINITION`, `HANDOFF_STEP`, `HANDOFF_CONDITION` evidence;
- bilateral contact as the minimum handoff validity condition;
- center offset, force asymmetry, and normal opposition as diagnostics only;
- the required failure taxonomy:
  `PRE_HANDOFF_VLA_GRASP_FAILURE`,
  `POST_GRASP_FORCE_INSUFFICIENT`,
  `LOW_LEVEL_FORCE_EXECUTION_FAILURE`, and
  `POST_HANDOFF_VLA_MOTION_CONTACT_FAILURE`;
- `same_state_force_branches()` for same VLA snapshot / same hidden physics /
  same arm trajectory, changing only `F_des`.

The gripper controller still uses:

```text
F_meas = 2 * min(F_left_target_normal, F_right_target_normal)
e_F = F_des - F_meas
d_final = d_nominal + force_correction
```

It never controls approach or arm pose. The gate does not require center offset
or asymmetry thresholds; those values are recorded for analysis.

[`source/tac_manip/tac_manip/tasks/manipulation/libero/mdp/true_physical_force_hybrid.py`](../source/tac_manip/tac_manip/tasks/manipulation/libero/mdp/true_physical_force_hybrid.py)
is the Tabero runtime bridge. It reads
`contact_grasp_<target_object>.data.force_matrix_w`, verifies the target
filter/body metadata when available, and preserves nominal arm slots `0:6`.
It writes the hybrid aperture only to slot `6` and clears legacy force slots
`7:13`. Tabero's old squeeze feedback and 1.9x feed-forward are disabled, so
`DOUBLE_FORCE_CONTROL = NO`.

## Legacy comparison protection

The previous
[`activeforcing_vs_fixedmax_physical_20260904.py`](activeforcing_vs_fixedmax_physical_20260904.py)
is now marked as a deprecated pre-handoff diagnostic and is blocked by
default. It must not be used for a force-selector claim. Future Fixed-Max
comparison is allowed only from the same valid post-grasp snapshot and only
after the 4N chain is explained.

## Verification

[`analysis/test_tabero_true_physical_force_hybrid.py`](test_tabero_true_physical_force_hybrid.py)
passes all 11 tests. They cover the physical bilateral measurement, semantic
handoff selection, handoff-first failure classification, same-state branch
fairness, diagnostics-not-gates behavior, force signs and limits, contact-loss
handling, action preservation, feed-forward semantics, and the single-loop
contract. Both the pure controller and Isaac bridge pass Python compilation.

The existing object-filtered telemetry also confirms the negative case that
motivated this correction: one bilateral row had approximately
`F_left=14.346N`, `F_right=4.693N`, `F_meas=9.386N`, asymmetry `0.507`, and
center offset `19.19mm`. It is now a valid *handoff contact* observation with
poor grasp-quality diagnostics, not an automatically discarded episode and
not evidence of a controller failure.

The implementation tests pass (`12/12`) and the corrected live runner was
compiled successfully. The minimal live run used root `7400`, task
`libero_10/task5`, object `black_book_1`, target `desk_caddy_1`, and the
existing native Tabero pre-handoff squeeze (`7.0 N`, not ActiveForcing). The
frozen VLA action-chunk replay uses the native 10-step replan window.

## Post-grasp 4 N live evidence

Output: [`POST_GRASP_4N_RESULT.json`](results/post_grasp_4n_live_validation_20260904_v9/POST_GRASP_4N_RESULT.json),
[`POST_GRASP_4N_TIMESERIES.csv`](results/post_grasp_4n_live_validation_20260904_v9/POST_GRASP_4N_TIMESERIES.csv),
[`POST_GRASP_SNAPSHOT.pt`](results/post_grasp_4n_live_validation_20260904_v9/POST_GRASP_SNAPSHOT.pt).

```text
GPU_SAFE_TO_RUN = YES (run launched without touching existing workers)
ISAAC_RUNTIME = Isaac Sim 5.1 / NVIDIA A100-SXM4-40GB
POLICY_SERVER = existing servers untouched; task5 candidate metadata verified on port 18883
VLA_NATIVE_GRASP = PASS
POST_GRASP_HANDOFF = PASS at step 96
SNAPSHOT_RESTORE_PARITY = PASS
4N_POST_GRASP_STATIC = FAIL
4N_NORMAL_VLA_LIFT = NOT_RUN
4N_PURE_VERTICAL_DIAGNOSTIC = NOT_RUN
4N_STATIC_MAE = 4.1417 N (2 observed samples; not a sustained tracking metric)
4N_LIFT_MAE = NOT_RUN
PRIMARY_FAILURE_CLASS = POST_HANDOFF_CONTACT_INSTABILITY
READY_FOR_2_4_6_SAME_STATE_BRANCHING = NO
READY_FOR_POST_GRASP_ACTIVEFORCING_EVAL = NO
```

| Stage | Bilateral contact | F_des | F_meas | Object outcome |
|---|---:|---:|---:|---|
| handoff | YES | — | 10.86 N | VLA grasp established |
| static | 1/2 samples | 4 N | 10.86 N → 0 N | right target contact lost on force reduction |
| normal lift | NOT RUN | 4 N | — | stopped after static failure |
| pure vertical diagnostic | NOT RUN | 4 N | — | conditional diagnostic not reached |

At handoff the target-object sensor reported `F_left=12.532 N`,
`F_right=5.428 N`, `F_meas=10.856 N`, and asymmetry `0.396`; these are
diagnostics, not handoff gates. Snapshot restore reproduced object position,
EE position, joint state, and bilateral contact. On the first static control
step the executor correctly entered `FORCE_TRACK` and issued the bounded
opening correction (`+0.00006 m`). The next observation had only left target
contact (`F_left=0.639 N`, `F_right=0 N`), so the protocol stopped and
classified the result as `POST_HANDOFF_CONTACT_INSTABILITY`.

This run does **not** establish `4N_POST_GRASP_FORCE_TRACKING_VALID`, does
not prove 4 N insufficient, and does not authorize 2/4/6 same-state
branching. It does establish a valid VLA-created post-grasp handoff and a
reproducible contact-instability blocker for this state.

The existing protected processes (two legacy comparison workers and the
frozen policy servers) were not killed or modified. The deprecated
pre-handoff comparison runner remains guarded.

The next run is strictly a controller/contact-stability diagnosis from the
saved post-grasp snapshot, if authorized; it must not change grasp geometry,
training, selector, Utility, or the formal benchmark.

Prior to this live run, no new Isaac dynamic result was claimed. The prior
blocked wording is retained only in the historical implementation context;
the current status above is authoritative for the live validation.

The intended protocol remains:

```text
Frozen VLA normal approach/grasp
→ semantic post-grasp bilateral handoff snapshot
→ short 4N post-grasp static hold
→ unchanged normal VLA lift
→ pure-vertical diagnostic only if normal lift fails
```
