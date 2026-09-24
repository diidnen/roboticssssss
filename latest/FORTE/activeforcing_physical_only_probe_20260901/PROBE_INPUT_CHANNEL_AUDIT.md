# PROBE input channel audit

## Exact current Learned Probe input

The authoritative current Probe implementation is `run_probe_conditioned_wm.py` → `load_raw_probe()` → `P5.sequence_dataframe()`. It loads each `P5S0C_PROBE_TELEMETRY/*_probe_timesteps.csv`, constructs the exact columns below from `P5S0C_NORMALIZATION.json`, fits feature normalization on the estimator's training contexts, then feeds a 215-step sequence to `Linear(input_dim,16) → ReLU → GRU(16,16) → friction head`. The current model therefore uses **46 physical/action-derived channels and no visual input**.

There is no RGB tensor, frozen visual embedding, object visual feature, task ID, language, object ID, simulator seed, GT friction, downstream outcome, or object pose in the current Probe model input. `eef_dx/eef_dy/eef_dz` are derived from logged EEF pose by subtracting the first row; they are not absolute object pose. Six contact-frame orientation channels are present in the historical 46D legal tensor but are simulator-geometry-derived, so they are excluded from both main estimators in this deployment-constrained run.

`VisionPhysical-Probe` in this run is a matched diagnostic: it adds the archived 4096D frozen π0 visual embedding to every timestep while retaining the exact same 16D projection, GRU, heteroscedastic heads, optimizer, seeds, and NLL. The visual archive exists for 72 current TRAIN contexts only; the historical 144-sequence Probe population is retained as the complete authoritative audit population but has no matched visual capture for its 72 non-TRAIN contexts.

## Field-by-field classification

`VISUAL`, `TASK/IDENTITY`, `PHYSICAL_OBSERVATION`, `ACTION`, `PRIVILEGED_SIM_ONLY`, and `DEPLOYABLE_AT_TEST` are reported as separate audit columns. The current 46D model uses no fields in the first or second classes. The six contact-frame orientation fields are marked simulator-only and excluded from the main 40D deployability-constrained model. Deployability here means the current P4-B telemetry interface can emit the signal online during the fixed probe; transfer to a physical robot still requires sensor-parity validation.

| index | exact channel | primary class | VISUAL | TASK/IDENTITY | PRIVILEGED_SIM_ONLY | ACTION | DEPLOYABLE_AT_TEST | basis |
|---:|---|---|---|---|---|---|---|---|
| 0 | `t_s` | 4 ACTION | no | no | no | yes | yes | commanded probe timing/action or deterministic phase encoding |
| 1 | `force_target` | 4 ACTION | no | no | no | yes | yes | commanded probe timing/action or deterministic phase encoding |
| 2 | `measured_squeeze` | 3 PHYSICAL_OBSERVATION | no | no | no | no | yes | live response/proprioceptive/contact telemetry emitted by the probe logger |
| 3 | `target_normal_force` | 4 ACTION | no | no | no | yes | yes | commanded probe timing/action or deterministic phase encoding |
| 4 | `measured_fn` | 3 PHYSICAL_OBSERVATION | no | no | no | no | yes | live response/proprioceptive/contact telemetry emitted by the probe logger |
| 5 | `measured_ft` | 3 PHYSICAL_OBSERVATION | no | no | no | no | yes | live response/proprioceptive/contact telemetry emitted by the probe logger |
| 6 | `ft_over_fn` | 3 PHYSICAL_OBSERVATION | no | no | no | no | yes | live response/proprioceptive/contact telemetry emitted by the probe logger |
| 7 | `left_fx` | 3 PHYSICAL_OBSERVATION | no | no | no | no | yes | live response/proprioceptive/contact telemetry emitted by the probe logger |
| 8 | `left_fy` | 3 PHYSICAL_OBSERVATION | no | no | no | no | yes | live response/proprioceptive/contact telemetry emitted by the probe logger |
| 9 | `left_fz` | 3 PHYSICAL_OBSERVATION | no | no | no | no | yes | live response/proprioceptive/contact telemetry emitted by the probe logger |
| 10 | `right_fx` | 3 PHYSICAL_OBSERVATION | no | no | no | no | yes | live response/proprioceptive/contact telemetry emitted by the probe logger |
| 11 | `right_fy` | 3 PHYSICAL_OBSERVATION | no | no | no | no | yes | live response/proprioceptive/contact telemetry emitted by the probe logger |
| 12 | `right_fz` | 3 PHYSICAL_OBSERVATION | no | no | no | no | yes | live response/proprioceptive/contact telemetry emitted by the probe logger |
| 13 | `force_imbalance` | 3 PHYSICAL_OBSERVATION | no | no | no | no | yes | live response/proprioceptive/contact telemetry emitted by the probe logger |
| 14 | `force_imbalance_ratio` | 3 PHYSICAL_OBSERVATION | no | no | no | no | yes | live response/proprioceptive/contact telemetry emitted by the probe logger |
| 15 | `gripper_opening` | 3 PHYSICAL_OBSERVATION | no | no | no | no | yes | live response/proprioceptive/contact telemetry emitted by the probe logger |
| 16 | `contact_normal_x` | 5 PRIVILEGED_SIM_ONLY | no | no | yes | no | no | contact frame is computed from simulator geometry in the P4-B implementation |
| 17 | `contact_normal_y` | 5 PRIVILEGED_SIM_ONLY | no | no | yes | no | no | contact frame is computed from simulator geometry in the P4-B implementation |
| 18 | `contact_normal_z` | 5 PRIVILEGED_SIM_ONLY | no | no | yes | no | no | contact frame is computed from simulator geometry in the P4-B implementation |
| 19 | `contact_tangent_x` | 5 PRIVILEGED_SIM_ONLY | no | no | yes | no | no | contact frame is computed from simulator geometry in the P4-B implementation |
| 20 | `contact_tangent_y` | 5 PRIVILEGED_SIM_ONLY | no | no | yes | no | no | contact frame is computed from simulator geometry in the P4-B implementation |
| 21 | `contact_tangent_z` | 5 PRIVILEGED_SIM_ONLY | no | no | yes | no | no | contact frame is computed from simulator geometry in the P4-B implementation |
| 22 | `commanded_tangent_increment_mm` | 4 ACTION | no | no | no | yes | yes | commanded probe timing/action or deterministic phase encoding |
| 23 | `accumulated_displacement_mm` | 3 PHYSICAL_OBSERVATION | no | no | no | no | yes | live response/proprioceptive/contact telemetry emitted by the probe logger |
| 24 | `marker_motion` | 3 PHYSICAL_OBSERVATION | no | no | no | no | yes | live response/proprioceptive/contact telemetry emitted by the probe logger |
| 25 | `marker_tangential` | 3 PHYSICAL_OBSERVATION | no | no | no | no | yes | live response/proprioceptive/contact telemetry emitted by the probe logger |
| 26 | `marker_velocity` | 3 PHYSICAL_OBSERVATION | no | no | no | no | yes | live response/proprioceptive/contact telemetry emitted by the probe logger |
| 27 | `marker_loading_unloading` | 3 PHYSICAL_OBSERVATION | no | no | no | no | yes | live response/proprioceptive/contact telemetry emitted by the probe logger |
| 28 | `contact_left` | 3 PHYSICAL_OBSERVATION | no | no | no | no | yes | live response/proprioceptive/contact telemetry emitted by the probe logger |
| 29 | `contact_right` | 3 PHYSICAL_OBSERVATION | no | no | no | no | yes | live response/proprioceptive/contact telemetry emitted by the probe logger |
| 30 | `tactile_ok` | 3 PHYSICAL_OBSERVATION | no | no | no | no | yes | live response/proprioceptive/contact telemetry emitted by the probe logger |
| 31 | `eef_dx` | 3 PHYSICAL_OBSERVATION | no | no | no | no | yes | live response/proprioceptive/contact telemetry emitted by the probe logger |
| 32 | `eef_dy` | 3 PHYSICAL_OBSERVATION | no | no | no | no | yes | live response/proprioceptive/contact telemetry emitted by the probe logger |
| 33 | `eef_dz` | 3 PHYSICAL_OBSERVATION | no | no | no | no | yes | live response/proprioceptive/contact telemetry emitted by the probe logger |
| 34 | `phase=approach` | 4 ACTION | no | no | no | yes | yes | commanded probe timing/action or deterministic phase encoding |
| 35 | `phase=close` | 4 ACTION | no | no | no | yes | yes | commanded probe timing/action or deterministic phase encoding |
| 36 | `phase=descend` | 4 ACTION | no | no | no | yes | yes | commanded probe timing/action or deterministic phase encoding |
| 37 | `phase=hold` | 4 ACTION | no | no | no | yes | yes | commanded probe timing/action or deterministic phase encoding |
| 38 | `phase=probe_back` | 4 ACTION | no | no | no | yes | yes | commanded probe timing/action or deterministic phase encoding |
| 39 | `phase=probe_hold` | 4 ACTION | no | no | no | yes | yes | commanded probe timing/action or deterministic phase encoding |
| 40 | `phase=probe_out` | 4 ACTION | no | no | no | yes | yes | commanded probe timing/action or deterministic phase encoding |
| 41 | `contact_state=bilateral` | 3 PHYSICAL_OBSERVATION | no | no | no | no | yes | live response/proprioceptive/contact telemetry emitted by the probe logger |
| 42 | `contact_state=none` | 3 PHYSICAL_OBSERVATION | no | no | no | no | yes | live response/proprioceptive/contact telemetry emitted by the probe logger |
| 43 | `contact_state=unilateral` | 3 PHYSICAL_OBSERVATION | no | no | no | no | yes | live response/proprioceptive/contact telemetry emitted by the probe logger |
| 44 | `probe_phase_unknown` | 3 PHYSICAL_OBSERVATION | no | no | no | no | yes | live response/proprioceptive/contact telemetry emitted by the probe logger |
| 45 | `contact_state_unknown` | 3 PHYSICAL_OBSERVATION | no | no | no | no | yes | live response/proprioceptive/contact telemetry emitted by the probe logger |

## Explicitly excluded source fields

`friction`, `hidden_friction_analysis_only`, `object_x_priv`, `object_y_priv`, `object_z_priv`, `object_qw_priv`, `object_qx_priv`, `object_qy_priv`, `object_qz_priv`, `future_branch_telemetry`, `measured_branch_force`, `full_task_success_y`
