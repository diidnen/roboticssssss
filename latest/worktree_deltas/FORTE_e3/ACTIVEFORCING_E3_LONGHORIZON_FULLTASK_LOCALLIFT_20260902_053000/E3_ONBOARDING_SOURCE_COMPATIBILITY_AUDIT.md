# E3 benchmark-onboarding source compatibility audit

## Decision

The few-demo fallback is now data-ready for a **five-demo Isaac replay gate**, but it is not yet training-ready and no π0 weights have been changed. Two distinct sources were audited and must not be conflated.

## Upstream LIBERO source: reference only

- Repository: `yifengzhu-hf/LIBERO-datasets`
- Frozen revision: `e329580e402fb5f07ae3b1f18475fc3b63783b91`
- File: `libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_demo.hdf5`
- Local SHA-256: `2e0127d0cf73afceacc2c0854e505d5743b6d3d5aaf5572c842f84c6a1c8173b`
- Schema: 50 demonstrations; 7D actions; RGB observations and LIBERO/robosuite state arrays; no `obs/gripper_net_force`, no `obs/gripper_marker_motion`.

This file is valid upstream LIBERO evidence, but it is **not** a direct input to the authoritative tactile π0 pipeline. The project converter strictly requires 13D `7dpf` actions plus real force and marker histories. No zeros or synthetic tactile fields may be inserted.

## Project-native Isaac source: eligible for replay

- Repository: `NathanWu7/Isaaclab_Libero`
- Frozen revision: `fe6263d3ff1f600f123522e298d6d755bc827d33`
- Assembled file: `assembled_hdf5/libero_10_task5_STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_demo.hdf5`
- Assembled SHA-256: `98fb9f3fab6b46a0fbb6d63fc686dccfbb31a372c825c0c87da8fe21f99a6219`
- Project replay file: `replayed_demos/libero_10_task5_STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_demo.hdf5`
- Project replay SHA-256: `1823b4172400a6845262e24b9bda4799122086e0215a70bc39fa64e1bb8cf04f`
- Local root: `/media/volume/newdata/exouser/activeforcing_e3/PROJECT_LIBERO_TASK5_DEMOS_20260902`

The assembled file has 50 episodes with 8D replay actions and Isaac-native `initial_state` trees for the Franka and the exact task objects (`black_book_1`, `desk_caddy_1`, `white_yellow_mug_1`). The fixed project replay contains 18 successful episodes. Its first five successful source IDs are `1, 2, 11, 12, 19`. Selection uses the fixed project replay-success manifest, not ActiveForcing, force efficiency, or E3 branch outcomes.

The project documents and implements the required conversion path:

`Isaac-native assembled_hdf5 (8D) -> replay_demos_with_camera.py -> Isaac-Libero-Franka-Replay-Camera-Tactile-v0 -> recorder_type=7dpf -> 13D actions + real gripper_net_force + gripper_marker_motion + RGB/tactile videos -> convert_all_libero_to_tabero.py -> Tabero tactile-field dataset`

This is compatible in principle with `TaberoTacFieldDataConfig`; compatibility is not yet claimed in practice until one selected demo passes replay and strict converter QA.

## Frozen five-demo gate

- TRAIN source IDs: `1, 2, 11, 12, 19`.
- Replay one demo first into a timestamped, previously nonexistent output directory.
- Required HDF5 fields: 13D `actions`, `obs/eef_pose`, `obs/gripper_pos`, `obs/gripper_net_force`, and `obs/gripper_marker_motion`.
- Required media: both RGB views and both tactile sensor streams.
- The episode must retain exact task5 semantics and pass the authoritative final evaluator.
- If the smoke passes, replay the remaining four IDs serially; never share an output filename.
- Run the strict existing Tabero converter. No force/tactile placeholders and no LeRobot π0 replacement are allowed.
- Only after all five demonstrations pass schema and converter QA may a timestamped onboarding-only LoRA config be created from checkpoint `49999`.

## Resource and scientific boundary

Replay is an Isaac/tactile GPU-heavy job. It remains `RESOURCE_GATED`; no replay or training process was started during the current E5/MASS occupancy. This onboarding path teaches nominal task execution only. It does not define Fmax, modify Utility, use TEST roots, or use ActiveForcing outcomes. The existing Utility hash remains `c5bc4f39861a84b9ba95d55cdb5f8633a8b004d205b3864a02799340653d9b10`.
