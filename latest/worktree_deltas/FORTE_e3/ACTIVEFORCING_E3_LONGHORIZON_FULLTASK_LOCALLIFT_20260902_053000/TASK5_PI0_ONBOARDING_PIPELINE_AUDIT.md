# Task5 few-demo π0 onboarding pipeline audit

## Verdict

`5DEMO_MANIFEST_COMPLETE / TRAINING_BLOCKED_PENDING_REAL_TACTILE_7DPF_REPLAY_GATE`

The exact task5 TRAIN demonstrations exist and the deterministic five-demo manifest is complete. They cannot yet be fed to the authoritative frozen tactile π0 checkpoint: the present successful replay data is 8D and contains only `actions`, `eef_pose`, `gripper_pos`, and two RGB videos. It has no real force history, marker motion, or tactile video. No π0 weights were changed.

## Evidence and selection

The source is `NathanWu7/Isaaclab_Libero` revision `fe6263d3ff1f600f123522e298d6d755bc827d33`. The assembled task5 HDF5 SHA-256 is `98fb9f3fab6b46a0fbb6d63fc686dccfbb31a372c825c0c87da8fe21f99a6219`; it contains 50 Isaac-native demos and exact `black_book_1`, `desk_caddy_1`, and `white_yellow_mug_1` initial-state trees. The immutable successful-replay file SHA-256 is `1823b4172400a6845262e24b9bda4799122086e0215a70bc39fa64e1bb8cf04f`; 18 demos are marked successful.

The fixed rule selects the first five successful IDs in ascending order: `1, 2, 11, 12, 19`. This was independent of ActiveForcing outcomes, force efficiency, DEV/TEST results, or any onboarding result. The set has 901 steps. Each success flag, action length, action digest, assembled initial-state tree digest, and both RGB videos were checked individually. Both videos decode fully at 256×256, 20 fps, and exactly match the action length. See `TASK5_PI0_ONBOARDING_5DEMO_MANIFEST.csv`.

## Project-validated components

The repository documents and implements the real-data path:

`8D Isaac assembled trajectory -> scripts/tools/replay_demos_with_camera.py -> Isaac-Libero-Franka-Replay-Camera-Tactile-v0 -> recorder_type=7dpf -> strict benchmarks/common/convert_all_libero_to_tabero.py -> tactile-field LeRobot dataset`

- Isolated Tabero checkout: commit `80ab3be09ce884f86cfc2037d3af30bc28061426`.
- Replay script SHA-256: `455b58f1c736d6a84633c6f0df56524bd2f8061c715a68f7b1c81bc5fc90c131`.
- Strict tactile converter SHA-256: `0126b486b059b76f1bf9eaefb0e01450ddbb166199e657e3d829fb12dce42be8`.
- OpenPI checkout: commit `1ed9cf44c05bc63fa3b3dbc0ca83ce9dbc8b7b2e`.
- Training entrypoint SHA-256: `fbd99899927e1db63c5cf25cf588ff8b6ea17ad2efae5b53c06415d9bd2efced`.
- Training config SHA-256: `5f53332dfd6955cd7f968f12a147161f0df7dd01ec292034c702a2a310c9f9c5`.
- Base checkpoint: `pi0_lora_tacfield_tabero/49999`, 9.6 GiB, `_CHECKPOINT_METADATA` SHA-256 `f8519027dab6cf8eb113771fbc04d78042a0fb2b38bf770ce4c07866d2b6e022`.

These components are project-validated for tactile replay, strict conversion, and ordinary Tabero LoRA training. The exact task5 five-demo onboarding experiment has not been run historically and is not claimed as previously validated.

## Why direct training is blocked

`TaberoTacFieldInputs` raises `KeyError` when `tactile_marker_motion` is missing. `LiberoForceOutputs` uses 13 action dimensions. The strict converter rejects every 7D/8D trajectory and requires actual `gripper_net_force` plus marker history. The visual-only converter accepts 8D input, but produces a different no-tactile 7D contract. No repository history or result artifact validates any of the following for checkpoint 49999:

- padding force or tactile fields with zeros;
- masking absent tactile fields;
- fine-tuning a no-tactile model and merging its LoRA into the tactile checkpoint;
- using TEST or force-selection outcomes as nominal-task supervision.

All are prohibited here. In addition, the current `tabero_tasks.json` whitelist has `libero_10: []`; an isolated onboarding-only whitelist containing exactly task5 must be frozen before conversion. This is a transparent engineering inclusion, not a change to model or method semantics.

## Training-side readiness after replay

Once true 13D/tactile replays pass the gate, run the existing strict converter in an isolated LeRobot environment, with an isolated `tabero_tasks.json` containing only `libero_10/task5`. Then compute fresh normalization statistics and define a timestamped `pi0_lora_tacfield_tabero_task5_5demo` config initialized from checkpoint 49999. Never overwrite the base checkpoint or its assets. Training and nominal DEV evaluation remain separate future gates; neither has started.

No Utility/Fmax claim is made. The authoritative Utility hash remains `c5bc4f39861a84b9ba95d55cdb5f8633a8b004d205b3864a02799340653d9b10`.
