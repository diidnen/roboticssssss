# Task5 real tactile 7dpf replay readiness

## Status

`PREPARED_NOT_AUTHORIZED`

This is the minimum project-native path that can remove the data-contract blocker. It is prepared only; no replay, conversion, training, or collection was launched.

## Frozen inputs

- TRAIN IDs: `1, 2, 11, 12, 19`.
- Assembled HDF5: `/media/volume/newdata/exouser/activeforcing_e3/PROJECT_LIBERO_TASK5_DEMOS_20260902/assembled_hdf5/libero_10_task5_STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_demo.hdf5`.
- Task: `libero_10/task5`, exact book-to-caddy instruction.
- Isolated code: `/media/volume/newdata/exouser/Tabero_e3lh`, commit `80ab3be09ce884f86cfc2037d3af30bc28061426`.
- Output: a new timestamped directory under `/media/volume/newdata/exouser/activeforcing_e3`; never the downloaded source or shared dataset symlink.

## Exact recorder invocation template

For each frozen source subset, set:

```bash
HDF5_TRAJ_SOURCE_DIR=<directory-containing-only-the-frozen-assembled-subset>
OUTPUT_REPLAYED_DEMOS_DIR=<fresh-output>/replayed_demos
OUTPUT_REPLAYED_VIDEOS_DIR=<fresh-output>/video_datasets
REPLAYED_DEMOS_DIR=<fresh-output>/replayed_demos
USE_TABERO_TASKS=0
```

Then, from the isolated Tabero checkout, use the project-documented command:

```bash
<isaac-python> scripts/tools/replay_demos_with_camera.py \
  --task Isaac-Libero-Franka-Replay-Camera-Tactile-v0 \
  --task_suite libero_10 \
  --task_id 5 \
  --num_envs 1 \
  --video \
  --camera_view_list agentview eye_in_hand \
  --tactile_sensor_list gsmini_left gsmini_right \
  --tactile_output_type tactile_rgb \
  --recorder_type 7dpf \
  --dump_data \
  --headless
```

The subset directory must retain the original task5 filename so suite/task resolution remains authoritative. Subsetting only copies the five already-frozen source groups and root metadata; it must not alter actions, initial states, IDs, or labels.

## Minimum staged execution

The preregistered fail-closed gate requires two GPU jobs:

1. One-demo smoke for ID `1` only. Stop if replay success, HDF5 schema, tactile media, or task identity fails.
2. If and only if the smoke passes, one four-demo batch containing IDs `2, 11, 12, 19`.

Without the smoke requirement all five could run in one Isaac process, but that would bypass the frozen gate. Five single-demo jobs are a simpler fallback if batch collation is operationally risky. Planning estimate: roughly 10–25 minutes total for the two-job path (Isaac startup plus 901 replay steps), or 15–35 minutes for five isolated jobs. These are capacity estimates, not measured results; the smoke wall time must be used to update the second-job estimate.

Before each job: append fresh `nvidia-smi`, `ps`, and `pgrep` evidence; launch only under the E3 GPU policy and with no duplicate; protect π0/E5/MASS/unknown jobs. Run serially with `num_envs=1`.

## Fail-closed acceptance checks

For every selected ID:

1. Exact task suite/id, instruction, demo ID, source action digest, and initial-state tree digest match the frozen manifest.
2. Authoritative environment evaluator reports semantic task success; failed replays are not exported or silently replaced.
3. Recorded `actions` shape is `[T,13]`, finite, with real measured left/right 3D forces in dimensions 7–12; `T` matches the frozen source trajectory.
4. `obs/eef_pose`, `obs/gripper_pos`, `obs/gripper_net_force`, and `obs/gripper_marker_motion` exist, are finite, nonempty, and length-aligned. Force/marker values may not be fabricated or zero-filled as a compatibility device.
5. Agentview, eye-in-hand, gsmini-left tactile, and gsmini-right tactile videos exist, decode fully, and are frame-aligned.
6. The strict converter accepts all five episodes without skip/warning and emits exactly five episodes using isolated `libero_10:[5]` task inclusion.
7. Freeze source, replay, media, converter output, config, and code hashes before normalization or training.

Only after all checks pass is status allowed to advance to `TRAINING_DATA_READY`. Training still requires separate explicit authorization.
