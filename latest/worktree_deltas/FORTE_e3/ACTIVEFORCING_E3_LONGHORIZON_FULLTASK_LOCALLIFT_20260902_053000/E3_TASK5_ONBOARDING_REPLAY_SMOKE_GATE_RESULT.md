# E3 task5 onboarding tactile replay smoke gate

## Result

`PASS`

The fresh retry at `/media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_REPLAY5_20260902_105400_RETRY1/smoke_demo1` validates the project-native tactile demonstration path for selected source demo `1`.

- One successful exported episode, 187 steps.
- HDF5 SHA-256: `1d3feed9de6ea2fa22cee68e32e0bb1d3676868b269b51dfb407b553430a181b`.
- Actions: `(187, 13)`, the strict `7dpf` contract.
- Required observations present: `eef_pose`, `gripper_pos`, `gripper_net_force`, `gripper_marker_motion`.
- Signals are measured rather than placeholders: action-force absolute max `38.7164` with 102 nonzero values; net-force absolute max `38.7164` with 99 nonzero values; marker current-minus-reference absolute max `24.8158` with 40,411 nonzero values.
- RGB videos: agent view 187 frames; eye-in-hand 187 frames.
- Tactile videos: left 187 frames; right 187 frames.
- Strict QA: `E3_TASK5_ONBOARDING_REPLAY_QA.json`, zero errors.

The process ended after data export with the same known Isaac shutdown `exit -6` weak-reference exception seen in attempt 1. Scientific validity is determined by the complete HDF5/media contract, not that teardown-only code. Attempt 1 remains diagnostic because its MP4 encoding failed before the isolated ffmpeg-path repair.

## Boundary and next step

This is one of five preregistered TRAIN demonstrations. It is not π0 training, ActiveForcing training, an Fmax result, or a Utility result. The remaining frozen source IDs are `2, 11, 12, 19` and must be replayed serially with fresh output filenames. No training may start until all five pass the same strict gate and the existing Tabero tactile-field converter passes.
