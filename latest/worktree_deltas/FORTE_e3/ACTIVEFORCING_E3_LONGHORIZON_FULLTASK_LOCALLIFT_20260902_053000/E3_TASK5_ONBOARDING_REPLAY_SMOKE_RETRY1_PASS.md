# E3 task5 onboarding replay smoke — retry 1

Verdict: **PASS**.

- Source demo: preregistered project demo ID `1`.
- Output: `/media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_REPLAY5_20260902_105400_RETRY1/smoke_demo1`.
- The simulation/physics/actions/evaluator were unchanged from attempt 1. The only engineering correction was resolving the existing executable through `TABERO_FFMPEG_BIN=/media/volume/newdata/exouser/softvtbench/miniforge3/bin/ffmpeg`.
- HDF5: one successful episode, 187 steps, 13D actions, required `eef_pose`, `gripper_pos`, `gripper_net_force`, and `gripper_marker_motion` present.
- Camera video: agent and wrist streams each contain 187 frames.
- Tactile video: left and right GS-Mini streams each contain 187 frames.
- Strict QA JSON: `E3_TASK5_ONBOARDING_REPLAY_QA.json`, `gate_pass=true`, no errors.
- Protected MASS PID `647777` was not signaled or modified.

Hashes:

- HDF5: `1d3feed9de6ea2fa22cee68e32e0bb1d3676868b269b51dfb407b553430a181b`
- agent RGB: `afbac3a3ba7775973ec5de0311731f447c990171adf44f730a2718605907c925`
- wrist RGB: `db49334aaee186f3e0cc3ecfc2d56026a6018de2e2ee449c0988b8e2b0512aa2`
- left tactile RGB: `9afba36cf91026cf079522d1cbb127f25c8c1c35d8cc9a71160e1a45b3b1df25`
- right tactile RGB: `d05f8e21893192c5aad10afa877c4fe26b6d8194fb5a11bca588696b6b3e620a`

The remaining preregistered IDs `[2, 11, 12, 19]` may now be replayed, one GPU cell at a time with a fresh resource gate and unique output directory. Training remains forbidden until all five pass schema/media QA and are converted together.
