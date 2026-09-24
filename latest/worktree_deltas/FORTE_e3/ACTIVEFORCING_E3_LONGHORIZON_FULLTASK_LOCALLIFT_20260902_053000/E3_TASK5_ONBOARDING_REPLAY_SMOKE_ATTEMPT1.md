# E3 task5 onboarding replay smoke — attempt 1

- Output: `/media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_REPLAY5_20260902_104200/smoke_demo1`
- Process: PID `649969`; protected MASS PID `647777` was not modified.
- Steady resource sample: GPU utilization 85%, free memory 17,973 MiB; the 8 GiB fail-close threshold was never approached.
- Isaac process ended with the known shutdown `exit -6`/weak-reference exception after exporting the episode.

Strict QA result: `FAIL_MEDIA_ONLY`.

- HDF5 exists, SHA-256 `1d3feed9de6ea2fa22cee68e32e0bb1d3676868b269b51dfb407b553430a181b`.
- One successful exported episode, 187 steps.
- Actions are exactly 13D.
- Required real observations are present: `eef_pose`, `gripper_pos`, `gripper_net_force`, `gripper_marker_motion`.
- All four MP4 streams are absent.

Root cause is deterministic and unrelated to physics/model behavior: `replay_utils.create_video_from_images` hard-coded `/usr/bin/ffmpeg`, which does not exist on this host. The available ffmpeg is `/media/volume/newdata/exouser/softvtbench/miniforge3/bin/ffmpeg`. The encoder failure was printed into Python's buffered stdout and lost during the Isaac shutdown abort; the function then removed temporary PNG directories as written.

Attempt 1 is retained as diagnostic and must never be reused or combined with scientific data. A one-line I/O-only fix in the isolated Tabero E3 worktree resolves `TABERO_FFMPEG_BIN`/PATH while leaving simulation, observations, actions, evaluator, and encoding arguments unchanged. One fresh-output retry is authorized; no additional demonstration may be replayed until it passes strict HDF5+media QA.
