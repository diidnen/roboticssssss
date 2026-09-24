# Task5 onboarding ID1 real-tactile smoke result

## Verdict

`HDF5_SCHEMA_AND_SEMANTIC_SUCCESS / FAIL_CLOSED_MEDIA_ENCODING`

The authorized Agent B smoke ran naturally in the exact task5 tactile replay environment and exported a semantically successful, real 13D `7dpf` HDF5. The overall smoke gate is **not passed** because all four required MP4 streams are missing. This HDF5 is evidence only and is not admitted as final onboarding training data.

## Owned run

- PID: `646854` (exited naturally; no signal).
- Output: `/media/volume/newdata/exouser/activeforcing_e3/TASK5_REAL_TACTILE_7DPF_SMOKE_20260902_104800_ID1`.
- Source: exact `libero_10/task5` assembled demo ID1, 187 steps.
- Environment: `Isaac-Libero-Franka-Replay-Camera-Tactile-v0`.
- Recorder: `7dpf`, `num_envs=1`.
- Result HDF5 SHA-256: `1d3feed9de6ea2fa22cee68e32e0bb1d3676868b269b51dfb407b553430a181b`.
- Episode: `success=True`, `actions[187,13]`, all finite.
- `obs/gripper_net_force[187,1,2,3]`: present, finite, 99 nonzero scalars, SHA-256 `c76342d019a600c6861c9ffff3c988e82265391475d301fde80da6938553271c`.
- `obs/gripper_marker_motion[187,2,2,99,2]`: present, finite, nonempty, SHA-256 `a6c32908c33a81158971e40d6f8a940c654edc2ec45ace9f67030b4339f4d7b5`.
- All robot/object initial-state arrays exactly equal source demo1.

The replay printed semantic success and exported one episode. The post-completion Isaac camera destructor raised the repository-known shutdown `SIGABRT` pattern; it occurred after `Finished replaying 1 episode` and did not corrupt the closed HDF5.

## Fail-closed media finding

The project helper hardcoded `/usr/bin/ffmpeg`, which does not exist on this host. Four encode attempts failed, then the historical helper deleted the temporary RGB/tactile frames. Consequently agentview, eye-in-hand, gsmini-left and gsmini-right MP4s are all absent. This violates the frozen acceptance gate even though the HDF5 contract passed.

An independently queued run under `/media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_REPLAY5_20260902_104200/smoke_demo1` completed with the identical HDF5 SHA-256 and the same four missing-media errors. Its QA JSON SHA-256 is `86c45f7b2c323a9f78e30361c83888718f3649206c24e99b726c850f31de8b74`. It is corroborating evidence only and was never signaled or modified.

No IDs 2/11/12/19 were started. No Utility, Fmax, TEST data, physics, action, evaluator, recorder, or π0 parameter changed.
