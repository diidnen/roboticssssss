# E3 task5 onboarding replay — demo 19

Status: `DEMO19_REAL_TACTILE_7DPF_GATE_PASS`.

Verdict: `ACCEPT_DATA_INTEGRITY; DO_NOT_CLAIM_CLEAN_PROCESS_EXIT`.

Frozen TRAIN demonstration 19 was replayed through the project-native `Isaac-Libero-Franka-Replay-Camera-Tactile-v0` environment with the `7dpf` recorder. The independent gate records `source_demo_id=19`, `task=libero_10/task5`, `split=TRAIN`, `gate_pass=true`, `errors=[]`, and semantic success.

- Source identity: authoritative source HDF5 SHA-256 `98fb9f3fab6b46a0fbb6d63fc686dccfbb31a372c825c0c87da8fe21f99a6219`; source and replay initial-state digest `bd3b30ebc82ffe81ad40b7f8f38ad79f518da3f0129a9d13354b102cef29070f`.
- Episode: one successful group, 165 steps, finite actions `[165,13]`, finite `eef_pose [165,7]`, real force `[165,1,2,3]` with 123 nonzero scalars, and real marker motion `[165,2,2,99,2]` with 130,680 nonzero scalars.
- Media: agent view and wrist view each decode `165/165` frames at 20 fps and 512x512; left and right tactile RGB each decode `165/165` frames at 20 fps and 320x240.
- Failure record: no `failure*.jsonl` exists and the independent gate records `nonempty_failure_records=[]`.
- Producer/independent agreement: producer QA SHA-256 `83f5e62744d57d09520ea55f032102c85d783bfbd1940db6c121f17a1a37319f` exactly matches the independent gate. Both records agree on source ID, success, HDF5 hash, `[165,13]` actions, force shape/nonzero count, marker shape, and all four 165-frame media counts.
- File hashes: independent gate `7539e974592177c0861408fa14d86edee3e854de2ca74d67bd1852d797ad241d`; HDF5 `ca36d95a379be437843424ef7c8a2d5d9638d6733e1761076fb1fa9f25d19b85`; agent view `30bf4dcf464f5446f1b65b82f227e386ba1881c1478d67afc72586b504503e3d`; wrist view `4bb734e3ade8aec0e45ab213317a8efd47298e883c613b877543be1c553738e9`; left tactile `22f2dce8f65b8f459bfc368e3ac049c8164de05c532955ee5aee16844efc24ca`; right tactile `f3d8a948fbeacd4cbe303ed374b5af31fbf21252320794fe9965432cfa75f8ee`.

The replay log ends with an Isaac weak-reference teardown exception after complete HDF5 and media outputs were written. Acceptance is based on independent HDF5 traversal and full media decoding; it does not claim a clean process exit.

The exact replay command comes from `launch_task5_onboarding_remaining_single.sh` (SHA-256 `b9c5d46d318d7df71056764156611e53683ad7579f572acbd4c4eea35e3b3142`): project-native replay environment, `libero_10/task5`, `--demo_id 19`, `--num_envs 1`, four camera/tactile streams, and `--recorder_type 7dpf`. Producer QA comes from `qa_task5_onboarding_replay.py` (SHA-256 `51ed756277b87ca7f4d10af785ff6941cd4f5bc8f807195a4a26f7d957000d74`). The independent gate command interface comes from `validate_task5_onboarding_single_replay.py` (SHA-256 `3d8f199d2d8fa7287c5da819d6b8f7e1ee87010cf9b7d3f86993715159278b92`):

```bash
cd /home/exouser/FORTE_e3
/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python ACTIVEFORCING_E3_LONGHORIZON_FULLTASK_LOCALLIFT_20260902_053000/validate_task5_onboarding_single_replay.py \
  /media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_REPLAY5_20260902_110000/demo19 \
  19 \
  ACTIVEFORCING_E3_LONGHORIZON_FULLTASK_LOCALLIFT_20260902_053000/TASK5_PI0_ONBOARDING_DEMO19_GATE.json
```

This acceptance audit was CPU/read-only. It did not rerun replay or rewrite the gate. No TEST root, ActiveForcing outcome, Utility/Fmax target, synthetic tactile, or force-selector supervision was used. The authoritative Utility is unchanged.
