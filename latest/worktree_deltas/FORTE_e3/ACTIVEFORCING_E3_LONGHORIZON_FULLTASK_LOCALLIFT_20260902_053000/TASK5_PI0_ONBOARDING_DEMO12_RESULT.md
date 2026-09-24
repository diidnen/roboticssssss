# E3 task5 onboarding replay — demo 12

Status: `DEMO12_REAL_TACTILE_7DPF_GATE_PASS`.

Frozen TRAIN demonstration 12 was replayed once through the project-native tactile environment and `7dpf` recorder. It completed the semantic task successfully in 169 steps. Actions are `[169,13]`; the required pose, gripper, real net-force, and real marker-motion observations are present and finite, with nonzero physical signals. The replay initial-state digest exactly matches the frozen source demonstration.

Agent, wrist, left tactile, and right tactile video streams each decode to 169 frames. HDF5 SHA-256: `fc070bb326815774c118e5f753386d268c77f2f886d373a7f08bcae15fa10310`. The independent gate is `TASK5_PI0_ONBOARDING_DEMO12_GATE.json`.

This is benchmark onboarding only: no TEST root, ActiveForcing outcome, Utility/Fmax target, synthetic tactile, or force-selector supervision was used.

## Independent acceptance audit

Verdict: `ACCEPT_DATA_INTEGRITY; DO_NOT_CLAIM_CLEAN_PROCESS_EXIT`.

- Independent gate core: `source_demo_id=12`, `task=libero_10/task5`, `split=TRAIN`, `gate_pass=true`, `status=DEMO12_REAL_TACTILE_7DPF_GATE_PASS`, and `errors=[]`.
- Exact source identity: authoritative source HDF5 SHA-256 `98fb9f3fab6b46a0fbb6d63fc686dccfbb31a372c825c0c87da8fe21f99a6219`; source and replay initial-state digest `468c67959ae8a5d3067938e1cb27641a626fd731b5e6c33f945ada98f8602c6b`.
- Episode: one successful group, 169 steps, finite actions `[169,13]`, finite `eef_pose [169,7]`, real force `[169,1,2,3]` with 39 nonzero scalars, and real marker motion `[169,2,2,99,2]` with 133,848 nonzero scalars.
- Media: agent view and wrist view each decode `169/169` frames at 20 fps and 512x512; left and right tactile RGB each decode `169/169` frames at 20 fps and 320x240.
- Failure record: no `failure*.jsonl` exists and the independent gate records `nonempty_failure_records=[]`.
- Producer/independent agreement: producer QA SHA-256 is `c78be44aea3c8fdfd4f21afa383019c5bbda57b15a987757cfc85f1bcba09ae7`, exactly matching `producer_qa_sha256` in the independent gate. Producer and independent records agree on source ID, success, HDF5 hash, `[169,13]` actions, force shape/nonzero count, marker shape, and all four 169-frame media counts.
- File hashes: independent gate `f7f690470c5e5917e49db2201e1da28bbb16744b1075c95ead46b2f15e5908a8`; HDF5 `fc070bb326815774c118e5f753386d268c77f2f886d373a7f08bcae15fa10310`; agent view `60b3ac0deb2ce82135c8e4e12722338eaa17d43c91034a65b0f57d0ceae45f46`; wrist view `7fd479dd1ee4751b1de16c33689427fbb75e22319b5f84eb30f019d2fcad4f25`; left tactile `b1e5e6473e5e1cb9bb66e8d76c39b5900b99e4ec6271f1076dc6336ef2c73768`; right tactile `9f220ff23696c1877c800bb3ace6157beae0ce1d9bad8a9379becf1fd0caec67`.

The replay log ends with an Isaac weak-reference teardown exception after the complete HDF5 and media outputs were written. Acceptance is based on independent HDF5 traversal and full media decoding; it does not claim a clean process exit.

The exact replay command comes from `launch_task5_onboarding_remaining_single.sh` (SHA-256 `b9c5d46d318d7df71056764156611e53683ad7579f572acbd4c4eea35e3b3142`): project-native `Isaac-Libero-Franka-Replay-Camera-Tactile-v0`, `libero_10/task5`, `--demo_id 12`, `--num_envs 1`, four camera/tactile streams, and `--recorder_type 7dpf`. Producer QA comes from `qa_task5_onboarding_replay.py` (SHA-256 `51ed756277b87ca7f4d10af785ff6941cd4f5bc8f807195a4a26f7d957000d74`). The exact independent gate command interface comes from `validate_task5_onboarding_single_replay.py` (SHA-256 `3d8f199d2d8fa7287c5da819d6b8f7e1ee87010cf9b7d3f86993715159278b92`):

```bash
cd /home/exouser/FORTE_e3
/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python ACTIVEFORCING_E3_LONGHORIZON_FULLTASK_LOCALLIFT_20260902_053000/validate_task5_onboarding_single_replay.py \
  /media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_REPLAY5_20260902_110000/demo12 \
  12 \
  ACTIVEFORCING_E3_LONGHORIZON_FULLTASK_LOCALLIFT_20260902_053000/TASK5_PI0_ONBOARDING_DEMO12_GATE.json
```

This audit was CPU/read-only. It did not rerun replay or rewrite the gate.
