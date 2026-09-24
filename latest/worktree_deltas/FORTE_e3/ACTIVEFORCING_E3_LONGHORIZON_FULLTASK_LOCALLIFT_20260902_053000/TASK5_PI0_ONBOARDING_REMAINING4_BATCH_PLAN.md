# Task5 remaining-four tactile replay batch

## Status

`REMAINING4_BATCH_PREPARED_WAITING_EXPLICIT_GO`

No GPU job has been launched by this preparation. The current gate is closed at 85% GPU utilization with protected π0, MASS PID 647777, and E5 PID 655134. A fresh `nvidia-smi`, `ps`, and `pgrep` audit plus explicit coordinator GO are mandatory before the single batch.

## Frozen batch

- TRAIN-only source IDs: `2, 11, 12, 19`; ID1 is explicitly excluded.
- Steps: `180 + 200 + 169 + 165 = 714`.
- Immutable subset: `/media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_REMAINING4_INPUT_20260902_110100`.
- Subset HDF5 SHA-256: `d4f7323ae5efefbd6e15b3478483bea926f2fa57f7a85468bd9f758190d5e43d`.
- Input manifest SHA-256: `877e5c3588fd8fc23b0a3e8262512de4a5a964a87cd8d67e9dd9a439bbeee6c0`.
- Each copied source-group content digest exactly matches its source. No TEST, force outcome, Utility, or Fmax field entered selection or preparation.

The launcher runs one native `Replay-Camera-Tactile-v0` process, `num_envs=1`, with the identical patched recorder and explicit ffmpeg binary that passed ID1. It refuses reused output, an altered source subset, altered recorder/ffmpeg binaries, a missing coordinator token, or a closed numeric resource gate.

## Identity and schema gate

The native recorder writes successful HDF5 groups sequentially (`demo_0...`) but camera/tactile files retain source IDs. The post-run validator therefore uses the exact `initial_state` tree digest to map every raw group back to one frozen source ID. It requires a bijection over exactly `[2,11,12,19]`; ambiguity, missing success, or extra output fails closed.

Only after all four episodes pass semantic success, finite `[T,13]` actions, real nonzero force and marker fields, exact expected lengths, four fully decoded 20-fps media streams, and an empty failure record may the validator write a separate source-ID-normalized copy. It never mutates the raw output. Every normalized HDF5 group and copied media file is checked by content/file digest.

ID1 remains immutable at `/media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_REPLAY5_20260902_105400_RETRY1/smoke_demo1`; it is not part of this replay batch.
