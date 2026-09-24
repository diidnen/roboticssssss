# E3 reserve task2 retry readiness

## Status

`READY_OFFLINE_RESOURCE_GATED_WAIT_EXPLICIT_GO`

The first `libero_10/task2` attempt exited 137 under an unexpected concurrent task5 DEV process and produced no episode, step, chunk, or root-hash evidence. It is `RESOURCE_ABORT_NO_SCIENTIFIC_OUTCOME`; its empty directory `/media/volume/newdata/exouser/activeforcing_e3/RESERVE_LONGHORIZON_QUALIFICATION_20260902_095917/t2` is preserved and forbidden from reuse.

## Frozen retry

- Candidate: `libero_10/task2`, TRAIN nominal capability only.
- Root seed: 7600; friction: 0.6; diagnostic setpoint: 8 N.
- Policy: unchanged frozen pi0 checkpoint step49999 on port18881.
- Outcome: authoritative conjunctive environment FullTask success.
- Scope exclusions: no TEST, Fmax inference, Utility evaluation, selector change, task8, or pi0 update.
- Output: create a new timestamped root; never append to the exit137 directory.
- Launch: exactly one foreground job after an immediate `nvidia-smi`/`ps`/`pgrep` audit and explicit root GO.

## Static QA and provenance

- Frozen plan SHA-256: `b2c70aabdd67b5e3ab62e9ad6a6587ea48be87636a5ab62807cd1c3ae15a2f0e`.
- Launcher SHA-256: `fa9ab3fbade9b8f19df7ada1da01aed8b06920a7cd113ef47293724a45f70513`.
- Wrapper SHA-256: `ad5faabc8f0ef9c40d04f1f1fb996ed43a0b0171f4d4c0b7ce07415d0c9e047d`.
- Analyzer SHA-256: `142f8aea81e76df6193cefbc387c8739f00b187c20c793d092b6371739cc3aa1`.
- Python compile and shell syntax checks pass using a temporary bytecode cache.
- Static source audit found no backgrounding, recursive launch, subprocess, or self-trigger mechanism in either E3 launcher. Historical parent shells were already gone, so the source of prior queued invocations cannot be attributed from process state.
- Safe duplicate mitigation: unique timestamped root, exact candidate/process scan immediately before launch, foreground single-cell execution, and no task8 process.

## Current gate

At 10:09-10:10 UTC GPU utilization remained 87-94% with protected pi0 PID33931, MASS PID603571, and E5 PID611219. Retry is prohibited until a fresh audit passes and root sends explicit GO.
