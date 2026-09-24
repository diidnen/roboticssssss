# Task5 ID1 patched real-tactile retry result

## Verdict

`ID1_REAL_TACTILE_7DPF_SMOKE_PASS`

The externally queued but exact patched retry completed naturally. Agent B treated PID 654230 -> 654254 as protected pending ownership, sent no signal, launched no duplicate, and independently validated the closed output.

The output root is `/media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_REPLAY5_20260902_105400_RETRY1/smoke_demo1`. It contains one semantically successful 187-step episode, finite 13D actions, real nonzero force, finite marker motion, exact source initial state, and four nonempty frame-aligned videos. Each video independently decoded to 187 frames at 20 fps. Full hashes and dimensions are in `TASK5_PI0_ONBOARDING_ID1_RETRY_GATE.json`.

The HDF5 SHA-256 is `1d3feed9de6ea2fa22cee68e32e0bb1d3676868b269b51dfb407b553430a181b`, identical to both prior real-physics ID1 exports. Producer QA reports `gate_pass=true`, `errors=[]`; its SHA-256 is `1ff2851139caadf3bc46f29b3be4697613280ef0f9d132a7a64394965e73af79`.

The post-finish Isaac camera destructor emitted the known shutdown-abort pattern after all artifacts were closed. Acceptance is based on data integrity and media decoding, not process timing or a nominal exit code.

Protected E5 PID 655134 started concurrently after this replay had launched, raising GPU utilization to 99%. Neither process was signaled. No additional replay is permitted until explicit root GO and a fresh resource gate. IDs 2/11/12/19 remain not started.
