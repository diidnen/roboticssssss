# E3 task5 TRAIN pilot checkpoint

- Canonical immutable result root: `/media/volume/newdata/exouser/activeforcing_e3/TASK5_FORCE_PHYSICS_PILOT_20260902_062448`.
- Six of fifteen preregistered TRAIN cells are complete: all five force cells at `mu=0.2`, and `mu=0.6, F=1 N`.
- Every completed row has the same root-state SHA-256: `a26a8457835421558a23e34b54c04a9f3f6815b6f4cad87a84fcda6e9bb6c816`.
- All six completed cells are `LOCAL_FAILURE`: `pick_success=1`, `lift_success=0`, `full_success=0`.
- The latest cell (`mu=0.6, F=1 N`) completed 500 steps, with mean measured force `0.607960 N` and peak measured force `15.467623 N`.
- This partial result is insufficient to evaluate the preregistered TRAIN gate. Nine cells remain (`mu=0.6, F=2..5 N`; `mu=1.0, F=1..5 N`).
- The external sequential runner stopped fail-closed after PID `515506`; no later E3 cell was launched. At the stop checkpoint, E5 occupied the experiment slot and a protected MASS successor was present. E3 remains paused pending scheduler reassignment.
