# Task5 demo2 real-tactile replay result

## Status

`DEMO2_REAL_TACTILE_7DPF_GATE_PASS`

The preregistered single-demo fallback amendment was committed before this outcome was inspected. Demo2 ran under external core-GPU-scheduler ownership and exited naturally; Agent B did not launch or signal it.

- Root: `/media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_REPLAY5_20260902_110000/demo2`.
- Exact frozen source ID/initial-state digest: pass.
- Semantic task success: true.
- HDF5: 180 steps, finite `[180,13]` actions, SHA-256 `e9bd7466baa3c1e2dfccc039ebaa63b7cc7ab48d2ba0e49263c354ac6c222a50`.
- Real force: `[180,1,2,3]`, 195 nonzero scalars, finite.
- Real marker motion: `[180,2,2,99,2]`, 142,560 nonzero scalars, finite.
- Media: agentview, wrist, left tactile, and right tactile each decode exactly 180 frames at 20 fps; RGB is 512×512 and tactile is 320×240.
- Failure records: empty.
- Independent gate errors: none.

The output counts toward the frozen five-demo TRAIN onboarding set. IDs11/12/19 remain unrun and require one-at-a-time explicit GO plus fresh resource/duplicate audit. This result does not authorize training and does not change Utility or Fmax.
