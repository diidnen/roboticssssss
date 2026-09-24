# E3 reserve task8 launch readiness

Status: `STATIC_READY_RESOURCE_GATED_WAIT_EXPLICIT_GO`

Task2 completed as a valid nominal failure with frozen pi0/no grasp. The next candidate in the already-frozen order is `libero_10/task8`; this document prepares that one cell without changing the protocol or authorizing launch.

## Frozen cell

- Exact authoritative instruction: `put both moka pots on the stove`.
- Authoritative config goals: `moka_pot_1 on flat_stove_1` AND `moka_pot_2 on flat_stove_1` AND `flat_stove_1 turnon`.
- LocalLift audit object: `moka_pot_1`; authoritative nominal qualification remains conjunction over all three goals.
- Split/root: TRAIN/root7600; friction 0.6; one episode; max 80 ten-step chunks.
- Frozen pi0: checkpoint step49999 on shared port18881; nominal motion and low-level controller unchanged.
- Fixed 8 N slot: robust nominal diagnostic only, not Fmax, Utility, safety, or runtime-selector evidence.
- New timestamped output root required; no reuse of any task2 directory.

The exact instruction and three goals were re-read from isolated config `/media/volume/newdata/exouser/Tabero_e3lh/benchmarks/datasets/libero/config/libero_10.json`. Required assets were already statically preflighted. No correct local task8 HDF5/demo exists, so the runner will use a deterministic reset rather than substitute another task's demonstration.

## Launch guard

Launch exactly one foreground `t8` cell only after root sends explicit GO and an immediate `nvidia-smi`/`ps`/`pgrep` audit passes. Refuse launch if a task8/reserve process or target directory already exists. Protect all pi0/E5/MASS processes. No task g3, Utility/Fmax experiment, TEST rollout, or pi0 onboarding may start concurrently.

Current gate is closed by protected pi0 PID33931, E5 PID611219, and MASS task9 PID619215.
