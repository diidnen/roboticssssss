# E3 task5 TRAIN force-support gate result

## Verdict

`THREE_REGIME_TRAIN_SUPPORT_GATE_PASS`

The corrected TRAIN-only same-root support extension stopped at 7 N after its first FullTask success. The long-horizon `libero_10/task5` candidate now has all three required label regimes under the same frozen pi0, root, and friction. Force 8 N was not run because the preregistered stopping rule fired.

## Evidence

- Task: `libero_10/task5`, book -> back compartment of caddy.
- Split/root/friction: TRAIN / root7400 / mu=0.6.
- Common reset hash: `a26a8457835421558a23e34b54c04a9f3f6815b6f4cad87a84fcda6e9bb6c816`.
- Forces through stop: 1, 2, 3, 4, 5, 6, 7 N; no missing or unexpected cells.
- Query state valid: 7/7.
- Regimes: Local failure 4; Local-success/downstream-failure 2; FullTask success 1.
- F6: pick/lift=1, transport/place/final=0; 500 steps; mean measured 0.594 N, peak 9.051 N.
- F7: pick/lift/transport/place/final=1; 202 steps; mean measured 4.671 N, peak 14.397 N.
- Analyzer errors: 0; gate pass: true.

Machine-readable gate:
`/media/volume/newdata/exouser/activeforcing_e3/TASK5_FORCE_PHYSICS_PILOT_20260902_062448/E3_TASK5_TRAIN_SUPPORT_EXTENSION_GATE.json`
(`sha256=c685c3d16819bc8ad27b3d46b7374db78a30c4613ddc49be8e01e13d76cd1224`).

Cell table:
`/media/volume/newdata/exouser/activeforcing_e3/TASK5_FORCE_PHYSICS_PILOT_20260902_062448/E3_TASK5_TRAIN_SUPPORT_EXTENSION_TABLE.csv`
(`sha256=f7afab85c4eb26c16d62bf71d763ce4ee20dfce8d57f65db9b9b0fd1f9b071ad`).

## Scientific scope

This is a label-support gate, not an Fmax or Utility experiment. `libero_10/task5` still has no authoritative task-specific Fmax. The E1 `libero_object/task5` 5 N value remains inapplicable. The authoritative Utility hash and Expected-Utility selector were not changed, and no TEST data or ActiveForcing outcomes were used.

The frozen pi0 completed the semantic task at 7 N, so few-demo onboarding is `NOT_NEEDED`. The next step is the separately frozen six-cell root-heldout DEV confirmation at roots 7500/7501 and fixed anchors 4/6/7 N. E3 is not complete until that gate and paired FullTask-vs-LocalLift comparison are produced.
