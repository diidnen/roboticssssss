# E3 reserve qualification launch readiness

Status: `STATIC_QA_PASS_DEFERRED_BEHIND_TASK5_SUPPORT_EXTENSION`

## Reproducibility and isolation

- FORTE E3 worktree: `/home/exouser/FORTE_e3`, branch `activeforcing-e3-longhorizon`, commit `7f88d0184c1617ed95e67502da96e60be07b3689`.
- Tabero E3 worktree: `/media/volume/newdata/exouser/Tabero_e3lh`, branch `activeforcing-e3-longhorizon`, commit `80ab3be09ce884f86cfc2037d3af30bc28061426`.
- Historical B5 client SHA-256: `3b38317efad4ee0b2d9e1bfc4379198bbe0d2eb146665e170070b3ad82ec0a18`.
- Frozen pi0 server/checkpoint lineage: `pi0_lora_tacfield_tabero/49999` on port 18881; weights and nominal motion are unchanged.
- Mutable results must be written only to a new timestamped directory under `/media/volume/newdata/exouser/activeforcing_e3/`.
- Candidate code/config reads come from the isolated worktrees. Immutable USD/HDF5 reads come from `/media/volume/newdata/exouser/tabero/data/Isaaclab_Libero`.

## Static QA

| Check | Result |
|---|---|
| Reserve order frozen before candidate outcomes | PASS |
| Project config has task2/task8/goal-task3 and conjunctive goals | PASS |
| `moka_pot`, `flat_stove`, `akita_black_bowl`, `wooden_cabinet` USD assets | PASS |
| Generic final evaluator ANDs all goals | PASS |
| Stove `turnon` threshold `(0.5, 2.1)` | PASS |
| Wooden-cabinet broadened-open evaluator lineage recorded | PASS_WITH_LINEAGE_NOTE |
| Historical source-transform seam cardinality | PASS |
| Transformed source compilation | PASS (93,541 bytes) |
| Wrapper/analyzer `py_compile` | PASS |
| Launcher `bash -n` | PASS |
| Analyzer smoke on prior task5 qualification | PASS; exactly reproduced nominal qualification and per-goal status |
| Overwrite guard on candidate output directory | PASS |

## Qualification semantics

If the task5 6/7/8 N same-root support extension fails to produce three-regime support, each reserve candidate gets at most one TRAIN-nominal episode, in the frozen order, at seed 7600, friction 0.6, fixed 8 N, 80 ten-step chunks. Stop after the first authoritative full-task success. The fixed force changes only the two force setpoint slots after frozen-policy inference; the policy's motion actions and low-level controller remain unchanged.

This is a semantic-capability qualification only. It does not define a task Fmax, does not train a model, does not evaluate ActiveForcing, and does not use Utility. The authoritative final runtime selector remains Expected Utility; hard-rho is prohibited.

## Resource decision

Host audits at 08:39, 08:46, 08:48, and 08:50 UTC observed 89–93% GPU utilization with protected MASS and E5 Isaac processes plus the shared pi0 server. The required `<70%` utilization gate was not met, so no E3 Isaac work was launched. A fresh `nvidia-smi`, `ps`, and `pgrep` audit is mandatory immediately before any future launch. When the gate opens, task5 support extension has priority over reserve qualification because the prior Fmax boundary was invalidated by a suite-local ID collision.
