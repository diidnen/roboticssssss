# Mass existing-asset audit

Status: **COMPLETE**. This audit is isolated to the `activeforcing-mass` worktree and does not alter the running friction experiment.

## Classification

| Asset | Status | Evidence | Interpretation |
|---|---|---|---|
| Fixed/uniform mass parser | READY | `Tabero/.../libero/physics_config.py` | Validated fixed and reset/startup-uniform mass contracts. |
| PhysX mass/inertia wiring | READY | `franka_libero_env_cfg.py` | Mass properties and reset-time `randomize_rigid_body_mass(..., recompute_inertia=True)` are implemented. |
| Mass-friction damage telemetry | READY | `libero/mdp/terminations.py` | Reset-time mass is logged and can derive a threshold; it is not an identifier. |
| Uniform mass profiles | READY | `benchmarks/datasets/libero/config_profiles/*uniform_mass*` | Configuration assets only; no rollout evidence. |
| COM/inertia smoke checks | READY | `p6g0_smoke_task0.../com_validation.csv`, `geometry.csv` | Runtime hooks preserve visual geometry and reset pose in the smoke test. |
| Historical mass-scale sweep | PARTIAL | `gate_s1.../FINAL_VERDICT.json` | 63 grasp+lift trials, but all lifted and the best force stayed constant across mass. Negative substrate, not a positive benchmark. |
| Additional task design | PARTIAL | `TASK_BENCHMARK_RECOVERY.md`, B2-R2 breadth artifacts | Existing 9-task LIBERO-object catalogue plus planned downstream structures; no mass qualification. |
| Mass-sensitive query | NOT_FOUND | — | Must be tested rather than assumed. |
| Mass estimator/checkpoint | NOT_FOUND | — | Must be trained only after query protocol freeze. |
| Mass branches/manifests | NOT_FOUND | — | No formal Hidden-Mass dataset exists. |

## Important recovered facts

The old mass sweep used the same rigid cream-cheese object at mass scales 0.5/1/2, with 3 seeds and 7 commanded forces. All 63 trials lifted. Measured squeeze changed with commanded force, while mass changed it only weakly; object motion and marker motion were not useful for mass ranking. Therefore this line needs a downstream dynamic response that contains `m` through acceleration/load, not a re-run of the old hold-and-lift protocol.

The Tabero substrate already supports mass randomization and recomputed inertia. The existing friction line remains an independent 720-branch archive and is not copied, relabeled, or overwritten.

The machine currently has a separate `FORTE` IsaacLab process using about 7.6 GiB of GPU memory. The mass branch uses `/home/exouser/FORTE_mass` and `/home/exouser/Tabero_mass`, branch `activeforcing-mass`, based on `7f88d01` and `80ab3be` respectively.

## Audit conclusion

**M0 is complete.** The engineering substrate is READY, historical mass evidence is PARTIAL and negative for the old query, while the scientific mass query, estimator, branches, and paper tables are NOT_FOUND. Work proceeds to development-root identifiability.
