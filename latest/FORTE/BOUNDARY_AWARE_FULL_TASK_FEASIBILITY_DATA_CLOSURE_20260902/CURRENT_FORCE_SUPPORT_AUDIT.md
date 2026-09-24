# Current force support audit

Status: `AUTHORITATIVE_720_AUDITED`

Frozen population: `/home/exouser/FORTE/activeforcing_residual_utility_20260901_055605/POOLED_OOF_UTILITY_DATASET.csv` (SHA-256 `ebfd959c8a45053863adfa626511f04c1b75d17c4a00786d61296f4d6fad6a13`).

Collection manifest: `/home/exouser/Tabero/analysis/results/gnp_style_visual_context_prospective_20260831_011000/PROSPECTIVE_TRAIN_RUN_MANIFEST.csv` (SHA-256 `5ade1a84c0407a2711a228ef899eaf2d7bed277ad645ee1631cf07463289d6e1`).

The 720 rows align one-to-one on task, root, context, force, and repeat. All 720 trajectory files are present. Local lift is reconstructed from the original collector's exact `object_z - initial_z >= 0.03 m` rule; direct cumulative lift labels agree on every row where both exist.

| Task | Branches | Roots | Contexts | Force support (N) | Lift + / - | Full + / - | Lift=1, full=0 | Full labels direct/reconstructed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 180 | 6 | 18 | 3.0342–4.9782 | 180/0 | 141/39 | 39 | 180/0 |
| 1 | 180 | 6 | 18 | 4.0018–5.9901 | 169/11 | 125/55 | 44 | 40/140 |
| 5 | 180 | 6 | 18 | 3.0042–4.9643 | 180/0 | 133/47 | 47 | 180/0 |
| 6 | 180 | 6 | 18 | 3.0015–3.9966 | 178/2 | 175/5 | 3 | 180/0 |

## Boundary interpretation

`LEFT_CENSORED` means the lowest tested force already succeeded; the transition lies below observed support and is not estimated. `RIGHT_CENSORED` means even the highest force failed. `STOCHASTIC_OR_NONMONOTONIC` preserves repeat disagreement or force-order reversals rather than inventing a frontier. Only `BRACKETED` rows expose a valid empirical interval `(highest failure, lowest success]`.

## Data-quality caveats

- Task1 has 140/180 frozen terminal-height fallback full-task labels. Its full-task boundary rows are development evidence with a material reconstructed-label caveat.
- Task1 local-lift labels for rows absent from the cumulative branch table are reconstructed from physical trajectories using the original 3 cm rule.
- Legacy `failure_stage=full_task_rollout` is too coarse. This audit only asserts `LOCAL_LIFT` when the 3 cm criterion fails; downstream failures without direct stage telemetry remain `DOWNSTREAM_UNRESOLVED`.
- Requested force is the acquisition coordinate. Measured steady force, peak, tracking MAE, and bias are retained separately and are not substituted into the candidate identity.
- These are observed TRAIN/DEV data, not fresh TEST evidence.
