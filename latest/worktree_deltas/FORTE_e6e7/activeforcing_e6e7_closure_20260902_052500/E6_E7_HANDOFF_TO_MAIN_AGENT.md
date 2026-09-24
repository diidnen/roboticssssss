# E6/E7 Handoff to Main Agent

## Frozen values

- physical belief: three matched identifier members; checkpoints and hashes are in `E6_E7_LOCKED_HANDOFF/SHA256_MANIFEST.csv`.
- Direct backend: frozen FEASIBILITY_ONLY continuous checkpoints copied read-only from `/home/exouser/FORTE/gnp_style_continuous_20260830_125107`.
- rho: `0.90`; selected only from registered TRAIN/DEV candidates.
- query budget: 2; primary rule `DECISION_CONSENSUS`; second-query protocol is frozen but not validated.
- candidate generators: FIXED_GRID, UNIFORM_CONTINUOUS, STRATIFIED_CONTINUOUS, PROPOSAL_GUIDED; matched K `[5, 10, 20]`.
- posterior samples: 3 physical members; fallback: maximum safe force within task bounds.

## Do not change on locked TEST

Do not retune rho, uncertainty threshold, consensus rule, K, proposal distribution, fallback, architecture, or second-query rule using locked TEST outcomes. Do not replace the negative second-query/simulator status with a duplicated-query or snapped-force proxy.

## Scientific boundary

This handoff is `E6_COMPLETE_NEGATIVE` / `E7_COMPLETE_NEGATIVE` for the current DEV closure: E6 has calibrated/disagreement diagnostics but no validated second-query evidence; E7 has offline planners and freeze artifacts but no passed continuous reliability gate or new arbitrary-force simulator validation.
