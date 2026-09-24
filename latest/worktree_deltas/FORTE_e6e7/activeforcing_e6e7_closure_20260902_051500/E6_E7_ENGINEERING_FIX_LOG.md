# E6/E7 Engineering Fix Log

- Isolated execution to `/home/exouser/FORTE_e6e7` and output namespace under `analysis/results/activeforcing_e6e7_*`.
- Filtered P5-S0-C contexts to TRAIN/DEV before feature loading; original TEST was not parsed.
- Implemented matched 3-member physical belief with root bootstrap resampling and explicit epistemic/aleatoric semantics.
- Used CPU for the small identifier models because the shared A100 already had two running processes; no process was killed.
- Loaded frozen Direct checkpoints read-only and kept one common backend across all planners/disagreement diagnostics.
- Added explicit NOT_EVALUATED artifacts for missing second-query evidence and simulator validation.
