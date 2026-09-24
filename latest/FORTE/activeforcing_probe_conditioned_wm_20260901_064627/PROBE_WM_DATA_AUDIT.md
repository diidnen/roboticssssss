# Probe-WM Data Audit

**PASS for pooled TRAIN-root OOF development; existing estimator is overlap-only diagnostic.**

- Authoritative current population: 24 root families, 72 friction-conditioned contexts, 720 force branches.
- Raw Probe: 72/72 complete files, each 215 timesteps × 46 legal dynamic features. Force/contact, proprioception, probe action/phase and tactile/contact-state signals are present. GT friction and downstream outcome are excluded from the 46 inputs.
- A legal estimator latent exists: the 16D final hidden state of projection16→GRU16, and it is exported for audit. Independently trained fold GRUs do not share an identifiable latent coordinate system, so downstream rich models use a fold-invariant, pre-WM fixed 16D summary: final/mean/std/max of measured normal force, tangential force, tangential/normal ratio, and marker tangential motion. This choice was frozen before WM results.
- `sigma_mu` remains diagnostic only and is not used as a posterior.
- Existing estimator TRAIN-root overlap with current force-selection roots: 24/24 after canonicalizing the different `p5s0c_train_*` versus `pv_train_*` prefixes by task, root index, and simulator seed. It cannot support a held-root claim.
- This run retrains 3 fold-specific estimators × 3 seeds. Every held root's mu_hat and exported h_e come from an estimator trained on the other 16 roots. Probe normalization is fit inside that estimator's TRAIN roots.
- In `POOLED_OOF_PROBE_PREDICTIONS.csv`, the 216 seed-level rows contain `h_00..h_15` and `e_00..e_15`. The 72 `ENSEMBLE_MEAN` rows intentionally leave those columns blank: independently initialized GRU hidden coordinates cannot be averaged, while the ensemble scalar mu_hat is meaningful. These are expected structural NAs, not missing source data.
- Downstream nested training features use two-fold inner OOF Probe estimates; the fixed rich summary has no fitted encoder. Neither a scored root nor the outer validation roots enter their Probe estimator.
- Root IDs, task IDs and friction-conditioned contexts are retained only as grouping/audit keys; root ID is never a model input.

Task1's force outcomes retain the known reconstructed-label caveat (140/180); Probe friction labels and traces themselves are direct simulator records. No untouched TEST was read.
