# OLD Direct boundary-calibration baseline

Status: `FROZEN_BEFORE_NEW_BOUNDARY_DATA`

All metrics use frozen root-held-out OOF `p_D_OOF_ensemble` and observed TRAIN/DEV outcomes. ECE uses 10 equal-width bins. NLL clips probabilities only for numerical evaluation at 1e-6. Force ranking compares every unanimously failing lower-force cell with every unanimously successful higher-force cell in the same context.

Boundary MAE is deliberately conservative: it is reported only when a context has a clean unanimous bracket. The empirical point is the midpoint between highest unanimous failure and lowest unanimous success; the predicted transition is the lowest candidate with p>=0.5. Censored or overlapping contexts are excluded, not imputed.

| Task | Brier | ECE | NLL | AUROC | Ranking accuracy | Mean p gap | Clean boundaries | Boundary MAE (N) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0515 | 0.0426 | 0.1589 | 0.9810 | 1.0000 | 0.8347 | 8 | 0.1990 |
| 1 | 0.0419 | 0.0543 | 0.1229 | 0.9918 | 1.0000 | 0.5998 | 7 | 0.2802 |
| 5 | 0.0252 | 0.0351 | 0.0812 | 0.9954 | 1.0000 | 0.8171 | 9 | 0.2733 |
| 6 | 0.0339 | 0.0380 | 0.2745 | 0.3377 | NA | NA | 0 | NA |

## Task6 low-force diagnosis

The exact 3.01–3.06 N slice contains 10 branches, 4 full-task failures, and mean predicted success 0.9802. This is the preregistered slice for testing whether boundary augmentation lowers false confidence; no threshold or training change has been made.

These baseline metrics do not establish that new data repairs Direct. They freeze the comparison target for OLD DATA ONLY versus OLD + BOUNDARY DATA.

Task1 retains the 140/180 reconstructed full-task label caveat.
