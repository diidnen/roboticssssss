# True Force-Level Interpolation

## Result

| protocol | n_branches | n_roots | probability_MAE | Brier | ECE_10bin | NLL | ranking_spearman_mean | monotonic_context_rate | selected_force_error_MAE_N | utility_regret_mean |
|---|---|---|---|---|---|---|---|---|---|---|
| ALTERNATING_ANCHOR | 288 | 24 | 0.0650 | 0.0316 | 0.0426 | 0.1679 | 0.8667 | 1.0000 | 0.1650 | 0.0253 |
| HELDOUT_ROOT | 720 | 24 | 0.0760 | 0.0378 | 0.0410 | 0.1738 | 0.7732 | 1.0000 | 0.2455 | 0.0736 |
| LEAVE_ONE_FORCE_REGION_OUT | 720 | 24 | 0.0795 | 0.0473 | 0.0341 | 0.2339 | 0.7153 | 0.6111 | 0.2275 | 0.0954 |
| DEV_GT_PHYSICS | 27 | 7 | 0.1586 | 0.0431 | 0.1128 | 0.3405 | 0.8809 | 1.0000 | 0.1944 | 0.0375 |
| DEV_POINT | 27 | 7 | 0.1468 | 0.0557 | 0.1132 | 0.4020 | 0.8809 | 1.0000 | 0.1667 | 0.1417 |
| DEV_POSTERIOR | 27 | 7 | 0.1632 | 0.0581 | 0.1054 | 0.3849 | 0.8809 | 1.0000 | 0.1389 | 0.0250 |

This is a real target holdout, not a finite-prediction flag. Alternating-anchor fits use strata 1/3/5 and evaluate 2/4 on an entirely held-out physical root. Leave-one-force-region-out excludes both the evaluation root and its entire force stratum. Exact native floats are retained; no rounded or nearest-force joins are used. Model family and regularization were selected by TRAIN root-heldout Brier score only. DEV is reported once after the choice. TEST was not loaded.
