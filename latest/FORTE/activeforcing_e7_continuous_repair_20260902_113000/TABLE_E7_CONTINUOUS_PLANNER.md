# E7 Continuous Planner — Offline DEV

| planning | planner | K | n_contexts | mean_selected_force_N | mean_predicted_success | mean_predicted_EU | mean_predicted_utility_regret | under_force_rate | mean_excess_force_N | mean_latency_ms | offgrid_selection_rate | mean_effective_unique_K |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| POINT | FIXED_GRID | 5 | 9 | 4.9167 | 0.9497 | 0.0285 | 0.0100 | 0.1250 | 0.4375 | 0.1695 | 0.0000 | 5.0000 |
| POINT | FIXED_GRID | 10 | 9 | 4.8472 | 0.9405 | 0.0328 | 0.0057 | 0.1250 | 0.3594 | 0.1419 | 0.1111 | 6.3333 |
| POINT | PROPOSAL_GUIDED | 5 | 9 | 4.8598 | 0.9044 | -0.0090 | 0.0475 | 0.1250 | 0.3933 | 0.2535 | 0.8889 | 5.0000 |
| POINT | PROPOSAL_GUIDED | 10 | 9 | 4.8286 | 0.9438 | 0.0383 | 0.0002 | 0.1250 | 0.3281 | 0.2483 | 0.7778 | 9.6667 |
| POINT | STRATIFIED_CONTINUOUS | 5 | 9 | 4.8202 | 0.8784 | -0.0281 | 0.0666 | 0.2500 | 0.3645 | 0.1507 | 1.0000 | 5.0000 |
| POINT | STRATIFIED_CONTINUOUS | 10 | 9 | 4.8229 | 0.9323 | 0.0278 | 0.0107 | 0.2500 | 0.3368 | 0.1483 | 1.0000 | 10.0000 |
| POINT | UNIFORM_CONTINUOUS | 5 | 9 | 4.8049 | 0.8482 | -0.0658 | 0.1043 | 0.1250 | 0.3535 | 0.1329 | 1.0000 | 5.0000 |
| POINT | UNIFORM_CONTINUOUS | 10 | 9 | 4.8542 | 0.9284 | 0.0183 | 0.0202 | 0.2500 | 0.3798 | 0.1282 | 1.0000 | 10.0000 |
| POSTERIOR | FIXED_GRID | 5 | 9 | 5.0278 | 0.9468 | 0.0037 | 0.0078 | 0.0000 | 0.5312 | 0.2556 | 0.0000 | 5.0000 |
| POSTERIOR | FIXED_GRID | 10 | 9 | 4.9028 | 0.9294 | 0.0104 | 0.0011 | 0.0000 | 0.3906 | 0.2327 | 0.1111 | 6.3333 |
| POSTERIOR | PROPOSAL_GUIDED | 5 | 9 | 4.9503 | 0.9354 | 0.0064 | 0.0051 | 0.1250 | 0.4449 | 0.3357 | 0.8889 | 5.0000 |
| POSTERIOR | PROPOSAL_GUIDED | 10 | 9 | 4.9341 | 0.9360 | 0.0107 | 0.0008 | 0.0000 | 0.4258 | 0.3312 | 0.7778 | 9.6667 |
| POSTERIOR | STRATIFIED_CONTINUOUS | 5 | 9 | 4.8860 | 0.8827 | -0.0372 | 0.0486 | 0.1250 | 0.4075 | 0.2384 | 1.0000 | 5.0000 |
| POSTERIOR | STRATIFIED_CONTINUOUS | 10 | 9 | 4.9113 | 0.9182 | -0.0031 | 0.0146 | 0.1250 | 0.4138 | 0.2440 | 1.0000 | 10.0000 |
| POSTERIOR | UNIFORM_CONTINUOUS | 5 | 9 | 4.8585 | 0.8891 | -0.0256 | 0.0371 | 0.1250 | 0.3698 | 0.2227 | 1.0000 | 5.0000 |
| POSTERIOR | UNIFORM_CONTINUOUS | 10 | 9 | 4.8200 | 0.8609 | -0.0496 | 0.0611 | 0.1250 | 0.3337 | 0.2171 | 1.0000 | 10.0000 |

Predicted metrics are diagnostic until the exact-float simulator manifest is executed. Realized SR/force/utility/regret are intentionally not imputed from nearby grid points.
