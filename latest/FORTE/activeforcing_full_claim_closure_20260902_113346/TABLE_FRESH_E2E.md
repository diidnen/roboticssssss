# Fresh reset-to-end E2E

This table is an explicit current full-lineage accounting, but not a locked-test or balanced-task claim. It contains 12 contexts and 60 method rows, with overall, per-task, and per-friction-band views. Repeated retries and smoke-only roots are excluded; missing locked TEST rows are not imputed.

| scope | task | friction band | method | n | full-task SR | query reach | query valid | mean selected force (N) | lineage valid |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| overall | ALL | ALL | FROZEN_VLA_DEFAULT | 14 | 0.14285714285714285 | 0.0 | 0.0 | NA | 1.0 |
| overall | ALL | ALL | FIXED_MAX | 14 | 0.0 | 0.0 | 0.0 | 5.0 | 1.0 |
| overall | ALL | ALL | NO_QUERY_PRIOR | 14 | 0.0 | 0.0 | 0.0 | 3.6964285714285716 | 1.0 |
| overall | ALL | ALL | ACTIVEFORCING_DIRECT | 14 | 0.14285714285714285 | 1.0 | 0.5714285714285714 | 3.6517857142857144 | 1.0 |
| overall | ALL | ALL | GT_PHYSICS_DIRECT | 14 | 0.07142857142857142 | 1.0 | 0.5714285714285714 | 4.071428571428571 | 1.0 |
| task | 0 | ALL | FROZEN_VLA_DEFAULT | 5 | 0.4 | 0.0 | 0.0 | NA | 1.0 |
| task | 0 | ALL | FIXED_MAX | 5 | 0.0 | 0.0 | 0.0 | 5.0 | 1.0 |
| task | 0 | ALL | NO_QUERY_PRIOR | 5 | 0.0 | 0.0 | 0.0 | 3.75 | 1.0 |
| task | 0 | ALL | ACTIVEFORCING_DIRECT | 5 | 0.4 | 1.0 | 1.0 | 3.85 | 1.0 |
| task | 0 | ALL | GT_PHYSICS_DIRECT | 5 | 0.2 | 1.0 | 1.0 | 4.15 | 1.0 |
| task | 1 | ALL | FROZEN_VLA_DEFAULT | 3 | 0.0 | 0.0 | 0.0 | NA | 1.0 |
| task | 1 | ALL | FIXED_MAX | 3 | 0.0 | 0.0 | 0.0 | 6.0 | 1.0 |
| task | 1 | ALL | NO_QUERY_PRIOR | 3 | 0.0 | 0.0 | 0.0 | 4.25 | 1.0 |
| task | 1 | ALL | ACTIVEFORCING_DIRECT | 3 | 0.0 | 1.0 | 0.0 | 4.083333333333333 | 1.0 |
| task | 1 | ALL | GT_PHYSICS_DIRECT | 3 | 0.0 | 1.0 | 0.0 | 4.916666666666667 | 1.0 |
| task | 5 | ALL | FROZEN_VLA_DEFAULT | 3 | 0.0 | 0.0 | 0.0 | NA | 1.0 |
| task | 5 | ALL | FIXED_MAX | 3 | 0.0 | 0.0 | 0.0 | 5.0 | 1.0 |
| task | 5 | ALL | NO_QUERY_PRIOR | 3 | 0.0 | 0.0 | 0.0 | 3.75 | 1.0 |
| task | 5 | ALL | ACTIVEFORCING_DIRECT | 3 | 0.0 | 1.0 | 1.0 | 3.25 | 1.0 |
| task | 5 | ALL | GT_PHYSICS_DIRECT | 3 | 0.0 | 1.0 | 1.0 | 3.8333333333333335 | 1.0 |
| task | 6 | ALL | FROZEN_VLA_DEFAULT | 3 | 0.0 | 0.0 | 0.0 | NA | 1.0 |
| task | 6 | ALL | FIXED_MAX | 3 | 0.0 | 0.0 | 0.0 | 4.0 | 1.0 |
| task | 6 | ALL | NO_QUERY_PRIOR | 3 | 0.0 | 0.0 | 0.0 | 3.0 | 1.0 |
| task | 6 | ALL | ACTIVEFORCING_DIRECT | 3 | 0.0 | 1.0 | 0.0 | 3.2916666666666665 | 1.0 |
| task | 6 | ALL | GT_PHYSICS_DIRECT | 3 | 0.0 | 1.0 | 0.0 | 3.3333333333333335 | 1.0 |
| friction_band | ALL | HIGH | FROZEN_VLA_DEFAULT | 4 | 0.25 | 0.0 | 0.0 | NA | 1.0 |
| friction_band | ALL | HIGH | FIXED_MAX | 4 | 0.0 | 0.0 | 0.0 | 5.0 | 1.0 |
| friction_band | ALL | HIGH | NO_QUERY_PRIOR | 4 | 0.0 | 0.0 | 0.0 | 3.6875 | 1.0 |
| friction_band | ALL | HIGH | ACTIVEFORCING_DIRECT | 4 | 0.0 | 1.0 | 0.5 | 3.75 | 1.0 |
| friction_band | ALL | HIGH | GT_PHYSICS_DIRECT | 4 | 0.0 | 1.0 | 0.5 | 3.53125 | 1.0 |
| friction_band | ALL | LOW | FROZEN_VLA_DEFAULT | 5 | 0.2 | 0.0 | 0.0 | NA | 1.0 |
| friction_band | ALL | LOW | FIXED_MAX | 5 | 0.0 | 0.0 | 0.0 | 5.0 | 1.0 |
| friction_band | ALL | LOW | NO_QUERY_PRIOR | 5 | 0.0 | 0.0 | 0.0 | 3.7 | 1.0 |
| friction_band | ALL | LOW | ACTIVEFORCING_DIRECT | 5 | 0.2 | 1.0 | 0.6 | 3.6 | 1.0 |
| friction_band | ALL | LOW | GT_PHYSICS_DIRECT | 5 | 0.0 | 1.0 | 0.6 | 4.775 | 1.0 |
| friction_band | ALL | MID | FROZEN_VLA_DEFAULT | 5 | 0.0 | 0.0 | 0.0 | NA | 1.0 |
| friction_band | ALL | MID | FIXED_MAX | 5 | 0.0 | 0.0 | 0.0 | 5.0 | 1.0 |
| friction_band | ALL | MID | NO_QUERY_PRIOR | 5 | 0.0 | 0.0 | 0.0 | 3.7 | 1.0 |
| friction_band | ALL | MID | ACTIVEFORCING_DIRECT | 5 | 0.2 | 1.0 | 0.6 | 3.625 | 1.0 |
| friction_band | ALL | MID | GT_PHYSICS_DIRECT | 5 | 0.2 | 1.0 | 0.6 | 3.8 | 1.0 |
