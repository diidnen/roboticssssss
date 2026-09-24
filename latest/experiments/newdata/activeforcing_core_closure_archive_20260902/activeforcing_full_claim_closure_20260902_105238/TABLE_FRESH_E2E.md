# Fresh reset-to-end E2E

This table is an explicit current full-lineage accounting, but not a locked-test or balanced-task claim. It contains 12 contexts and 60 method rows, with overall, per-task, and per-friction-band views. Repeated retries and smoke-only roots are excluded; missing locked TEST rows are not imputed.

| scope | task | friction band | method | n | full-task SR | query reach | query valid | mean selected force (N) | lineage valid |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| overall | ALL | ALL | FROZEN_VLA_DEFAULT | 12 | 0.16666666666666666 | 0.0 | 0.0 | NA | 1.0 |
| overall | ALL | ALL | FIXED_MAX | 12 | 0.0 | 0.0 | 0.0 | 5.083333333333333 | 1.0 |
| overall | ALL | ALL | NO_QUERY_PRIOR | 12 | 0.0 | 0.0 | 0.0 | 3.75 | 1.0 |
| overall | ALL | ALL | ACTIVEFORCING_DIRECT | 12 | 0.16666666666666666 | 1.0 | 0.5833333333333334 | 3.7083333333333335 | 1.0 |
| overall | ALL | ALL | GT_PHYSICS_DIRECT | 12 | 0.08333333333333333 | 1.0 | 0.5833333333333334 | 4.166666666666667 | 1.0 |
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
| task | 5 | ALL | FROZEN_VLA_DEFAULT | 2 | 0.0 | 0.0 | 0.0 | NA | 1.0 |
| task | 5 | ALL | FIXED_MAX | 2 | 0.0 | 0.0 | 0.0 | 5.0 | 1.0 |
| task | 5 | ALL | NO_QUERY_PRIOR | 2 | 0.0 | 0.0 | 0.0 | 3.75 | 1.0 |
| task | 5 | ALL | ACTIVEFORCING_DIRECT | 2 | 0.0 | 1.0 | 1.0 | 3.25 | 1.0 |
| task | 5 | ALL | GT_PHYSICS_DIRECT | 2 | 0.0 | 1.0 | 1.0 | 3.875 | 1.0 |
| task | 6 | ALL | FROZEN_VLA_DEFAULT | 2 | 0.0 | 0.0 | 0.0 | NA | 1.0 |
| task | 6 | ALL | FIXED_MAX | 2 | 0.0 | 0.0 | 0.0 | 4.0 | 1.0 |
| task | 6 | ALL | NO_QUERY_PRIOR | 2 | 0.0 | 0.0 | 0.0 | 3.0 | 1.0 |
| task | 6 | ALL | ACTIVEFORCING_DIRECT | 2 | 0.0 | 1.0 | 0.0 | 3.25 | 1.0 |
| task | 6 | ALL | GT_PHYSICS_DIRECT | 2 | 0.0 | 1.0 | 0.0 | 3.375 | 1.0 |
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
| friction_band | ALL | MID | FROZEN_VLA_DEFAULT | 3 | 0.0 | 0.0 | 0.0 | NA | 1.0 |
| friction_band | ALL | MID | FIXED_MAX | 3 | 0.0 | 0.0 | 0.0 | 5.333333333333333 | 1.0 |
| friction_band | ALL | MID | NO_QUERY_PRIOR | 3 | 0.0 | 0.0 | 0.0 | 3.9166666666666665 | 1.0 |
| friction_band | ALL | MID | ACTIVEFORCING_DIRECT | 3 | 0.3333333333333333 | 1.0 | 0.6666666666666666 | 3.8333333333333335 | 1.0 |
| friction_band | ALL | MID | GT_PHYSICS_DIRECT | 3 | 0.3333333333333333 | 1.0 | 0.6666666666666666 | 4.0 | 1.0 |
