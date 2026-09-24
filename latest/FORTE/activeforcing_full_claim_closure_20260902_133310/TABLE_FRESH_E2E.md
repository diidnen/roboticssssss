# Fresh reset-to-end E2E

This table is an explicit current full-lineage accounting. The root3/LOW rows are marked as a separately frozen locked test; the development allocation remains unbalanced and the locked subset is small. It contains 21 contexts and 102 method rows, with overall, per-task, and per-friction-band views. Repeated retries and smoke-only roots are excluded; missing locked TEST rows are not imputed.

| scope | task | friction band | method | n | full-task SR | SR 95% CI | query reach | query valid | conditional downstream SR | query-qualified SR | mean force (N) | query duration (s) | lineage valid |
|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| overall | ALL | ALL | FROZEN_VLA_DEFAULT | 21 | 0.14285714285714285 | [0.0, 0.36363636363636365] | 0.0 | 0.0 | NA | NA | NA | NA | 1.0 |
| overall | ALL | ALL | FIXED_MAX | 20 | 0.05 | [0.0, 0.1] | 0.0 | 0.0 | NA | NA | 5.0 | NA | 1.0 |
| overall | ALL | ALL | NO_QUERY_PRIOR | 20 | 0.05 | [0.0, 0.1] | 0.0 | 0.0 | NA | NA | 3.7 | NA | 1.0 |
| overall | ALL | ALL | ACTIVEFORCING_DIRECT | 21 | 0.09523809523809523 | [0.0, 0.1515151515151515] | 1.0 | 0.5714285714285714 | 0.09523809523809523 | 0.16666666666666666 | 3.6726190476190474 | 0.8000000000000002 | 1.0 |
| overall | ALL | ALL | GT_PHYSICS_DIRECT | 20 | 0.05 | [0.0, 0.1] | 1.0 | 0.6 | 0.05 | 0.08333333333333333 | 4.25 | 0.8000000000000002 | 1.0 |
| task | 0 | ALL | FROZEN_VLA_DEFAULT | 8 | 0.375 | [0.08333333333333333, 0.8333333333333334] | 0.0 | 0.0 | NA | NA | NA | NA | 1.0 |
| task | 0 | ALL | FIXED_MAX | 8 | 0.125 | [0.0, 0.25] | 0.0 | 0.0 | NA | NA | 5.0 | NA | 1.0 |
| task | 0 | ALL | NO_QUERY_PRIOR | 8 | 0.125 | [0.0, 0.25] | 0.0 | 0.0 | NA | NA | 3.75 | NA | 1.0 |
| task | 0 | ALL | ACTIVEFORCING_DIRECT | 8 | 0.25 | [0.0, 0.3333333333333333] | 1.0 | 1.0 | 0.25 | 0.25 | 3.78125 | 0.7999999999999999 | 1.0 |
| task | 0 | ALL | GT_PHYSICS_DIRECT | 8 | 0.125 | [0.0, 0.25] | 1.0 | 1.0 | 0.125 | 0.125 | 4.28125 | 0.7999999999999999 | 1.0 |
| task | 1 | ALL | FROZEN_VLA_DEFAULT | 5 | 0.0 | [0.0, 0.0] | 0.0 | 0.0 | NA | NA | NA | NA | 1.0 |
| task | 1 | ALL | FIXED_MAX | 4 | 0.0 | [0.0, 0.0] | 0.0 | 0.0 | NA | NA | 6.0 | NA | 1.0 |
| task | 1 | ALL | NO_QUERY_PRIOR | 4 | 0.0 | [0.0, 0.0] | 0.0 | 0.0 | NA | NA | 4.25 | NA | 1.0 |
| task | 1 | ALL | ACTIVEFORCING_DIRECT | 5 | 0.0 | [0.0, 0.0] | 1.0 | 0.0 | 0.0 | NA | 4.15 | 0.8 | 1.0 |
| task | 1 | ALL | GT_PHYSICS_DIRECT | 4 | 0.0 | [0.0, 0.0] | 1.0 | 0.0 | 0.0 | NA | 5.1875 | 0.8 | 1.0 |
| task | 5 | ALL | FROZEN_VLA_DEFAULT | 4 | 0.0 | [0.0, 0.0] | 0.0 | 0.0 | NA | NA | NA | NA | 1.0 |
| task | 5 | ALL | FIXED_MAX | 4 | 0.0 | [0.0, 0.0] | 0.0 | 0.0 | NA | NA | 5.0 | NA | 1.0 |
| task | 5 | ALL | NO_QUERY_PRIOR | 4 | 0.0 | [0.0, 0.0] | 0.0 | 0.0 | NA | NA | 3.75 | NA | 1.0 |
| task | 5 | ALL | ACTIVEFORCING_DIRECT | 4 | 0.0 | [0.0, 0.0] | 1.0 | 1.0 | 0.0 | 0.0 | 3.25 | 0.8 | 1.0 |
| task | 5 | ALL | GT_PHYSICS_DIRECT | 4 | 0.0 | [0.0, 0.0] | 1.0 | 1.0 | 0.0 | 0.0 | 4.125 | 0.8 | 1.0 |
| task | 6 | ALL | FROZEN_VLA_DEFAULT | 4 | 0.0 | [0.0, 0.0] | 0.0 | 0.0 | NA | NA | NA | NA | 1.0 |
| task | 6 | ALL | FIXED_MAX | 4 | 0.0 | [0.0, 0.0] | 0.0 | 0.0 | NA | NA | 4.0 | NA | 1.0 |
| task | 6 | ALL | NO_QUERY_PRIOR | 4 | 0.0 | [0.0, 0.0] | 0.0 | 0.0 | NA | NA | 3.0 | NA | 1.0 |
| task | 6 | ALL | ACTIVEFORCING_DIRECT | 4 | 0.0 | [0.0, 0.0] | 1.0 | 0.0 | 0.0 | NA | 3.28125 | 0.8 | 1.0 |
| task | 6 | ALL | GT_PHYSICS_DIRECT | 4 | 0.0 | [0.0, 0.0] | 1.0 | 0.0 | 0.0 | NA | 3.375 | 0.8 | 1.0 |
| friction_band | ALL | HIGH | FROZEN_VLA_DEFAULT | 5 | 0.2 | [0.0, 0.6] | 0.0 | 0.0 | NA | NA | NA | NA | 1.0 |
| friction_band | ALL | HIGH | FIXED_MAX | 5 | 0.2 | [0.0, 0.6] | 0.0 | 0.0 | NA | NA | 5.0 | NA | 1.0 |
| friction_band | ALL | HIGH | NO_QUERY_PRIOR | 5 | 0.2 | [0.0, 0.6] | 0.0 | 0.0 | NA | NA | 3.7 | NA | 1.0 |
| friction_band | ALL | HIGH | ACTIVEFORCING_DIRECT | 5 | 0.0 | [0.0, 0.0] | 1.0 | 0.6 | 0.0 | 0.0 | 3.75 | 0.8 | 1.0 |
| friction_band | ALL | HIGH | GT_PHYSICS_DIRECT | 5 | 0.0 | [0.0, 0.0] | 1.0 | 0.6 | 0.0 | 0.0 | 3.575 | 0.8 | 1.0 |
| friction_band | ALL | LOW | FROZEN_VLA_DEFAULT | 11 | 0.18181818181818182 | [0.0, 0.45454545454545453] | 0.0 | 0.0 | NA | NA | NA | NA | 1.0 |
| friction_band | ALL | LOW | FIXED_MAX | 10 | 0.0 | [0.0, 0.0] | 0.0 | 0.0 | NA | NA | 5.0 | NA | 1.0 |
| friction_band | ALL | LOW | NO_QUERY_PRIOR | 10 | 0.0 | [0.0, 0.0] | 0.0 | 0.0 | NA | NA | 3.7 | NA | 1.0 |
| friction_band | ALL | LOW | ACTIVEFORCING_DIRECT | 11 | 0.09090909090909091 | [0.0, 0.2727272727272727] | 1.0 | 0.5454545454545454 | 0.09090909090909091 | 0.16666666666666666 | 3.659090909090909 | 0.7999999999999999 | 1.0 |
| friction_band | ALL | LOW | GT_PHYSICS_DIRECT | 10 | 0.0 | [0.0, 0.0] | 1.0 | 0.6 | 0.0 | 0.0 | 4.8125 | 0.7999999999999999 | 1.0 |
| friction_band | ALL | MID | FROZEN_VLA_DEFAULT | 5 | 0.0 | [0.0, 0.0] | 0.0 | 0.0 | NA | NA | NA | NA | 1.0 |
| friction_band | ALL | MID | FIXED_MAX | 5 | 0.0 | [0.0, 0.0] | 0.0 | 0.0 | NA | NA | 5.0 | NA | 1.0 |
| friction_band | ALL | MID | NO_QUERY_PRIOR | 5 | 0.0 | [0.0, 0.0] | 0.0 | 0.0 | NA | NA | 3.7 | NA | 1.0 |
| friction_band | ALL | MID | ACTIVEFORCING_DIRECT | 5 | 0.2 | [0.0, 0.6] | 1.0 | 0.6 | 0.2 | 0.3333333333333333 | 3.625 | 0.8 | 1.0 |
| friction_band | ALL | MID | GT_PHYSICS_DIRECT | 5 | 0.2 | [0.0, 0.6] | 1.0 | 0.6 | 0.2 | 0.3333333333333333 | 3.8 | 0.8 | 1.0 |
