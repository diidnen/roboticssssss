# Current Direct utility fresh E2E (partial)

This is a separate lineage-bearing accounting of timestamped utility-run rows. It is not the required all-task final E2E table: missing task/root tuples remain missing and are not imputed.

| method | n | full-task SR | query reach | query valid | mean selected force (N) | lineage valid |
|---|---:|---:|---:|---:|---:|---:|
| FROZEN_VLA_DEFAULT | 8 | 0.375 | 0.0 | 0.0 | NA | 1.0 |
| FIXED_MAX | 8 | 0.125 | 0.0 | 0.0 | 5.0 | 0.875 |
| NO_QUERY_PRIOR | 6 | 0.0 | 0.0 | 0.0 | 3.75 | 1.0 |
| ACTIVEFORCING_DIRECT | 7 | 0.2857142857142857 | 1.0 | 1.0 | 3.8214285714285716 | 1.0 |
| GT_PHYSICS_DIRECT | 6 | 0.3333333333333333 | 1.0 | 1.0 | 4.291666666666667 | 1.0 |
