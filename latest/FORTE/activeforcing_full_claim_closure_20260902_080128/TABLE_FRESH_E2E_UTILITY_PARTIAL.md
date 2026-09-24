# Current Direct utility fresh E2E (partial)

This is a separate lineage-bearing accounting of timestamped utility-run rows. It is not the required all-task final E2E table: missing task/root tuples remain missing and are not imputed.

| method | n | full-task SR | query reach | query valid | mean selected force (N) | lineage valid |
|---|---:|---:|---:|---:|---:|---:|
| FROZEN_VLA_DEFAULT | 6 | 0.3333333333333333 | 0.0 | 0.0 | NA | 1.0 |
| FIXED_MAX | 6 | 0.16666666666666666 | 0.0 | 0.0 | 5.0 | 0.8333333333333334 |
| NO_QUERY_PRIOR | 4 | 0.0 | 0.0 | 0.0 | 3.75 | 1.0 |
| ACTIVEFORCING_DIRECT | 5 | 0.2 | 1.0 | 1.0 | 3.85 | 1.0 |
| GT_PHYSICS_DIRECT | 4 | 0.25 | 1.0 | 1.0 | 4.3125 | 1.0 |
