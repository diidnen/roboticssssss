# Current Direct utility fresh E2E (partial)

This is a separate lineage-bearing accounting of timestamped utility-run rows. Smoke-only roots are excluded and repeated retries are deduplicated by task/root/friction/method. It is not the required all-task final E2E table: missing task/root tuples remain missing and are not imputed.

| method | n | full-task SR | query reach | query valid | mean selected force (N) | lineage valid |
|---|---:|---:|---:|---:|---:|---:|
| FROZEN_VLA_DEFAULT | 12 | 0.16666666666666666 | 0.0 | 0.0 | NA | 1.0 |
| FIXED_MAX | 12 | 0.0 | 0.0 | 0.0 | 5.083333333333333 | 1.0 |
| NO_QUERY_PRIOR | 12 | 0.0 | 0.0 | 0.0 | 3.75 | 1.0 |
| ACTIVEFORCING_DIRECT | 12 | 0.16666666666666666 | 1.0 | 0.5833333333333334 | 3.7083333333333335 | 1.0 |
| GT_PHYSICS_DIRECT | 12 | 0.08333333333333333 | 1.0 | 0.5833333333333334 | 4.166666666666667 | 1.0 |
