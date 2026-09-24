# Current Direct utility fresh E2E (partial)

This is a separate lineage-bearing accounting of timestamped utility-run rows. Smoke-only roots are excluded and repeated retries are deduplicated by task/root/method. It is not the required all-task final E2E table: missing task/root tuples remain missing and are not imputed.

| method | n | full-task SR | query reach | query valid | mean selected force (N) | lineage valid |
|---|---:|---:|---:|---:|---:|---:|
| FROZEN_VLA_DEFAULT | 5 | 0.2 | 0.0 | 0.0 | NA | 1.0 |
| FIXED_MAX | 5 | 0.0 | 0.0 | 0.0 | 5.0 | 1.0 |
| NO_QUERY_PRIOR | 5 | 0.0 | 0.0 | 0.0 | 3.7 | 1.0 |
| ACTIVEFORCING_DIRECT | 5 | 0.0 | 1.0 | 0.6 | 3.75 | 1.0 |
| GT_PHYSICS_DIRECT | 5 | 0.2 | 1.0 | 0.6 | 3.525 | 1.0 |
