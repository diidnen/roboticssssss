# Current Direct utility fresh E2E (partial)

This is a separate lineage-bearing accounting of timestamped utility-run rows. It is not the required all-task final E2E table: missing task/root tuples remain missing and are not imputed.

| method | n | full-task SR | query reach | query valid | mean selected force (N) | lineage valid |
|---|---:|---:|---:|---:|---:|---:|
| FROZEN_VLA_DEFAULT | 10 | 0.3 | 0.0 | 0.0 | NA | 1.0 |
| FIXED_MAX | 10 | 0.1 | 0.0 | 0.0 | 5.2 | 0.9 |
| NO_QUERY_PRIOR | 8 | 0.0 | 0.0 | 0.0 | 3.875 | 1.0 |
| ACTIVEFORCING_DIRECT | 9 | 0.2222222222222222 | 1.0 | 0.7777777777777778 | 3.861111111111111 | 1.0 |
| GT_PHYSICS_DIRECT | 8 | 0.25 | 1.0 | 0.75 | 4.5625 | 1.0 |
