# Current Direct utility fresh E2E (partial)

This is a separate lineage-bearing accounting of timestamped utility-run rows. It is not the required all-task final E2E table: missing task/root tuples remain missing and are not imputed.

| method | n | full-task SR | query reach | query valid | mean selected force (N) | lineage valid |
|---|---:|---:|---:|---:|---:|---:|
| FROZEN_VLA_DEFAULT | 12 | 0.25 | 0.0 | 0.0 | NA | 1.0 |
| FIXED_MAX | 12 | 0.08333333333333333 | 0.0 | 0.0 | 5.083333333333333 | 0.9166666666666666 |
| NO_QUERY_PRIOR | 10 | 0.0 | 0.0 | 0.0 | 3.775 | 1.0 |
| ACTIVEFORCING_DIRECT | 11 | 0.18181818181818182 | 1.0 | 0.7272727272727273 | 3.75 | 1.0 |
| GT_PHYSICS_DIRECT | 10 | 0.2 | 1.0 | 0.7 | 4.4625 | 1.0 |
