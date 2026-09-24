# Current Direct utility fresh E2E (partial)

This is a separate lineage-bearing accounting of timestamped utility-run rows. It is not the required all-task final E2E table: missing task/root tuples remain missing and are not imputed.

| method | n | full-task SR | query reach | query valid | mean selected force (N) | lineage valid |
|---|---:|---:|---:|---:|---:|---:|
| FROZEN_VLA_DEFAULT | 11 | 0.2727272727272727 | 0.0 | 0.0 | NA | 1.0 |
| FIXED_MAX | 11 | 0.09090909090909091 | 0.0 | 0.0 | 5.181818181818182 | 0.9090909090909091 |
| NO_QUERY_PRIOR | 9 | 0.0 | 0.0 | 0.0 | 3.861111111111111 | 1.0 |
| ACTIVEFORCING_DIRECT | 10 | 0.2 | 1.0 | 0.8 | 3.8 | 1.0 |
| GT_PHYSICS_DIRECT | 9 | 0.2222222222222222 | 1.0 | 0.7777777777777778 | 4.583333333333333 | 1.0 |
