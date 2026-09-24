# Current Direct utility fresh E2E (partial)

This is a separate lineage-bearing accounting of timestamped utility-run rows. Smoke-only roots are excluded and repeated retries are deduplicated by task/root/friction/method. It is not the required all-task final E2E table: missing task/root tuples remain missing and are not imputed.

| method | n | full-task SR | query reach | query valid | mean selected force (N) | lineage valid |
|---|---:|---:|---:|---:|---:|---:|
| FROZEN_VLA_DEFAULT | 11 | 0.18181818181818182 | 0.0 | 0.0 | NA | 1.0 |
| FIXED_MAX | 11 | 0.0 | 0.0 | 0.0 | 5.181818181818182 | 1.0 |
| NO_QUERY_PRIOR | 11 | 0.0 | 0.0 | 0.0 | 3.8181818181818183 | 1.0 |
| ACTIVEFORCING_DIRECT | 11 | 0.18181818181818182 | 1.0 | 0.6363636363636364 | 3.75 | 1.0 |
| GT_PHYSICS_DIRECT | 11 | 0.09090909090909091 | 1.0 | 0.6363636363636364 | 4.238636363636363 | 1.0 |
