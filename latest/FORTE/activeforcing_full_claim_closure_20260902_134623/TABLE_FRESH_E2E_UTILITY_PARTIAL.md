# Current Direct utility fresh E2E (partial)

This is a separate lineage-bearing accounting of timestamped utility-run rows. Smoke-only roots are excluded and repeated retries are deduplicated by task/root/friction/method. It is not the required all-task final E2E table: missing task/root tuples remain missing and are not imputed.

| method | n | full-task SR | query reach | query valid | mean selected force (N) | lineage valid |
|---|---:|---:|---:|---:|---:|---:|
| FROZEN_VLA_DEFAULT | 21 | 0.14285714285714285 | 0.0 | 0.0 | NA | 1.0 |
| FIXED_MAX | 21 | 0.047619047619047616 | 0.0 | 0.0 | 5.0476190476190474 | 1.0 |
| NO_QUERY_PRIOR | 21 | 0.047619047619047616 | 0.0 | 0.0 | 3.7261904761904763 | 1.0 |
| ACTIVEFORCING_DIRECT | 21 | 0.09523809523809523 | 1.0 | 0.5714285714285714 | 3.6726190476190474 | 1.0 |
| GT_PHYSICS_DIRECT | 21 | 0.047619047619047616 | 1.0 | 0.5714285714285714 | 4.333333333333333 | 1.0 |
