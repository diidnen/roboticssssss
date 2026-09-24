# Fresh reset-to-end E2E

This table is an explicit partial accounting, not a completed final-E2E claim. The current utility runner has successful lineage-bearing subsets for task0/task1/task5/task6; earlier task5/task6 failures and diagnostic retries are retained separately. The available tuples remain unbalanced and no locked TEST rows were loaded.

| method | n | full-task SR | query reach | query valid | mean selected force (N) |
|---|---:|---:|---:|---:|---:|
| FROZEN_VLA_DEFAULT | 6 | 0.0 | 0.0 | 0.0 | 0.0 |
| FIXED_MAX | 6 | 0.0 | 0.0 | 0.0 | 5.0 |
| NO_QUERY_PRIOR | 6 | 0.0 | 0.0 | 0.0 | 3.125 |
| ACTIVEFORCING_DIRECT | 6 | 0.0 | 1.0 | 0.5 | 3.5833333333333335 |
| GT_PHYSICS_DIRECT | 6 | 0.0 | 1.0 | 0.5 | 4.083333333333333 |
