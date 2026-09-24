# Fresh reset-to-end E2E

This table is an explicit partial accounting, not a completed final-E2E claim. The historical task0/task1 smoke rows are retained; task5 has no rollout artifact, task6 hit an Isaac GPU-foundation segmentation fault, and the current expected-utility runner retries are separately marked diagnostic/partial. No locked TEST rows were loaded.

| method | n | full-task SR | query reach | query valid | mean selected force (N) |
|---|---:|---:|---:|---:|---:|
| FROZEN_VLA_DEFAULT | 6 | 0.0 | 0.0 | 0.0 | 0.0 |
| FIXED_MAX | 6 | 0.0 | 0.0 | 0.0 | 5.0 |
| NO_QUERY_PRIOR | 6 | 0.0 | 0.0 | 0.0 | 3.125 |
| ACTIVEFORCING_DIRECT | 6 | 0.0 | 1.0 | 0.5 | 3.5833333333333335 |
| GT_PHYSICS_DIRECT | 6 | 0.0 | 1.0 | 0.5 | 4.083333333333333 |
