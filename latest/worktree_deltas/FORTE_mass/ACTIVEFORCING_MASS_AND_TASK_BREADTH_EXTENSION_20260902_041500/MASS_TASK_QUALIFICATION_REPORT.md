# Mass task qualification report

Qualification is a development screen and is not part of formal totals.

| task | valid query contexts | branches | SR | force-sensitive | mass-sensitive | delayed failures | status |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 9 | 45 | 1.000 | 0 | 0 | 0 | NOT_QUALIFIED |
| 1 | 0 | 0 | 0.000 | 0 | 0 | 0 | NOT_QUALIFIED_QUERY_INVALID |
| 2 | 6 | 30 | 1.000 | 1 | 1 | 20 | QUALIFIED_STRUCTURED_TASK2 |
| 3 | 6 | 30 | 1.000 | 0 | 0 | 0 | NOT_QUALIFIED |
| 5 | 0 | 0 | 0.000 | 0 | 0 | 0 | NOT_QUALIFIED_QUERY_INVALID |
| 6 | 0 | 0 | 0.000 | 0 | 0 | 0 | NOT_QUALIFIED_QUERY_INVALID |
| 7 | 3 | 15 | 1.000 | 0 | 0 | 0 | NOT_QUALIFIED |
| 8 | 0 | 0 | 0.000 | 0 | 0 | 0 | NOT_QUALIFIED_QUERY_INVALID |
| 9 | 3 | 15 | 1.000 | 0 | 0 | 0 | NOT_QUALIFIED |

## Promotion rule

A task is promoted only if the query is valid, outcomes are mixed, and force or mass changes the outcome. The existing frozen-LIBERO reach asset is recorded as a prerequisite; no new VLA is trained.

## Structured-path recovery update

The basic basket path was saturated for task0/task2/task3 and invalid for
task1/task5, so it was not promoted.
The recovered long-transport/turning/acceleration structure was then qualified
with the frozen P4-B query, masses 0.05/0.10/0.20 kg, and forces
0.5/1.0/1.5/2.5/4.0 N. Task2 passed on two independent roots (30 branches;
100% lift; mixed transport/full-task outcomes; force and fixed-force mass
sensitivity). Its full details are in
`analysis/MASS_STRUCTURED_QUALIFICATION_REPORT.md`. Task3, task7, and task9
completed negative structured screens; task6 and task8 were query-invalid.
No second mass-sensitive task was found, so the breadth gate is a completed
negative result rather than an unreported planned task.
