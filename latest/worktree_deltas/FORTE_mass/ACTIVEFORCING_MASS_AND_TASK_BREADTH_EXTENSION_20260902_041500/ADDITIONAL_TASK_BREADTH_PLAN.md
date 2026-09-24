# Additional task breadth plan

## Current decision

The basic pick/place screen saturated task0/task2/task3 or lacked valid query
contexts for task1/task5. The first and only credible mass-sensitive candidate
is task2 salad dressing under the recovered
LONG_TRANSPORT_TURNING_ACCELERATION_V1 structure. Tasks0/1/3/5/6/7/8/9 were
screened and did not pass; this is a completed negative breadth result, not a
pending task list. This does not alter the existing four-task friction line.

## Qualification matrix

For each candidate task, run two independent roots × three fixed masses
(0.05/0.10/0.20 kg) × five frozen forces (0.5/1.0/1.5/2.5/4.0 N), with one P4-B query and one
downstream branch per force. Required evidence is lift reach, full-task outcome,
force sensitivity, delayed-failure count, and failure stage. The qualification
screen is excluded from paper totals.

## Breadth backlog

1. **Basic pick/place:** tasks0/1/5/6 were screened and did not qualify under
   the frozen query and fixed force set.
2. **Long transport:** no authoritative current run was recovered. Reuse the
   same object/root conventions only after a task-specific HDF5 and frozen-VLA
   reach check.
3. **Turning/acceleration:** existing acceleration notes reject unvalidated
   collector acceleration; no task is claimed complete. A future candidate
   must preserve the force-setpoint-only controller interface.
4. **Constrained/precise placement:** not found as an authoritative current
   task. Do not infer this property from ordinary basket placement.

## Promotion rule

A task is promoted only if the qualification artifact records frozen-VLA reach
as available, valid P4-B query traces, at least two force outcomes among
success/failure, and at least one delayed downstream failure or a preregistered
negative result. Planned tasks remain `PLANNED_ONLY` in the recovery table.
