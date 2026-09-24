# E3 current-task three-regime audit

Source: authoritative 720-row branch-label audit at `/home/exouser/FORTE/activeforcing_final_closure_20260902_034923/E3_BRANCH_LABEL_AUDIT.csv`.

All four current tasks have a LocalLift positive rate of exactly `1.0`. They provide many `LOCAL SUCCESS + DOWNSTREAM FAILURE` rows and full-task successes, but zero `LOCAL FAILURE` rows. A learned LocalLift classifier would therefore be scientifically degenerate.

The older B2 coarse force scan independently shows the same pattern for task0/task5 at 4 N and all-success task6. It does contain a single local failure for task2 and a stronger three-regime pattern for task8, but those are short food-object-to-basket tasks and are not substituted for the preregistered long-horizon qualification.

Decision: do not force the current task0/1/5/6 family into E3. Qualify historical long-horizon tasks without selecting tasks based on ActiveForcing performance.
