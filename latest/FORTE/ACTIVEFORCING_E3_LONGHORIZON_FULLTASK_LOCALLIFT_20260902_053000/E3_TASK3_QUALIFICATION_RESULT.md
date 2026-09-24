# LIBERO-10 task3 nominal qualification

- Candidate: black bowl -> bottom drawer -> close.
- Execution: frozen authoritative OpenPI, default task reset, friction 0.6, robust qualification-only 8 N squeeze, one episode, seed 7300, 50 x 10-step frozen-policy chunks.
- Result: **NOT QUALIFIED** for a force study with this frozen checkpoint.

| Stage | Result |
|---|---:|
| Query-state/contact reached | 1 |
| Grasp | 0 |
| Lift | 0 |
| Transport >= 0.10 m | 0 |
| Max object XY displacement | 0.0172401 m |
| Placement goal | 0 |
| Close goal | 0 |
| Authoritative final success | 0 |

The episode exhausted 500 control steps without a grasp. Mean and peak measured squeeze were both 0 N, despite the post-inference 8 N setpoint, because the gripper never established grasping contact. This is a nominal semantic/control capability failure, not evidence about ActiveForcing or about the task's force regimes. It therefore does not enter LocalLift/FullTask model fitting.

Inspectable evidence:

- `/media/volume/newdata/exouser/activeforcing_e3/QUALIFICATION_RUN_20260902_055431_retry2/task3/QUALIFICATION_SUMMARY.json`
- `/media/volume/newdata/exouser/activeforcing_e3/QUALIFICATION_RUN_20260902_055431_retry2/task3/logs/task3_mu0.6_episodes.csv`
- `/media/volume/newdata/exouser/activeforcing_e3/QUALIFICATION_RUN_20260902_055431_retry2/task3/logs/task3_mu0.6_steps.csv`

Per the outcome-independent candidate order, task9 is evaluated next. The result does not change the utility, selector, task Fmax, policy, or candidate ordering.
