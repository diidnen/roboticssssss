# LIBERO-10 task9 nominal qualification

- Candidate: yellow-white mug -> microwave -> close.
- Execution: frozen authoritative OpenPI, default task reset, friction 0.6, robust qualification-only 8 N squeeze, one episode, seed 7300, 50 x 10-step frozen-policy chunks.
- Result: **NOT QUALIFIED** for a force study with this frozen checkpoint.

| Stage | Result |
|---|---:|
| Query-state/contact reached | 1 |
| Grasp flag | 1 |
| Lift | 0 |
| Transport >= 0.10 m | 0 |
| Max object XY displacement | 0.0377510 m |
| Placement goal | 0 |
| Close goal | 0 |
| Authoritative final success | 0 |

The policy triggered the environment's grasp subtask flag but never lifted or retained the mug; measured squeeze mean and peak were 0 N. It exhausted 500 steps without placement or microwave closure. Thus the result is a nominal semantic/control capability failure, not evidence about ActiveForcing or force-regime separability, and it is excluded from force-model fitting.

Evidence: `/media/volume/newdata/exouser/activeforcing_e3/QUALIFICATION_RUN_20260902_055939_task9/task9/QUALIFICATION_SUMMARY.json` and its referenced episode/step CSVs.

Per the pre-registered outcome-independent order, task5 is evaluated next. No utility, selector, task Fmax, policy, or candidate-order parameter changed after observing task9.
