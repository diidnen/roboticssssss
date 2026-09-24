# LIBERO-10 task5 nominal qualification

- Candidate: book -> back compartment of caddy.
- Execution: frozen authoritative OpenPI, default task reset, friction 0.6, robust qualification-only 8 N squeeze, one episode, seed 7300, 50 x 10-step maximum horizon.
- Result: **NOMINAL CAPABILITY QUALIFIED**.

| Stage | Result |
|---|---:|
| Query-state/contact reached | 1 |
| Grasp | 1 |
| Lift | 1 |
| Transport >= 0.10 m | 1 |
| Max object XY displacement | 0.433764 m |
| Placement goal | 1 |
| Authoritative final success | 1 |
| Control steps used | 206 |
| Mean measured squeeze | 3.56797 N |
| Peak measured squeeze | 15.56984 N |

Evidence: `/media/volume/newdata/exouser/activeforcing_e3/QUALIFICATION_RUN_20260902_060314_task5/task5/QUALIFICATION_SUMMARY.json` and its referenced episode/step CSVs.

This only qualifies the frozen policy's nominal semantic task capability. The 8 N setpoint is outside the authoritative task5 Utility domain (`Fmax=5 N`) and is excluded from any Utility or in-domain FullTask claim. The next pre-registered pilot searches for all three physical regimes strictly within 1--5 N on TRAIN/DEV roots. If 5 N cannot produce FullTask success, the task is marked outside the frozen Utility domain; Fmax and Utility must not be changed without lineage audit.
