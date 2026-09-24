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

This only qualifies the frozen policy's nominal semantic task capability. The 8 N setpoint is excluded from any Utility claim because this new long-horizon task has no frozen task-specific Fmax.

**Correction (2026-09-02):** the earlier statement that `Fmax=5 N` applied here was an invalid suite-local task-ID collision. E1 task5 is `libero_object/task5` (`tomato_sauce_1` -> basket); this candidate is `libero_10/task5` (`black_book_1` -> caddy). The E1 archive Fmax cannot be transferred by numeric ID. The qualification outcome does not define a new Fmax. It does justify a TRAIN-only same-root 6/7/8 N force-support extension for label-support analysis, with no Utility claim and no TEST use.
