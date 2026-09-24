# M1 Force-Sufficiency Model

Status: `M1_IID_WORKS_BUT_CROSS_TASK_FAILS`.

This directory contains the first OURS-style M1 experiment: a small task-conditioned force-sufficiency model trained on frozen full-task force labels.

The model answers: for this task/context, this probe evidence, and candidate force F, is full-task success likely?

No benchmark, D2, Tabero core, Tabero-VTLA checkpoint, FORTE optimization, probe policy, or canonical F* was modified.

Key artifacts:

- `M1_DATASET.parquet` and `M1_DATASET.csv`
- `TASK_ONLY_RESULTS.csv`, `PROBE_ONLY_RESULTS.csv`, `TASK_PROBE_RESULTS.csv`
- `IID_OFFLINE_RESULTS.csv`, `LOTO_OFFLINE_RESULTS.csv`, `OOD_FRICTION_RESULTS.csv`
- `MAIN_METHOD_TABLE.md`, `GENERALIZATION_TABLE.md`, `FAILURE_ANALYSIS.md`
- `FINAL_VERDICT.json`

Online full downstream was not run. `ONLINE_MAIN_RESULTS.csv` and `ONLINE_TASK_SUMMARY.csv` contain the offline counterfactual decision proxy and are explicitly marked as such.
