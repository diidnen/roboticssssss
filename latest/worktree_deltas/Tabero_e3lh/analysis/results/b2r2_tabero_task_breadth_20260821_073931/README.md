# B2-R2 Tabero official task breadth

Status: `B2R2_BENCHMARK_BREADTH_STRONG`

This directory contains the independent B2-R2 breadth pass for the 9 official Tabero LIBERO-object tasks. It only varies object friction (`mu = 0.20, 0.50, 1.00`) and evaluates full-task success under controlled force cells. No D2 thresholds, Tabero core code, provisional OURS thresholds, DeliGrasp, FORTE, or Tabero Neutral methods were changed or run here.

Primary artifacts:

- `FINAL_VERDICT.json`
- `TABERO_9TASK_PHYSICS_DECISION_TABLE.md`
- `FULLTASK_CANONICAL_FSTAR.csv`
- `FIXED_ROBUST_BY_TASK.csv`
- `GT_MINFORCE_BY_TASK.csv`
- `EXPANDED_POSITIVE_TASKS.csv`
- `BENCHMARK_BREADTH_SUMMARY.md`
- `CELL_QUALITY_SUMMARY.csv`
- `TASK_OBJECT_NOTES.csv`

Frozen read-only inputs:

- B2: `/home/exouser/Tabero/analysis/results/b2_tabero_benchmark_table_20260820_065520`
- E2E2: `/home/exouser/Tabero/analysis/results/e2e2_oracle_reconcile_milk_breadth_20260821_053312`

Generated at: `2026-08-21T12:38:42.927178+00:00`
