# Task benchmark recovery

This is the authoritative recovery map for the mass-extension branch. A row
marked `CURRENT_DATA_AVAILABLE=YES` means existing data can be inspected; it
does not mean the task is mass-qualified or part of the new formal dataset.

| TASK_ID | TASK_NAME | BENCHMARK_SOURCE | OBJECT | DOWNSTREAM_STRUCTURE | CURRENT_DATA_AVAILABLE | FRICTION_SENSITIVE | MASS_SENSITIVE | DELAYED_FAILURE_AVAILABLE |
|---:|---|---|---|---|---|---|---|---|
| 0 | pick up alphabet soup and place in basket | LIBERO object / Tabero assembled HDF5 | alphabet_soup_1 | lift → transport → basket placement → release | YES | YES | NOT_QUALIFIED_STRUCTURED | Standard and valid-range screens had no force/fixed-force mass sensitivity; standard HIGH query invalid |
| 1 | pick up cream cheese and place in basket | LIBERO object / Tabero assembled HDF5 | cream_cheese_1 | lift → transport → basket placement → release | YES | YES | NOT_QUALIFIED_QUERY_INVALID | All development contexts failed frozen P4-B validity |
| 2 | pick up salad dressing and place in basket | LIBERO object / B2-R2 breadth artifact | salad_dressing_1 | lift → long transport → turning/acceleration → basket placement → release | YES | YES | QUALIFIED_STRUCTURED_TASK2 | YES structured delayed transport failures |
| 3 | pick up BBQ sauce and place in basket | LIBERO object / B2-R2 breadth artifact | bbq_sauce_1 | lift → long transport → turning/acceleration → basket placement → release | YES | NO under friction oracle | NOT_QUALIFIED_STRUCTURED | 15-branch screen had no force/fixed-force mass sensitivity |
| 4 | pick up ketchup and place in basket | LIBERO object suite; no qualifying B2-R2 row recovered | ketchup_1 | lift → transport → basket placement → release | NO_AUTHORITATIVE_BREADTH_ROW | NOT_ASSESSED | NOT_TESTED | UNKNOWN |
| 5 | pick up tomato sauce and place in basket | LIBERO object / Tabero assembled HDF5 | tomato_sauce_1 | lift → transport → basket placement → release | YES | YES | UNKNOWN → M2 qualification | YES in friction archive; mass pending |
| 6 | pick up butter and place in basket | LIBERO object / Tabero assembled HDF5 | butter_1 | lift → transport → basket placement → release | YES | YES | NOT_YET_TESTED | YES in friction archive; mass pending |
| 7 | pick up milk and place in basket | LIBERO object / B2-R2 breadth artifact | milk_1 | lift → transport → basket placement → release | YES | NO under friction oracle | NOT_QUALIFIED_STRUCTURED | 15-branch screen lifted but lost transport retention for all forces |
| 8 | pick up chocolate pudding and place in basket | LIBERO object / B2-R2 breadth artifact | chocolate_pudding_1 | lift → transport → basket placement → release | YES | NO_FEASIBLE_RANGE under friction force range | NOT_QUALIFIED_QUERY_INVALID | All structured contexts failed frozen P4-B validity |
| 9 | pick up orange juice and place in basket | LIBERO object / B2-R2 breadth artifact | orange_juice_1 | lift → transport → basket placement → release | YES | NO under friction oracle | NOT_QUALIFIED_STRUCTURED | 15-branch screen lifted but lost transport retention for all forces |
| L1 | long post-lift transport | historical LIBERO-style design notes; no authoritative run | TBD | extended carry before placement | NO | PLANNED_ONLY | PLANNED_ONLY | NOT_FOUND |
| T1 | turning / acceleration transport | acceleration audit and design notes; no authoritative mass run | TBD | carry with direction change / acceleration | NO | PLANNED_ONLY | PLANNED_ONLY | NOT_FOUND |
| P1 | precise/constrained placement | historical design notes; no authoritative run | TBD | constrained placement tolerance | NO | PLANNED_ONLY | PLANNED_ONLY | NOT_FOUND |

## Source trail

- Existing friction-core task set and caveats: `FORTE/activeforcing_final_closure_20260902_034923/TASK_BENCHMARK_RECOVERY.md`.
- Nine-task friction breadth and classifications: `Tabero/analysis/results/b2r2_tabero_task_breadth_20260821_073931/BENCHMARK_BREADTH_SUMMARY.md`.
- Object identity and available geometry metadata: `.../TASK_OBJECT_NOTES.csv`.
- Task prompts and object names: `Tabero/benchmarks/datasets/libero/config/libero_object.json`.

No planned long-transport, turning, acceleration, constrained-placement, or
multi-stage task is promoted to completed. The recovered structured path was
qualified only for task2; all other screened tasks are explicitly negative or
query-invalid. Planned tasks remain planned and are not included in formal
mass totals.
