# B5 Tabero-VTLA Neutral Hidden-Physics Baseline

Status: `B5_TABERO_NEUTRAL_HIDDEN_PHYSICS_BLIND_WITH_TASK_FAILURES`.

`METHOD_CHANGE = NONE`. This run used the official Tabero-VTLA / openpi inference path with neutral official task instructions and no force adverbs.

## Protocol

- Tasks: `[0, 1, 2, 5, 6]`.
- Hidden frictions: `[0.2, 0.5, 1.0]`.
- Main grid: `N=10` per task/friction cell.
- Key stress expansion: task2 salad dressing at `mu=0.20` rerun at `N=20`.
- Horizon: `max_inference_steps=50`, `replan_steps=10`, `num_success_steps=8`.
- Force adverbs: none.
- Benchmark F*: frozen; not modified.

## Main Result

Aggregate Neutral full SR: `0.55` vs Fixed Robust `0.993`.

Aggregate Neutral measured force: `7.335N`; GT-MinForce reference measured force: `2.555N`.

| task | object | low-mu SR/F | mid-mu SR/F | high-mu SR/F | canonical F* | classification |
| ---: | --- | ---: | ---: | ---: | --- | --- |
| 0 | alphabet_soup_1 | 1.0 / 15.391N (n=10) | 1.0 / 11.393N | 1.0 / 6.585N | 5/4/3N | `NEUTRAL_ROBUST_OVERFORCE` |
| 1 | cream_cheese_1 | 0.7 / 11.818N (n=10) | 0.9 / 14.412N | 0.7 / 11.563N | 6/5/3N | `NEUTRAL_TASK_FAILURE` |
| 2 | salad_dressing_1 | 0.05 / 0.644N (n=20) | 0.4 / 5.763N | 0.8 / 6.914N | 8/3/3N | `NEUTRAL_HIDDEN_PHYSICS_BLIND` |
| 5 | tomato_sauce_1 | 0.2 / 3.61N (n=10) | 0.5 / 7.463N | 0.1 / 2.345N | 5/4/3N | `NEUTRAL_TASK_FAILURE` |
| 6 | butter_1 | 0.1 / 2.075N (n=10) | 0.6 / 7.01N | 0.2 / 3.046N | 4/3/3N | `NEUTRAL_TASK_FAILURE` |

## Interpretation

Tabero Neutral is not a strong hidden-physics baseline on this frozen benchmark. It does not reliably recover low-friction cases, and predicted force slots do not show the canonical hidden-mu ordering on any task. Task0 is the strongest case for Tabero: it succeeds across all frictions, but with overforce relative to the minimum-force reference.

## Limitations

- Main grid is `N=10`; task2 low-mu stress cell is `N=20`.
- Measured squeeze force and predicted force slots are reported separately. Predicted slots are policy outputs and are not literal contact Newtons.
- Optional gentle/firm language reference and task7 negative control were not run.
