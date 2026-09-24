# Physical identification report

The frozen physical-history estimator has been re-summarized from the existing grouped-root OOF artifact. Vision-only and vision+physical are retained only where the existing matched visual artifact exists; no full-task recollection was performed.

| split | method | contexts | MAE | RMSE | Spearman | pair ranking |
|---|---|---:|---:|---:|---:|---:|
| LOTO_TASK_OBJECT_HELDOUT | ExplicitSysID | 144 | 0.1810 | 0.2181 | 0.7297 | 0.8611 |
| LOTO_TASK_OBJECT_HELDOUT | LearnedProbe | 576 | 0.5149 | 0.5665 | 0.5258 | 0.7917 |
| ROOT_HELDOUT_OOF | Probe-PhysicalHistory | 72 | 0.0655 | 0.1044 | 0.9018 | NA |

Caveat: LOTO is task/object-family heldout because the archive confounds task and object family. The current frozen estimator is a point estimate; sigma is diagnostic only.
