# Probe task/object relation audit

The task/object mapping below is recovered from the authoritative P5-S0-C context manifest and the current prospective TRAIN context manifest. It is not inferred from model results.

| task | object family | historical roots | historical contexts | current common roots | current common contexts |
|---:|---|---:|---:|---:|---:|
| 0 | `alphabet_soup` | 12 | 36 | 6 | 18 |
| 1 | `cream_cheese` | 12 | 36 | 6 | 18 |
| 5 | `tomato_sauce` | 12 | 36 | 6 | 18 |
| 6 | `butter` | 12 | 36 | 6 | 18 |

The four authoritative tasks map to four distinct object families, and no object family repeats across tasks. Consequently leave-one-task-out also leaves out the corresponding object family. Task transfer and object-family transfer are fully confounded in this archive; the primary label is **task/object-distribution-held-out**. No independent object-family split is run because the existing archive contains no cross-task object-family repetition. No new data are collected.
