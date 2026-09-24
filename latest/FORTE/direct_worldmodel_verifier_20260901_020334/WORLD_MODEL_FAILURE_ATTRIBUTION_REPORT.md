# World-model hard-search failure attribution

This forensic separates H8 outcome observability, predicted-trajectory error, and evaluator error using the 2x2 matrix of REAL/PRED trajectory inputs and REAL/PRED-trained evaluators. No controller or frozen model was changed.

| Input / evaluator | AUROC | Brier | NLL |
|---|---:|---:|---:|
| REAL trajectory → REAL-trained evaluator | 0.689 | 0.168 | 0.518 |
| PRED trajectory → REAL-trained evaluator | 0.321 | 0.253 | 0.928 |
| REAL trajectory → PRED-trained evaluator | 0.650 | 0.166 | 0.510 |
| PRED trajectory → PRED-trained evaluator | 0.894 | 0.135 | 0.443 |

Case attribution counts: `{"FORCE_SUPPORT_HAS_NO_OBSERVED_SUCCESS": 1, "PREDICTED_TRAJECTORY_EVALUATOR_FALSE_NEGATIVE": 3, "STOCHASTIC_REPEAT_AMBIGUITY": 1, "WORLD_MODEL_PREDICTED_TRAJECTORY_SHIFT": 1}`. See `WORLD_MODEL_FAILURE_CASE_ATTRIBUTION.csv` for exact contexts and top trajectory-error channels.
