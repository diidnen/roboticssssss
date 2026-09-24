# Direct Utility + Trajectory-Only World-Model Verifier

The evaluator was trained directly on frozen TRAIN imagined trajectories, removing the old real-to-imagined evaluator distribution mismatch. It sees only H8 predicted trajectory summaries. Device: `cpu`. No TEST was read.

## Probability quality

DEV branch AUROC=0.894, Brier=0.135, NLL=0.443; failed-branch p>0.8 rate=0.000.

## Decision result

| Policy | SR | Under-force | Mean force | Realized utility | Validated |
|---|---:|---:|---:|---:|---:|
| DIRECT_UTILITY | 0.875 | 0.042 | 3.932 | 0.315 | 1.000 |
| WORLD_UTILITY | 0.812 | 0.125 | 3.826 | -0.064 | 1.000 |
| DIRECT_THEN_WORLD_GT_0P8 | 0.958 | 0.000 | 4.718 | 0.026 | 0.167 |

## Gate audit

Direct proposals accepted=1/24; correctly rejected imperfect=4; incorrectly rejected perfect=19; incorrectly accepted imperfect=0.

This is a retrospective fixed-scene DEV mechanism diagnostic. It does not select a production backend and cannot be reported as fresh task0 TEST evidence.
