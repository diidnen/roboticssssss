# Joint Decision Alignment Final Report

## Frozen intervention

`Joint-DecisionAligned` changes only the training objective:

`L_new = L_original + 0.3 * L_rank`

`L_rank = softplus(-(s_success - s_failure))`, using 349 valid TRAIN-only pairs across 21/48 TRAIN contexts. The model, inputs, physics/IE/trajectory heads, feasibility head, optimizer, seeds, budget, final-epoch checkpoint rule, threshold `p_success >= 0.5`, candidate support, and downstream inference remain unchanged. No boundary loss, force-regression head, threshold change, or new data was added.

## Learning curve

| Data | Direct | Joint-Original | Joint-DecisionAligned |
|---|---:|---:|---:|
| 25% | 0.729 | 0.667 | 0.667 |
| 50% | 0.750 | 0.708 | 0.708 |
| 75% | 0.667 | 0.708 | 0.708 |
| 100% | 0.750 | 0.708 | 0.708 |

DecisionAligned does not improve over Original Joint at any fraction and does not reach Direct at 100%.

## Full-data DEV

| Method | SR | Under-force | Ranking | NLL | Mean Force |
|---|---:|---:|---:|---:|---:|
| Direct | 0.750 | 0.125 | 0.906 | 0.352 | 3.738 |
| Joint-Original | 0.708 | 0.167 | 0.821 | 0.700 | 3.712 |
| Joint-DecisionAligned | 0.708 | 0.167 | 0.772 | 0.840 | 3.677 |

At 100%, DecisionAligned remains SR `0.708`, under-force `0.167`, and is not an always-max-force solution: its mean selected force is `3.677 N` versus Original `3.712 N`, and its max-candidate selection rate is `0.042`.

## Task0 / per-scene

| Scene | Direct | Original Joint | DecisionAligned |
|---|---:|---:|---:|
| alphabet_soup | 0.667 | 0.500 | 0.500 |
| cream_cheese | 0.667 | 0.667 | 0.667 |
| tomato_sauce | 0.833 | 0.833 | 0.833 |
| butter | 0.833 | 0.833 | 0.833 |

DecisionAligned does not repair the alphabet_soup gap: task0 remains `0.500` versus Direct `0.667`.

## Mechanism conclusion

The ranking-only intervention did not support `H_JOINT_DECISION_ALIGNMENT`. It drove the ranking training loss down, but did not improve the downstream minimum-force decision. Relative to Original Joint, DecisionAligned has unchanged SR and under-force, worse boundary-proxy MAE (`0.155 N` vs `0.118 N`), worse all-pair ordering (`0.772` vs `0.821`), and worse feasibility NLL (`0.840` vs `0.700`).

The supported result is therefore that a simple pairwise ranking term is insufficient and may be harmful through score saturation/calibration drift. This does not prove that every decision-aligned objective would fail; it rejects this minimal ranking-only intervention under the frozen protocol.

## Verdict

`JOINT_AUXILIARY_OBJECTIVES_REMAIN_HARMFUL`

No boundary-loss variant was run in this turn. No architecture search or post-hoc hyperparameter selection was performed.

Data decision: `NO — DATA IS NOT THE MAIN BOTTLENECK`.

TEST: `TEST NOT OPENED`.
