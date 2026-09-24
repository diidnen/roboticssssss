# WHY DID OLD JOINT LOOK BETTER?

## Bottom line

The old Joint had a real but narrow **frontier-localization signal**. It did not establish a better controller. The signal occurred in a different data/evaluator regime: no visual input, 72 pooled contexts across four tasks, 24 roots, 14 outcome executions/context (10 new continuous plus four coarse), pooled normalization, and a small 9-context/7-root DEV benchmark. Current single-task runs use 18 contexts/six roots/10 executions per context. The matched current pooled diagnostic restores 72 contexts but does not restore a consistent calibrated or safe Joint advantage.

The authoritative old and current taskwise physics auxiliary are architecturally aligned: hidden 64, H8×13 trajectory target, target-matched adjacent-force IE, AdamW 8e-4/1e-4, 80 epochs, seeds 0/1/2, and native `Lphysics + LIE + 0.3 Lfeas`. Therefore an architecture change in that core is not the explanation. The original pooled Visual pipeline is a separate critical confound because it changes that formulation.

## What the old result actually showed

| model | probability MAE | Brier | NLL | frontier MAE N | under-force | nonmonotonic context rate |
| --- | --- | --- | --- | --- | --- | --- |
| FEASIBILITY_ONLY | 0.2449149610483981 | 0.085850007813188 | 0.4627065015901164 | 0.325 | 0.25 | 0.0 |
| JOINT | 0.2074505620756196 | 0.1351642865156097 | 0.9387264804756326 | 0.1249999999999998 | 0.375 | 0.6666666666666666 |

Old Joint improved frontier MAE from 0.325 N to 0.125 N and slightly reduced probability MAE. At the same time Brier rose from 0.0859 to 0.1352, NLL from 0.4627 to 0.9387, under-force from 25% to 37.5%, and systematic nonmonotonic contexts from 0% to 66.7%. The valid frontier set had only eight contexts. This supports `JOINT_HAS_FRONTIER_LOCALIZATION_SIGNAL`, not `JOINT_IS_A_BETTER_CONTROLLER`.

## Context and execution counts

| regime | independent contexts | simulator roots | executions/context | feasibility outcomes | physical/IE branches |
|---|---:|---:|---:|---:|---:|
| old pooled continuous | 72 | 24 | 14 | 1,008 | 720 |
| current single task | 18 | 6 | 10 | 180 | 180 |
| current matched pooled | 72 | 24 | 10 | 720 | 720 |

Executions and physical timesteps are not independent contexts.

## Did more contexts reproduce the old gain?

No, not consistently. In pooled root-CV, JNV improves NLL/Brier for task5 and task6 but not task0/task1; it improves probability MAE for all four tasks, while under-force worsens for all four and frontier worsens for task0/1/5. On prospective pooled DEV, JNV NLL is worse for task0/task1/task5 and improves only task6, which has one DEV root. This is partial regularization, not recovery of a reliable Joint.

## Remaining old/current confounds

1. Old feasibility supervision includes 288 additional coarse branches; current matched pooled does not.
2. The simulator-root population and sampled forces differ even though support ranges and five-stratum/two-repeat design match.
3. Old DEV has 27 real force cells/135 executions; current prospective DEV has 77 cells/385 executions and a broader per-context force grid.
4. Old and current normalization populations differ.
5. The old positive headline was frontier MAE; its probability calibration, safety and monotonicity were worse.

Because current pooled diversity does not consistently rescue the same authoritative no-visual formulation, the evidence does not permit “the old result worked simply because it had more data.” Dataset distribution, added coarse supervision and evaluator/frontier population remain genuine explanations to audit.
