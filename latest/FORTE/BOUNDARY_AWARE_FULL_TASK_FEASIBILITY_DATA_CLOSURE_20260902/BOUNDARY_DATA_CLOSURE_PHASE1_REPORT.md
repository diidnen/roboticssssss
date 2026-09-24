# Boundary-aware full-task feasibility data closure — Phase 1

Status: `CPU_PHASE1_COMPLETE_SIMULATOR_PHASE_DEFERRED`

`DATA_COVERAGE_HYPOTHESIS = NOT_YET_TESTED`

`UTILITY_CHANGE = NO`

`GLOBAL_FMAX_PROMOTION = NO`

## What the authoritative 720 rows establish

The population is 720 TRAIN/DEV branches: four tasks × six root families × three friction conditions × five forces × two repeats. All 720 physical trajectory files exist and align to the frozen Direct/Utility population.

| Task | Force support (N) | Local lift + / - | Full task + / - | Lift=1, full=0 |
|---:|---:|---:|---:|---:|
| 0 | 3.0342–4.9782 | 180 / 0 | 141 / 39 | 39 |
| 1 | 4.0018–5.9901 | 169 / 11 | 125 / 55 | 44 |
| 5 | 3.0042–4.9643 | 180 / 0 | 133 / 47 | 47 |
| 6 | 3.0015–3.9966 | 178 / 2 | 175 / 5 | 3 |

Thus total volume is not the right description of coverage. Task0 and task5 never sample the local-lift failure side. Task6 has only two local failures and three delayed failures, and 16/18 local boundaries are left-censored. No task6 full-task context has a clean unanimous monotone bracket: 13 are left-censored and five are stochastic/nonmonotonic.

Task1 has useful boundary variation but retains a material caveat: 140/180 full-task labels are frozen terminal-height reconstructions. The local-lift labels were independently recovered from trajectories using the original 3 cm rule and agree with every available direct lift label.

## OLD Direct boundary baseline

| Task | Brier | ECE | NLL | AUROC | Clean boundary MAE contexts |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.0515 | 0.0426 | 0.1589 | 0.9810 | 8 |
| 1 | 0.0419 | 0.0543 | 0.1229 | 0.9918 | 7 |
| 5 | 0.0252 | 0.0351 | 0.0812 | 0.9954 | 9 |
| 6 | 0.0339 | 0.0380 | 0.2745 | 0.3206 | 0 |

The preregistered task6 3.01–3.06 N slice contains 10 branches and four full-task failures. Their mean frozen OOF Direct probability across the slice is 0.9802. This is direct evidence of severe low-force overconfidence/ranking failure even though task6's aggregate Brier looks modest because 175/180 labels are positive.

This explains why the existing Expected Utility can choose too little force: the probability gap supplied to Utility is wrong or compressed at the feasibility boundary. It does not yet prove that boundary augmentation fixes the model.

## Acquisition frozen before new outcomes

The qualification population uses root indices 0 and 1 and all LOW/MID/HIGH friction bands for every task: 24 contexts. Task priority 6→0→1→5 affects scheduling only. The state machine is identical across tasks:

1. If the lowest force still lifts, step downward by the calibrated coarse step.
2. Once failure and success bracket a boundary, bisect to the measurable resolution.
3. If repeats overlap or violate force monotonicity, replicate before interpolation.
4. Resolve local lift and full-task transitions separately.
5. Stop at the lower guard and report censoring instead of inventing a frontier.

Every near-boundary force requires at least two repeats. Runtime candidate sets are not expanded; these branches are training/acquisition evidence only.

## Safety and resource gates

The controller accepts float targets and has prior 0.25 N command support. A different legacy task tracked 1.0 N at about 1.196 N, but no inspected source certifies a global safe minimum. Therefore the previous unlaunched 0.5 N assumption is rejected.

The mandatory minimal preflight must validate current-controller 1.0 N stability, 0.25 N adjacent measured separation, tracking MAE ≤0.4 N, no saturation, no runtime error, and peak force ≤8 N.

At the latest resource check, GPU utilization was 73% with 24,063 MiB used. The π0 server, Mass formal collection, and E5 fresh Utility E2E collection were active. No simulator was launched and no process was killed or modified.

## Prepared matched Direct comparison

The primary retraining protocol is frozen as OLD DATA ONLY versus OLD + BOUNDARY DATA, with the same FeasibilityOnly GRU/MLP, inputs, BCE, AdamW, learning rate, weight decay, 80 epochs, batch size, gradient clipping, grouped-root folds, seeds 0/1/2, and fixed optimizer-step budget. π0, identifier, P4-B, friction estimator, Utility, candidate set, and controller remain unchanged.

The training entry point is fail-closed until the augmented manifest is complete, all four tasks are covered, telemetry and state parity pass, and the new branch manifest exists. The current augmented manifest truthfully records 720 old and zero new branches.

## Mechanism handoff

Observed task1 and task6 already contain all three regimes: local failure, local-success/full-failure, and full success. `CURRENT_TASK_E3_MECHANISM_READY.json` is therefore valid as observed TRAIN/DEV mechanism evidence. It does not cancel or replace long-horizon E3.

## Remaining causal test

The following remain unrun:

- minimal force-interface preflight;
- adaptive four-task qualification collection;
- boundary-augmented dataset completion;
- three-seed matched Direct retraining;
- calibration, ranking, boundary MAE, and same-Utility controller comparison;
- promotion decision and genuinely fresh-root validation.

Consequently, neither `BOUNDARY_AWARE_FULL_TASK_FEASIBILITY_DATA_CLOSURE_COMPLETE` nor a Direct method promotion is claimed at this phase.
