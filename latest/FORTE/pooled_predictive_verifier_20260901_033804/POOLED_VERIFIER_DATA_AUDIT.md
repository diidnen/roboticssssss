# Pooled verifier data audit

PASS. The authoritative development population is **720 TRAIN branches from 72 friction-conditioned contexts and 24 independent root families**, pooled across task0/task1/task5/task6. Each task contributes 6 root families, 18 contexts, and 180 branches. A context is a root-state-friction realization; a force branch and its repeats are not independent contexts.

| Grain | Per task | Pooled |
|---|---:|---:|
| Independent root families | 6 | 24 |
| Friction-conditioned contexts | 18 | 72 |
| Force cells | 90 | 360 |
| Branches (5 forces × 2 repeats/context) | 180 | 720 |
| H8 physical timesteps | 1,440 | 5,760 |
| Adjacent-force IE pairs | 144 | 576 |

All 720 branches have corrected physical telemetry and a final full-task outcome. The World Model target is the frozen H=8, 13-channel physical trajectory; the IE target is the matched adjacent-force trajectory difference within a context and repeat. Visual features exist for all 72 contexts but are **excluded** from Direct, WM, and verifier inputs in this experiment.

## Label lineage and task1 caveat

task0/task5/task6 use complete direct branch outcomes. task1 has **40 direct labels and 140 frozen terminal-height fallback labels**. On the 40 branches where both exist, mismatches are 0. This does not remove the material caveat: task1 metrics partially measure the preregistered reconstruction rule, not uniformly direct execution labels.

## Grouped CV

Three folds are frozen before training. Each fold holds out 2 whole root families per task (8 pooled), including every friction condition, force, and repeat under those roots. Each model trains jointly on the remaining four tasks' roots: 16 roots, 48 contexts, 480 branches. Validation has 8 roots, 24 contexts, 240 branches. There is no root/context/branch leakage.

## Scope

This is pooled multi-task TRAIN-rootheldout development with authoritative GT physics. It is not Active Probe E2E, not untouched TEST, and not unseen-task or cross-object evidence. The already-viewed old fixed-scene DEV is used only for retrospective difficulty composition.
