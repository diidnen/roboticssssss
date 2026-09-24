# ActiveForcing final benchmark design report

## Status

`ACTIVEFORCING_FULL_CLAIM_EXPERIMENT_DESIGN_READY`

## Executive answer

The current authoritative benchmark has four tasks: alphabet soup, cream cheese, tomato sauce, and butter, each picked and placed in a basket. They are a coherent hidden-friction/force-adaptation core, but they are not four distinct manipulation classes. The minimum new benchmark design is one qualified long/delayed-transport task and one qualified turning/constrained-placement task; a separate mass-only plus 3×3 friction×mass collection is required for mass claims.

## Current task map and diagnostics

| ID | Task | Rows | Contexts | Full SR | Mixed-outcome contexts | Post-lift/full-failure candidates | Scientific role |
|---|---|---:|---:|---:|---:|---:|---|
| 0 | alphabet_soup_1 | 180 | 18 | 0.766667 | 11 | 42 | basic hidden-friction adaptation; short basket pick-place; breadth anchor |
| 1 | cream_cheese_1 | 180 | 18 | 0.672222 | 7 | 59 | basic adaptation plus exact-force critical; short basket pick-place; strongest grasp-force qualification |
| 5 | tomato_sauce_1 | 180 | 18 | 0.788889 | 8 | 38 | basic hidden-friction adaptation; short basket pick-place; breadth anchor |
| 6 | butter_1 | 180 | 18 | 0.966667 | 5 | 6 | narrow-force / delayed-failure stress case; short basket pick-place; breadth anchor |

The archive contains 720 valid rows, 72 contexts, 24 task-specific root families, five continuous equal-width force strata per context, and two repeats. Current force supports are task 0: 3–5 N, task 1: 4–6 N, task 5: 3–5 N, and task 6: 3–4 N. Current physics variation is object friction; mass is not varied.

The most important diagnostic is label scope. A post-lift phase is observable, and 145 rows are local-lift-positive/full-task-negative candidates, but the archive does not authoritatively distinguish grip loss, drop, transport, or placement failure for every row. The existing 123-row under-force and 145-row transport-candidate counts are archive-derived diagnostic buckets, not causal ground truth. LocalLift is unusable as a matched comparison because all 720 labels are positive.

## Historical design recovery

Recoverable designs include the current P5-S0-C continuous archive, B2-R2 nine-task oracle breadth, P6G0 t1/t6 grasp-force screening, and the legacy P5-S0-D fresh Q2F E2E run. The latter is useful historical context but is not current Direct all-task E5 evidence. Long transport, turning/acceleration, precise placement, mass, joint friction×mass, and LIBERO-PRO perturbations remain planned or require new qualification.

## Final benchmark taxonomy

Groups A–G are represented as follows: A (basic hidden-friction adaptation) is covered by all four; B (long transport) is not covered as a distinct semantic; C (turning/acceleration) is not covered; D (precise placement) is not covered; E (mass-sensitive) is not covered; F (joint friction+mass) is not covered; G (within-family breadth) is only partially covered. Therefore the four current tasks should remain the core, while the claim surface must stay narrower than the eventual benchmark taxonomy.

## Minimum additions

1. E6: one long/delayed transport task, preferably LIBERO-10 task 7 (two food objects into a basket) or a controlled LongTransportBasket wrapper if frozen-VLA reach fails.
2. E7: one turning/acceleration plus constrained-placement design. Use a controlled TurnAccelRoute and qualify LIBERO-10 task 5 (book to caddy back compartment) as the rigid, precise-placement reserve.
3. E8: after a rigid base task passes qualification, run mass-only low/mid/high and then a 3×3 friction×mass factorial. This is required for mass claims and is not interchangeable with E6/E7.

## Baselines and metrics

The archive-compatible matrix is GT-Direct, Query-Ignored/NoProbe-Direct, ProbeRich-Direct, and ProbeScalar-Direct. Report full-task success, selected/realized force, force-tracking error, under-force, excess-force, utility, and query cost. Exact frozen 0.25 N grid and true zero-query results require fresh matched collection. π0-Default and Tabero-Neutral are not currently safe as matched current baselines: the available report finds a redundant/neutral path and an incomplete fresh current chain.

## E0–E8 decision matrix

E0 integrity and E1 friction identification are complete. E2 is complete for archive-compatible continuous OOF. E3 is blocked by degenerate LocalLift labels. E4 is partial breadth qualification. E5 requires fresh current all-task reset-to-end E2E. E6 long transport, E7 turning/placement, and E8 mass/joint physics require new collections and stage-aware labels.

## Reproducibility and sources

All generated files in this directory are independent analysis outputs. Existing FORTE and Tabero files were read only. The benchmark inventory is grounded in the [official LIBERO repository](https://github.com/Lifelong-Robot-Learning/LIBERO), [LIBERO datasets](https://libero-project.github.io/datasets), [LIBERO paper](https://arxiv.org/abs/2306.03310), [official LIBERO-PRO repository](https://github.com/RLinf/LIBERO-PRO), and [LIBERO-PRO paper](https://arxiv.org/abs/2510.03827).

See `EXISTING_720_TASK_DIAGNOSTICS.csv` for machine-readable diagnostics, `FINAL_E0_E8_EXPERIMENT_MATRIX.csv` for the final stage matrix, and the three handoff files for execution ownership.
