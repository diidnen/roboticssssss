# Final Semantic Shared Direct Report

## Straight answer

**FULL_TASK_FEASIBILITY_IS_TASK_SPECIFIC.** Under accurate GT physics, changing only task representation moves macro B=0 SR from **81.94%** to **84.03%** (+2.08 pp), but B10/B20 change by **-7.41/-8.56 pp**, and the preregistered GT macro B* worsens from **30** (one-hot) to **60** (semantic). Therefore the previous poor transfer was **not primarily caused by the one-hot encoding**. This fixed semantic representation does not resolve transfer, and substantial target-task full-outcome supervision remains necessary.

The deployable physics comparison is separate: SysID-SemanticDirect and LearnedProbe-SemanticDirect use exactly the previous source-only estimates. Their macro B* values are **60** and **60**. Thus semantic task transfer and physics-estimator transfer are not conflated.

## Frozen semantic representation

The old Direct used a 4D task one-hot. The new Direct uses the actual neutral task instruction encoded by the frozen π0 checkpoint's PaliGemma/Gemma input embedding table: valid token embeddings are mask-mean pooled to 2048D, then passed through one outcome-independent frozen 2048→16 projection (seed 20260901), deterministic LayerNorm, and GELU. π0 is never fine-tuned. The GRU input drops the four one-hot coordinates; the 16D task vector enters the static branch. Trainable parameter count differs by only 256 parameters.

## GT physics: one-hot versus semantic

| Budget | OneHot SR | Semantic SR | Δ Semantic | Semantic under-force | Semantic utility |
|---:|---:|---:|---:|---:|---:|
| 0 | 81.94% | 84.03% | +2.08 pp | 14.58% | -0.0153 |
| 10 | 91.90% | 84.49% | -7.41 pp | 14.12% | 0.0227 |
| 20 | 90.97% | 82.41% | -8.56 pp | 16.20% | 0.0035 |
| 30 | 92.36% | 84.26% | -8.10 pp | 14.35% | 0.0108 |
| 60 | 92.36% | 89.81% | -2.55 pp | 8.80% | 0.0746 |

Curve total variation changes from **12.27 pp** (one-hot) to **9.95 pp** (semantic); downward transitions remain 1. It is numerically smoother only because it stays lower through B30 before rising at B60; this is **not** improved data efficiency.

## Physics-source separation

| Budget | GT Semantic SR | SysID Semantic SR | Learned Probe Semantic SR | GT−SysID gap |
|---:|---:|---:|---:|---:|
| 0 | 84.03% | 74.77% | 69.91% | +9.26 pp |
| 10 | 84.49% | 75.69% | 69.91% | +8.80 pp |
| 20 | 82.41% | 78.47% | 70.83% | +3.94 pp |
| 30 | 84.26% | 81.48% | 74.77% | +2.78 pp |
| 60 | 89.81% | 86.81% | 87.04% | +3.01 pp |

The prior estimator audit remains authoritative: Learned Probe LOTO MAE is 0.5149 and Explicit SysID MAE is 0.1810. Controller SR, however, is the headline; lower μ MAE is not automatically equated with better force decisions.

## Per-task adaptation budget

| Physics | task0 B* | task1 B* | task5 B* | task6 B* | Macro B* |
|---|---:|---:|---:|---:|---:|
| GT | 20 | 60 | 20 | 30 | 60 |
| ExplicitSysID | 60 | 60 | 60 | 30 | 60 |
| LearnedProbe | 60 | 60 | 60 | 60 | 60 |

## Why intermediate budgets can remain non-monotonic

The exact previous nested acquisition prefixes were reused byte-for-byte. More rows do not guarantee a monotone neural classifier: each prefix changes target/source class balance, root coverage, and near-boundary examples while the objective is branch BCE and the controller uses an argmax across forces. No warm-start or monotonic constraint exists. The semantic experiment therefore diagnoses whether task identity reduces the instability; it does not reinterpret a B10 spike as data efficiency.
task6 B10/B20 remains a one-class all-success adaptation anomaly: **True**. Those budgets are reported but cannot satisfy B*.

task1 remains in every macro result and retains the 140/180 reconstructed-terminal-label caveat. A selected-branch direct-label sensitivity is provided separately.

## Direct answers to the 20 requested questions

1. **Current Direct task representation:** 4D one-hot repeated at each of H=8 nominal-motion steps.
2. **Selected frozen π0 representation:** masked mean of exact PaliGemma/Gemma input token embeddings for the real neutral instruction.
3. **Dimensions:** 2048D raw; fixed 16D projected controller input.
4. **Frozen?** Yes: checkpoint, tokenizer, pooling, projection, and π0 weights are frozen; no target outcome enters them.
5. **GT B0:** OneHot 81.94%; Semantic 84.03%.
6. **B10/B20 deltas:** -7.41 pp and -8.56 pp.
7. **Smoother?** Numerically yes by total variation (12.27→9.95 pp), but scientifically no improvement: B10–B60 are lower and B* worsens 30→60.
8. **Semantic GT B*:** 60.
9. **SysID Semantic B*:** 60.
10. **Learned Probe Semantic B*:** 60.
11. **Does GT still need B60?** Yes.
12. **If GT needs 60:** that is evidence that full-task force feasibility remains task-specific under this fixed semantic representation, not proof for every possible representation.
13. **If GT needs ≤20 but Learned Probe needs 60:** physics estimation, not Direct semantics, is the dominant transfer bottleneck.
14. **Does SysID close the gap?** At B0/B10/B20 the GT−SysID gaps are +9.26/+8.80/+3.94 pp.
15. **Easiest zero-shot target under GT:** task5 (highest Semantic-GT B0 SR).
16. **Largest GT adaptation requirement:** task1.
17. **Does label composition explain non-monotonicity?** It contributes—especially one-class task6—but cannot alone explain every task/seed reversal; paired branch-BCE/argmax instability remains.
18. **task6 B10/B20 still anomalous?** Yes; all relevant folds are one-class all-success = True.
19. **Does '60 new-task rollouts required' still hold?** Yes under the frozen macro gate.
20. **Recommended pipeline:** Do **not** replace the current known-task Direct with this semantic encoder and do not claim zero/few-shot new-task transfer. Keep the existing shared Direct for the four trained tasks; for a new task, budget substantial task-specific outcomes. Explicit SysID remains preferable by calibrated μ MAE, but its controller must be chosen by frozen downstream metrics; the present Semantic-SysID pipeline also needs B60.

## Scope and limitations

This is archived grouped-root development evidence. Held-out task also changes object family, so task semantics and object-family shift are not separable. GT physics is privileged diagnostic only. No simulator was run, no root-scaling untouched TEST was opened, and no representation or projection was chosen from these results.

Paired GT analysis contains 2160 exact episode pairs. At B0/B10/B20, mean paired utility deltas are +0.0066, -0.0625, -0.0909.
