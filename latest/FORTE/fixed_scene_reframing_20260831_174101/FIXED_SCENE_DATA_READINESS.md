# Fixed-Scene Data Readiness

Decision: **YES — TRAIN NOW**

Historical Joint verdict: `NO_CLEAR_SCENE_GENERALIZATION_PATTERN`.

The authoritative common population contains 720 valid rows, 4 scene families, 72 friction contexts, and 480 TRAIN / 240 DEV labels under a context-level held-out-friction split. Every row is Joint-complete.

## Scene definition

A scene family is task/object plus the frozen P5-S0-C/P4-B trajectory protocol. The six underlying seeds within each task/object family are nested state/seed variation, not six independent scene identities.

## Readiness criteria

- scene families >=2: `True`
- >=10 friction contexts per scene: `True`
- 3–5 N coverage per scene: `True`
- success/failure transition contexts: `True`
- complete Joint inputs: `True`
- same-scene held-out friction split: `True`

The result is **YES — TRAIN NOW**. No 200-root requirement is applied. If training is started, use the exact frozen Direct/Imagination/Joint implementations and keep full-task success as the primary DEV metric; losses and ranking diagnostics remain secondary.

TEST status: `TEST NOT OPENED`.
