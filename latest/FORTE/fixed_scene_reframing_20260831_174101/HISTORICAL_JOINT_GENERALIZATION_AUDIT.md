# Historical Joint Generalization Audit

## Verdict

`C — NO_CLEAR_SCENE_GENERALIZATION_PATTERN`

The archived evidence does not contain a clean same-scene held-out-friction benchmark for the historical Direct/Imagination/Joint comparison. Therefore the proposed hypothesis cannot be classified as A or B without relabeling a different split. The old pooled Joint result is a 72-context, four-task diagnostic; the root-heldout result is cross-root/scene-like rather than same-scene friction holding.

## Evidence kept separate

- Old pooled continuous Joint: probability MAE improves from 0.2449 to 0.2075 and frontier MAE from 0.325 N to 0.125 N, but Brier worsens 0.0859 to 0.1352, NLL worsens 0.4627 to 0.9387, under-force worsens 0.25 to 0.375, and nonmonotonic contexts are 0.6667.
- Retrospective root-heldout Joint-NoVisual: probability MAE improves on all four tasks, while under-force worsens on all four and frontier worsens on tasks 0/1/5. This is a cross-root diagnostic, not same-scene friction generalization.
- Historical Imagination reports selection/frontier diagnostics, not a matched full-task SR comparison; its metrics are not merged with Direct/Joint probability metrics.

## Interpretation

The record supports a narrow historical frontier-localization signal and a failure to establish a reliable controller/generalization advantage. It does not establish “strong same-scene, weak cross-scene” or “overfits even within scene” because same-scene held-out friction was not measured under the same protocol.

The current study therefore proceeds under the explicitly narrowed question: `FIXED-SCENE / LOW-SCENE-DIVERSITY HIDDEN-FRICTION FORCE SELECTION`. The current 720-row common set is audited separately below; historical verdict C does not impose the obsolete 200-independent-root requirement.
