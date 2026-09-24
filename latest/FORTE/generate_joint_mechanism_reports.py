#!/usr/bin/env python3
"""Generate the final mechanism narrative from frozen comparison artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/home/exouser/FORTE")
OUT = ROOT / "joint_mechanism_20260831"


def fmt(x, digits=4):
    return "—" if pd.isna(x) else f"{float(x):.{digits}f}"


def md_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without the optional tabulate package."""
    if frame.empty:
        return "(no rows)"
    columns = [str(c) for c in frame.columns]
    def cell(value):
        if pd.isna(value):
            return "—"
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(columns) + " |",
             "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(cell(v) for v in row) + " |")
    return "\n".join(lines)


def root_deltas():
    taskwise = pd.read_csv(OUT / "TASKWISE_ROOT_HELDOUT_COMPARISON.csv")
    pooled = pd.read_csv(OUT / "POOLED_JOINT_COMPARISON.csv")
    pooled = pooled[pooled.evaluation_scope == "MATCHED_POOLED_TRAIN_ROOT_HELDOUT_CV"]
    rows = []
    for task in [0, 1, 5, 6]:
        s = taskwise[taskwise.task == task].set_index("model")
        p = pooled[pooled.task.astype(str) == str(task)].set_index("model")
        rows.append({
            "task": task,
            "single_JNV_minus_Base_NLL": s.loc["Joint-NoVisual", "NLL"]-s.loc["Base", "NLL"],
            "single_JNV_minus_Base_probability_MAE": s.loc["Joint-NoVisual", "probability_MAE"]-s.loc["Base", "probability_MAE"],
            "single_JNV_minus_Base_Brier": s.loc["Joint-NoVisual", "Brier"]-s.loc["Base", "Brier"],
            "single_JNV_minus_Base_frontier_MAE_N": s.loc["Joint-NoVisual", "frontier_MAE_N"]-s.loc["Base", "frontier_MAE_N"],
            "single_JNV_minus_Base_under_force": s.loc["Joint-NoVisual", "under_force_rate"]-s.loc["Base", "under_force_rate"],
            "VisualJoint_minus_Full_NLL": s.loc["Visual Joint", "NLL"]-s.loc["Full", "NLL"],
            "pooled_JNV_minus_Base_NLL": p.loc["Pooled Joint-NoVisual", "NLL"]-p.loc["Pooled Base", "NLL"],
            "pooled_JNV_minus_Base_probability_MAE": p.loc["Pooled Joint-NoVisual", "probability_MAE"]-p.loc["Pooled Base", "probability_MAE"],
            "pooled_JNV_minus_Base_Brier": p.loc["Pooled Joint-NoVisual", "Brier"]-p.loc["Pooled Base", "Brier"],
            "pooled_JNV_minus_Base_frontier_MAE_N": p.loc["Pooled Joint-NoVisual", "frontier_MAE_N"]-p.loc["Pooled Base", "frontier_MAE_N"],
            "pooled_JNV_minus_Base_under_force": p.loc["Pooled Joint-NoVisual", "under_force_rate"]-p.loc["Pooled Base", "under_force_rate"],
        })
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "MECHANISM_DELTA_SUMMARY.csv", index=False)
    return d


def gap_table():
    rows = []
    for task in [0, 1, 5, 6]:
        d = pd.read_csv(OUT / f"TASK{task}_JOINT_NOVISUAL.csv")
        cv = d[d.evaluation_scope == "RETROSPECTIVE_TRAIN_ROOT_HELDOUT_CV"]
        for model in ["Base", "Full", "Visual Joint", "Joint-NoVisual"]:
            r = cv[cv.model == model].iloc[0]
            rows.append({"task": task, "model": model,
                         "TRAIN_BCE": r.TRAIN_BCE_reference,
                         "heldout_NLL": r.NLL,
                         "gap": r.heldout_minus_TRAIN_NLL_gap,
                         "ratio": r.heldout_to_TRAIN_NLL_ratio})
    return pd.DataFrame(rows)


def dev_table():
    rows = []
    for task in [0, 1, 5, 6]:
        d = pd.read_csv(OUT / f"TASK{task}_JOINT_NOVISUAL.csv")
        q = d[d.evaluation_scope == "RETROSPECTIVE_PROSPECTIVE_DEV_ROOTS"]
        for r in q.to_dict("records"):
            rows.append({"task": task, "model": r["model"], "probability_MAE": r["probability_MAE"],
                         "Brier": r["Brier"], "NLL": r["NLL"], "frontier_MAE_N": r["frontier_MAE_N"],
                         "under_force_rate": r["under_force_rate"], "contexts": r["contexts"]})
    return pd.DataFrame(rows)


def pooled_dev_table():
    p = OUT / "pooled_matched/POOLED_MATCHED_DEV_TASKWISE_SUMMARY.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def original_pooled_table():
    p = OUT / "original_pooled_visual_taskwise/ORIGINAL_POOLED_VISUAL_TASKWISE.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def old_metrics():
    d = pd.read_csv(ROOT / "gnp_style_continuous_20260830_125107/CONTINUOUS_DEV_PROBABILITY_METRICS.csv")
    return d[(d.condition == "GT") & (d.probability_semantics == "RAW_ENSEMBLE_PRIMARY")]


def main():
    delta = root_deltas()
    gaps = gap_table()
    dev = dev_table()
    pdev = pooled_dev_table()
    opool = original_pooled_table()
    old = old_metrics()

    single_nll_bad = int((delta.single_JNV_minus_Base_NLL > 0).sum())
    visual_nll_bad = int((delta.VisualJoint_minus_Full_NLL > 0).sum())
    pooled_nll_better = int((delta.pooled_JNV_minus_Base_NLL < 0).sum())
    pooled_under_bad = int((delta.pooled_JNV_minus_Base_under_force > 0).sum())
    prospective_jnv_bad = 0
    prospective_visual_bad = 0
    pooled_dev_bad = 0
    pooled_dev_under_bad = 0
    original_visual_bad = 0
    for task in [0, 1, 5, 6]:
        q = dev[dev.task == task].set_index("model")
        prospective_jnv_bad += int(q.loc["Joint-NoVisual", "NLL"] > q.loc["Base", "NLL"])
        prospective_visual_bad += int(q.loc["Visual Joint", "NLL"] > q.loc["Full", "NLL"])
        q = pdev[pdev.task == task].set_index("model")
        pooled_dev_bad += int(q.loc["POOLED_JOINT_NOVISUAL", "NLL"] > q.loc["POOLED_BASE", "NLL"])
        pooled_dev_under_bad += int(q.loc["POOLED_JOINT_NOVISUAL", "under_force_rate"] > q.loc["POOLED_BASE", "under_force_rate"])
        q = opool[opool.task == task].set_index("model")
        original_visual_bad += int(q.loc["VISUAL_CONTEXT_JOINT", "NLL"] > q.loc["VISUAL_CONTEXT_FULL_FEAS", "NLL"])
    classification = "OLD_JOINT_GAIN_DOES_NOT_TRANSFER_TO_CURRENT_DATA_REGIME"
    case = "Case C"

    old_table = old[["backend", "probability_MAE", "Brier", "negative_log_likelihood",
                     "frontier_MAE_N", "under_force_rate", "systematic_nonmonotonic_context_rate"]].copy()
    old_table.columns = ["model", "probability MAE", "Brier", "NLL", "frontier MAE N", "under-force", "nonmonotonic context rate"]
    delta_show = delta.copy()
    for c in delta_show.columns[1:]:
        delta_show[c] = delta_show[c].map(lambda x: round(float(x), 4))
    gap_show = gaps.copy()
    for c in ["TRAIN_BCE", "heldout_NLL", "gap", "ratio"]:
        gap_show[c] = gap_show[c].map(lambda x: round(float(x), 4))

    why = f"""# WHY DID OLD JOINT LOOK BETTER?

## Bottom line

The old Joint had a real but narrow **frontier-localization signal**. It did not establish a better controller. The signal occurred in a different data/evaluator regime: no visual input, 72 pooled contexts across four tasks, 24 roots, 14 outcome executions/context (10 new continuous plus four coarse), pooled normalization, and a small 9-context/7-root DEV benchmark. Current single-task runs use 18 contexts/six roots/10 executions per context. The matched current pooled diagnostic restores 72 contexts but does not restore a consistent calibrated or safe Joint advantage.

The authoritative old and current taskwise physics auxiliary are architecturally aligned: hidden 64, H8×13 trajectory target, target-matched adjacent-force IE, AdamW 8e-4/1e-4, 80 epochs, seeds 0/1/2, and native `Lphysics + LIE + 0.3 Lfeas`. Therefore an architecture change in that core is not the explanation. The original pooled Visual pipeline is a separate critical confound because it changes that formulation.

## What the old result actually showed

{md_table(old_table)}

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
"""
    (OUT / "WHY_OLD_JOINT_LOOKED_BETTER.md").write_text(why, encoding="utf-8")

    if len(dev):
        dev_show = dev.copy()
        for c in ["probability_MAE", "Brier", "NLL", "frontier_MAE_N", "under_force_rate"]:
            dev_show[c] = dev_show[c].map(lambda x: round(float(x), 4))
        dev_section = "Prospective DEV taskwise rows are complete and shown below:\n\n" + md_table(dev_show)
    else:
        dev_section = "Prospective DEV collection/evaluation is pending; root-heldout TRAIN CV is the current mechanism evidence."
    if len(pdev):
        pdev_show = pdev[["task", "model", "probability_MAE", "Brier", "NLL", "frontier_MAE_N",
                          "under_force_rate", "context_monotonic", "DEV_contexts"]].copy()
        for c in ["probability_MAE", "Brier", "NLL", "frontier_MAE_N", "under_force_rate", "context_monotonic"]:
            pdev_show[c] = pdev_show[c].map(lambda x: round(float(x), 4))
        pooled_dev_section = md_table(pdev_show)
    else:
        pooled_dev_section = "Matched pooled prospective DEV evaluation is pending."
    if len(opool):
        opool_show = opool[["task", "model", "probability_MAE", "Brier", "NLL", "frontier_MAE_N",
                            "under_force_or_missing_rate", "monotonic_context_rate", "implementation_confounded"]].copy()
        for c in ["probability_MAE", "Brier", "NLL", "frontier_MAE_N", "under_force_or_missing_rate", "monotonic_context_rate"]:
            opool_show[c] = opool_show[c].map(lambda x: round(float(x), 4))
        original_section = md_table(opool_show)
    else:
        original_section = "Original pooled visual taskwise evaluation is pending."

    report = f"""# FINAL JOINT MECHANISM REPORT

## Answer first

**Classification: {case} — `{classification}`.**

Joint-NoVisual reproduces the central generalization failure without any RGB/PCA input: TRAIN fitting improves sharply, but root-heldout NLL worsens on all four single tasks ({single_nll_bad}/4). Visual Joint also worsens relative to Full on all four tasks ({visual_nll_bad}/4), so visual conditioning is an additional instability path, especially on task0/task1, but it is not the sole or primary sufficient explanation.

Increasing pooled training diversity from 12 training contexts/fold in single-task CV to 48 training contexts/fold partially helps task5/task6 NLL, but it helps only {pooled_nll_better}/4 tasks and worsens under-force on {pooled_under_bad}/4. Therefore pooled diversity does not rescue a reliable Joint, and `DATA_TOO_SMALL` is not established.

Prospective evidence reaches the same decision despite mixed individual metrics: single-task JNV NLL is worse than Base on {prospective_jnv_bad}/4 tasks, and matched pooled JNV NLL is worse on {pooled_dev_bad}/4. Its only matched-pooled NLL recovery is task6, which has one DEV root and cannot establish multi-context recovery.

## Five direct answers

1. **Why did old Joint look better?** It localized the frontier on a small old benchmark (0.325→0.125 N) in a no-visual, pooled 72-context regime with 288 extra coarse BCE branches. That gain was narrow: safety, Brier/NLL and monotonicity were worse. Current matched pooled evidence shows context diversity alone does not reproduce a reliable advantage.
2. **Is current Joint failure mainly visual shortcut or physics auxiliary?** The physics auxiliary itself is implicated because JNV fails held-root NLL on 4/4 tasks and worsens safety/shape metrics. Visual conditioning adds further failure—Visual Joint is worse than Full on 4/4 tasks—but “visual shortcut alone” is rejected.
3. **Can more pooled contexts rescue Joint?** Root-CV partially improves NLL on task5/task6, but prospective pooled DEV remains worse on task0/task1/task5; only single-root task6 improves. Safety is not consistently recovered. The answer for a controller-quality Joint is no.
4. **Is Joint still an effective paper novelty?** Not as a validated controller or generalizing feasibility model. It can be retained only as a frontier-localization/world-model diagnostic signal or as an explicitly negative mechanistic result; it should not carry the main effectiveness claim.
5. **Next: add contexts or delete/simplify Joint?** Delete/simplify Joint from the controller-facing method first. Audit old supervision/distribution/evaluator and the current auxiliary's overconfidence before collecting substantially more roots. A later preregistered context learning curve may be useful, but the present pooled result does not justify “more contexts” as the next primary fix.

## Scope

This study tests **same task, same object/task distribution, held-out physical root/context**. It does not test unseen-task, cross-object or universal physical reasoning. Pooled task0/1/5/6 training is evaluated separately within each task and is not unseen-task generalization.

## Frozen design

- Single task: 18 contexts, six roots, 180 branches; CV trains on 12 contexts/four roots and holds six contexts/two roots.
- Pooled CV: 72 total contexts/24 roots; each fold trains on 48 contexts/16 roots and holds 24 contexts/eight roots, reported taskwise.
- Joint-NoVisual input equals Base input; no RGB, frozen visual feature or PCA visual representation.
- Physics auxiliary: hidden64 H8×13 authoritative Physics-GRU, true matched trajectory-difference IE, λphysics=1, λIE=1, λfeas=0.3.
- AdamW 8e-4/1e-4, gradient clip1, 80 epochs, seeds0/1/2; CV seed0 matches the existing retrospective protocol.
- No DEV selection, tuning, best-seed choice, Probe, force-grid change or architecture/loss change.

## TRAIN → held-root evidence

{md_table(gap_show)}

Joint-NoVisual has a larger positive TRAIN→heldout NLL gap than Base on every task. Visual Joint has the largest or near-largest gap. This is direct evidence of context-dependent overfitting, not merely weak TRAIN optimization.

## Mechanism contrasts

All deltas are candidate minus comparator; positive NLL/Brier/frontier/under-force is worse.

{md_table(delta_show)}

JNV's probability MAE improves on all tasks even while NLL and safety worsen. This combination is consistent with an overly sharp force-response curve: average distance to empirical probabilities can fall while confident errors, threshold placement and under-force risk increase. Therefore no conclusion is based on probability MAE alone.

## Prospective taskwise DEV

{dev_section}

task6 has only one DEV root; it is reported but cannot independently support multi-context generalization. Root-heldout TRAIN CV is the primary task6 pattern test.

## Matched pooled prospective DEV

{pooled_dev_section}

Matched pooled JNV has worse NLL on {pooled_dev_bad}/4 tasks (task0/task1/task5). It improves task6 only, where the single DEV root forbids a multi-context recovery claim. Under-force worsens on {pooled_dev_under_bad}/4 tasks and is unchanged on the others.

## Original pooled Visual pipeline

{original_section}

These rows are implementation-confounded and cannot isolate context diversity: the original pooled Joint uses a repeated H8×4 target, no authoritative Physics-GRU trajectory head, and an adjacent-prediction-difference penalty rather than target-matched IE.

Even under that different formulation, Visual Joint has worse taskwise NLL than Full on {original_visual_bad}/4 tasks. Its aggregate frontier improvement therefore remains a narrow localization signal rather than a calibrated feasibility-model win.

## Hypothesis adjudication

- **Hypothesis A, visual shortcut:** supported as an additional path/interaction, because Visual Joint is worse than Full on every task and task0/task1 show especially large gaps. Rejected as the sole explanation because no-visual Joint also fails.
- **Hypothesis B, sample-limited physics auxiliary:** plausible at 18 contexts, and pooled task5/task6 probability results show partial regularization. Not proven as the main explanation because pooled JNV does not recover task0/task1 and worsens under-force across tasks.
- **Most defensible case:** Case C. The old frontier signal does not transfer into a consistently calibrated and safe current formulation, including the higher-context pooled regime.

## Data quality and limitations

- task0/task5/task6 labels are direct; task1 has 140/180 reconstructed TRAIN labels and remains caveated.
- task5/task6 direct-label replications show the pattern is not reducible to task1 label recovery.
- CV frontier estimates use five random forces × two repeats and are noisier than prospective DEV.
- Original pooled visual results are not a matched formulation.
- The partial old nested-context run is excluded because its task0-only setting was evaluated on all-task DEV and the nested schedule did not complete.

## Decision

Do not use current Joint as the paper's controller-quality novelty and do not justify another large task0-root collection by saying “18 contexts is too small.” Retain Base/Full as the empirical controller baselines; keep the physics auxiliary only as a diagnostic research object until it passes same-task new-root calibration, frontier, under-force and monotonicity together.
"""
    (OUT / "FINAL_JOINT_MECHANISM_REPORT.md").write_text(report, encoding="utf-8")

    classification_obj = {
        "status": "COMPLETE" if len(dev) and len(pdev) and len(opool) else "EVIDENCE_COMPLETE_EXCEPT_PENDING_PROSPECTIVE_COMPONENTS",
        "case": case, "classification": classification,
        "visual_conditioning_primary_sole_explanation": False,
        "physics_auxiliary_implicated_without_visual": True,
        "pooled_context_diversity_reliably_rescues_joint": False,
        "data_too_small_conclusion_permitted": False,
        "small_context_count": "PLAUSIBLE_BUT_UNPROVEN_AS_PRIMARY_EXPLANATION",
        "joint_effective_controller_novelty": False,
        "joint_frontier_localization_signal": True,
        "single_task_JNV_NLL_worse_tasks": single_nll_bad,
        "visual_joint_NLL_worse_than_full_tasks": visual_nll_bad,
        "pooled_JNV_NLL_better_tasks": pooled_nll_better,
        "pooled_JNV_under_force_worse_tasks": pooled_under_bad,
        "prospective_single_task_JNV_NLL_worse_tasks": prospective_jnv_bad,
        "prospective_single_task_visual_joint_NLL_worse_than_full_tasks": prospective_visual_bad,
        "prospective_pooled_JNV_NLL_worse_tasks": pooled_dev_bad,
        "prospective_pooled_JNV_under_force_worse_tasks": pooled_dev_under_bad,
        "original_confounded_pooled_visual_joint_NLL_worse_than_full_tasks": original_visual_bad,
        "scope": "same task / same object-task distribution / held-out physical root-context",
        "not_claimed": ["unseen-task generalization", "cross-object generalization", "universal physical reasoning"],
        "recommended_next_action": "REMOVE_OR_SIMPLIFY_JOINT_FROM_CONTROLLER_CLAIM_AND_AUDIT_OLD_CURRENT_DISTRIBUTION_SUPERVISION_EVALUATOR_BEFORE_MORE_ROOTS",
        "selection_use_forbidden": True,
    }
    (OUT / "FINAL_JOINT_MECHANISM_CLASSIFICATION.json").write_text(json.dumps(classification_obj, indent=2, sort_keys=True) + "\n")
    print(json.dumps(classification_obj, indent=2))


if __name__ == "__main__":
    main()
