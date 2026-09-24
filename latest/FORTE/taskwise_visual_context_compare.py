#!/usr/bin/env python3
"""Assemble the frozen per-task results into one cross-task mechanism audit."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE / "taskwise_visual_context_20260831_074000"
TASK0_TRAIN = HERE / "task0_visual_context_early_20260831_025000"
TASK0_DEV = HERE / "task0_visual_generalization_20260831_040609"
POOLED = Path("/home/exouser/Tabero/analysis/results/gnp_style_visual_context_prospective_20260831_011000")
MODELS = ["PROSPECTIVE_BASE_FEAS", "VISUAL_INTERCEPT_RESIDUAL", "VISUAL_CONTEXT_FULL_FEAS", "VISUAL_CONTEXT_JOINT"]
OBJECTS = {0:"alphabet soup",1:"cream cheese",5:"tomato sauce",6:"butter"}


def mean_row(path: Path, model: str):
    d = pd.read_csv(path)
    return d[(d.fold.astype(str)=="__MEAN__") & (d.model==model)].iloc[0]


def metric_row(path: Path, model: str):
    d = pd.read_csv(path)
    return d[(d.model==model) & (d.aggregation=="ENSEMBLE")].iloc[0]


def frontier_row(path: Path, model: str):
    d = pd.read_csv(path)
    return d[d.model==model].iloc[0]


def rel(a: float, b: float) -> float:
    return (a-b)/a if a else math.nan


def collect(task: int) -> dict:
    if task == 0:
        train_dir, dev_dir = TASK0_TRAIN, TASK0_DEV
        final_path = dev_dir/"TASK0_FINAL_CLASSIFICATION.json"
        cv_path = dev_dir/"TASK0_GROUP_HELDOUT_CV.csv"
        prob_path = dev_dir/"TASK0_DEV_PROBABILITY_METRICS.csv"
        front_path = dev_dir/"TASK0_DEV_FRONTIER_METRICS.csv"
        train_path = train_dir/"TASK0_TRAIN_IN_SAMPLE_METRICS.csv"
    else:
        train_dir, dev_dir = ROOT/f"task{task}_train_frozen", ROOT/f"task{task}_dev_validation"
        final_path = dev_dir/f"TASK{task}_FINAL_CLASSIFICATION.json"
        cv_path = train_dir/f"TASK{task}_GROUP_HELDOUT_CV.csv"
        prob_path = dev_dir/f"TASK{task}_DEV_PROBABILITY_METRICS.csv"
        front_path = dev_dir/f"TASK{task}_DEV_FRONTIER_METRICS.csv"
        train_path = train_dir/f"TASK{task}_TRAIN_IN_SAMPLE_METRICS.csv"
    final = json.loads(final_path.read_text())
    tr = pd.read_csv(train_path)
    train = {m:float(tr[tr.variant==m].bce_nll.mean()) for m in MODELS}
    cv = {m:float(mean_row(cv_path,m).BCE) for m in MODELS}
    prob = {m:metric_row(prob_path,m) for m in MODELS}
    front = {m:frontier_row(front_path,m) for m in MODELS}
    comp = final["comparison_tests"]
    strong_train_visual = rel(train[MODELS[0]],train[MODELS[2]]) >= .20
    cv_visual_support = cv[MODELS[2]] < cv[MODELS[0]]
    return {
        "task":task, "object":OBJECTS[task], "classification":final["classification"],
        "DEV_contexts":int(final.get("DEV_contexts",final.get("primary_DEV_contexts",0))),
        "task1_label_caveat":task==1,
        "TRAIN_Base_BCE":train[MODELS[0]], "TRAIN_Full_BCE":train[MODELS[2]], "TRAIN_Joint_BCE":train[MODELS[3]],
        "TRAIN_Full_relative_gain":rel(train[MODELS[0]],train[MODELS[2]]),
        "CV_Base_BCE":cv[MODELS[0]], "CV_Full_BCE":cv[MODELS[2]], "CV_Joint_BCE":cv[MODELS[3]],
        "CV_Full_relative_gain":rel(cv[MODELS[0]],cv[MODELS[2]]), "CV_Joint_vs_Full_relative_gain":rel(cv[MODELS[2]],cv[MODELS[3]]),
        "DEV_Base_probability_MAE":float(prob[MODELS[0]].probability_MAE),
        "DEV_Full_probability_MAE":float(prob[MODELS[2]].probability_MAE),
        "DEV_Joint_probability_MAE":float(prob[MODELS[3]].probability_MAE),
        "DEV_Base_frontier_MAE_N":float(front[MODELS[0]].frontier_MAE_N),
        "DEV_Full_frontier_MAE_N":float(front[MODELS[2]].frontier_MAE_N),
        "DEV_Joint_frontier_MAE_N":float(front[MODELS[3]].frontier_MAE_N),
        "DEV_Full_under_force_rate":float(front[MODELS[2]].under_force_rate),
        "DEV_Joint_under_force_rate":float(front[MODELS[3]].under_force_rate),
        "strong_TRAIN_visual_fit":strong_train_visual, "CV_visual_support":cv_visual_support,
        "task0_like_visual_memorization":bool(strong_train_visual and not cv_visual_support),
        "heldout_visual_generalization":bool(comp["visual_generalizes"]),
        "heldout_joint_independent_win":bool(comp["joint_independent_win"]),
        "multi_context_claim_permitted":task!=6,
    }


def main() -> None:
    rows = [collect(t) for t in (0,1,5,6)]
    d = pd.DataFrame(rows); d.to_csv(ROOT/"TASKWISE_MECHANISM_COMPARISON.csv",index=False)
    comparable = d[d.task.isin([0,1,5])]
    memo = comparable[comparable.task0_like_visual_memorization].task.astype(int).tolist()
    general = comparable[comparable.heldout_visual_generalization].task.astype(int).tolist()
    joint = comparable[comparable.heldout_joint_independent_win].task.astype(int).tolist()
    pooled_status = "COMPLETE" if (POOLED/"EVALUATION_SUMMARIES.json").exists() else "PENDING_UNCHANGED_MAIN_PIPELINE"
    pooled = json.loads((POOLED/"EVALUATION_SUMMARIES.json").read_text()) if pooled_status=="COMPLETE" else None
    verdict = {
        "status":"COMPLETE_TASK_SPECIFIC_CROSS_TASK_AUDIT", "tasks":[0,1,5,6],
        "multi_context_comparable_tasks":[0,1,5], "task6_limitation":"one DEV context; no task-level multi-context claim",
        "task0_like_visual_memorization_tasks":memo, "heldout_visual_generalization_tasks":general,
        "heldout_joint_independent_win_tasks":joint,
        "same_property_across_multi_context_tasks":len(memo)==3,
        "interpretation":"All multi-context tasks show strong TRAIN visual fit without TRAIN-root-heldout visual support." if len(memo)==3 else "The task0-like property is not uniform across multi-context tasks.",
        "task1_label_caveat":"material; 140 TRAIN labels reconstructed, so task1 is corroborative rather than decisive",
        "task5_role":"direct-label replication controlling the task1 label caveat",
        "pooled_all_task_joint_status":pooled_status, "pooled_summary":pooled,
    }
    (ROOT/"TASKWISE_FINAL_VERDICT.json").write_text(json.dumps(verdict,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    show=d[["task","object","classification","TRAIN_Full_relative_gain","CV_Full_relative_gain","CV_Joint_vs_Full_relative_gain","DEV_Full_probability_MAE","DEV_Joint_probability_MAE","task0_like_visual_memorization","heldout_visual_generalization","heldout_joint_independent_win"]]
    lines=["# Cross-task visual-context mechanism audit","",verdict["interpretation"],"","## Frozen task-specific evidence","",show.to_markdown(index=False),"","## Scope and caveats","","- task1 has a material reconstructed-label caveat; task5 supplies the clean direct-label replication.","- task6 has one DEV context, so it can contribute held-out metrics and TRAIN-root CV but cannot establish multi-context generalization by itself.",f"- Pooled all-task Joint pipeline: {pooled_status}.","- TRAIN fit is never used as the final conclusion."]
    (ROOT/"TASKWISE_FINAL_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    nb={"nbformat":4,"nbformat_minor":5,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}},"cells":[
        {"cell_type":"markdown","metadata":{},"source":["# Cross-task mechanism audit\n",verdict["interpretation"]+"\n"]},
        {"cell_type":"code","execution_count":1,"metadata":{},"source":["import pandas as pd\nprint(pd.read_csv('TASKWISE_MECHANISM_COMPARISON.csv').to_string(index=False))\n"],"outputs":[{"output_type":"stream","name":"stdout","text":[d.to_string(index=False)+"\n"]}]},
        {"cell_type":"code","execution_count":2,"metadata":{},"source":["import json\nprint(json.load(open('TASKWISE_FINAL_VERDICT.json')))\n"],"outputs":[{"output_type":"stream","name":"stdout","text":[json.dumps(verdict,indent=2)+"\n"]}]}
    ]}
    (ROOT/"TASKWISE_ANALYSIS.ipynb").write_text(json.dumps(nb,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(verdict,indent=2),flush=True)


if __name__=="__main__": main()
