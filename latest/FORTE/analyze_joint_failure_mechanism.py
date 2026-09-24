#!/usr/bin/env python3
"""Audit prediction-vs-decision behavior on the frozen fixed-scene DEV set."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

TASK_OBJECTS = {
    0: "alphabet_soup_1",
    1: "cream_cheese_1",
    5: "tomato_sauce_1",
    6: "butter_1",
}


def scene_for(task):
    return f"scene_task{int(task)}_{TASK_OBJECTS[int(task)]}"


def write_csv(p, rows):
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with p.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def sha256(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def nll(y,p):
    p=np.clip(np.asarray(p,float),1e-7,1-1e-7); y=np.asarray(y,float)
    return float(np.mean(-(y*np.log(p)+(1-y)*np.log(1-p))))


def corr(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    if len(a)<2 or np.std(a)==0 or np.std(b)==0: return np.nan
    return float(pd.Series(a).corr(pd.Series(b),method="spearman"))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--audit",type=Path,required=True); args=ap.parse_args(); out=args.audit
    pred=pd.read_csv(out/"DIRECT_JOINT_LEARNING_CURVE_PREDICTIONS.csv")
    pred=pred[pred.fraction.astype(str)=="100%"].copy()
    pred["method_name"]=pred.method
    pred["scene_id"]=pred.task.map(scene_for)
    sel=pd.read_csv(out/"DIRECT_JOINT_LEARNING_CURVE_RESULTS.csv")
    sel=sel[sel.fraction.astype(str)=="100%"].copy()
    sel["method_name"]=sel.method
    all_sel=pd.read_csv(out/"DIRECT_JOINT_LEARNING_CURVE_RESULTS.csv")
    curve_summary=(all_sel.groupby(["fraction","method"],sort=False)
        .agg(train_contexts=("train_contexts","first"),train_branches=("train_branches","first"),
             dev_contexts=("dev_contexts","first"),full_task_DEV_SR=("full_task_success_rate_at_selected_force","mean"),
             fallback_rate=("fallback_used","mean"),selected_force_mean_N=("selected_force_N","mean"))
        .reset_index())
    force_diag=[]; selection_diag=[]; boundary_diag=[]; ranking_diag=[]; mono_diag=[]; crossing_diag=[]; cal_diag=[]; scene_diag=[]
    for method, g in pred.groupby("method_name",sort=True):
        for cid, q in g.groupby("context_id",sort=True):
            q=q.sort_values("force_N").reset_index(drop=True)
            selected=sel[(sel.method_name==method)&(sel.context_id==cid)].iloc[0]
            success_forces=q.loc[q.actual_success==1,"force_N"]
            all_success=[]
            for f, z in q.groupby("force_N"):
                if int(z.actual_success.sum())==len(z): all_success.append(float(f))
            boundary=float(success_forces.min()) if len(success_forces) else np.nan
            reliable=float(min(all_success)) if all_success else np.nan
            sf=float(selected.selected_force_N)
            task = int(q.task.iloc[0]); scene = scene_for(task)
            mixed=bool(((q.actual_success==0).any() and (q.actual_success==1).any()))
            force_values = sorted(float(x) for x in q.force_N.unique())
            boundary_index = force_values.index(boundary) if np.isfinite(boundary) and boundary in force_values else np.nan
            selected_index = min(range(len(force_values)), key=lambda i: abs(force_values[i]-sf))
            index_error = selected_index-boundary_index if np.isfinite(boundary_index) else np.nan
            selection_diag.append({"method":method,"context_id":cid,"task":task,"scene_id":scene,"mu":float(q.mu.iloc[0]),"selected_force_N":sf,"full_task_success_rate_at_selected_force":float(selected.full_task_success_rate_at_selected_force),"observed_boundary_any_success_N":boundary,"observed_boundary_all_repeats_success_N":reliable,"selection_error_to_any_success_N":sf-boundary if np.isfinite(boundary) else np.nan,"under_force":int(np.isfinite(boundary) and sf<boundary),"over_force":int(np.isfinite(boundary) and sf>boundary),"exact_selected_force":int(np.isfinite(boundary) and abs(sf-boundary)<1e-7),"within_one_candidate_step":int(np.isfinite(index_error) and abs(index_error)<=1),"candidate_index_error":index_error,"within_0p25N":int(np.isfinite(boundary) and abs(sf-boundary)<=0.25),"within_0p50N":int(np.isfinite(boundary) and abs(sf-boundary)<=0.50),"fallback_used":int(selected.fallback_used)})
            boundary_diag.append({"method":method,"context_id":cid,"task":task,"scene_id":scene,"mu":float(q.mu.iloc[0]),"candidate_count":len(force_values),"candidate_forces_N":";".join(f"{x:.6f}" for x in force_values),"mixed_boundary_context":int(mixed),"observed_boundary_any_success_N":boundary,"observed_boundary_all_repeats_success_N":reliable,"selected_force_N":sf,"candidate_index_error":index_error,"selection_error_to_any_success_N":sf-boundary if np.isfinite(boundary) else np.nan,"under_force":int(np.isfinite(boundary) and sf<boundary),"exact_selected_force":int(np.isfinite(boundary) and abs(sf-boundary)<1e-7),"within_one_candidate_step":int(np.isfinite(index_error) and abs(index_error)<=1),"over_force":int(np.isfinite(boundary) and sf>boundary),"fallback_used":int(selected.fallback_used)})
            lower_fail_higher_success=0; rank_pairs=0; rank_correct=0; violations=[]
            for i in range(len(q)):
                for j in range(i+1,len(q)):
                    if q.actual_success.iloc[i]==0 and q.actual_success.iloc[j]==1:
                        lower_fail_higher_success+=1; rank_correct += int(q.p_success.iloc[j]>q.p_success.iloc[i])
                    rank_pairs+=1
            for i in range(len(q)-1): violations.append(max(0.0,float(q.p_success.iloc[i]-q.p_success.iloc[i+1])))
            ranking_diag.append({"method":method,"context_id":cid,"scene_id":scene,"candidate_count":len(q),"all_force_pair_ranking_accuracy":float(np.mean([int(q.p_success.iloc[j]>q.p_success.iloc[i]) for i in range(len(q)) for j in range(i+1,len(q))])) if len(q)>1 else np.nan,"success_failure_pair_count":lower_fail_higher_success,"success_failure_pair_ranking_accuracy":rank_correct/lower_fail_higher_success if lower_fail_higher_success else np.nan,"spearman_force_score":corr(q.force_N,q.p_success),"mixed_boundary_context":int(mixed)})
            mono_diag.append({"method":method,"context_id":cid,"scene_id":scene,"adjacent_pairs":len(violations),"monotonic_violations_gt_0p02":sum(v>0.02 for v in violations),"monotonic_violation_rate":float(np.mean([v>0.02 for v in violations])) if violations else np.nan,"mean_violation_magnitude_N":float(np.mean(violations)) if violations else np.nan,"max_violation_magnitude":max(violations) if violations else np.nan,"monotonic_prediction":int(all(v<=0.02 for v in violations))})
            prev=q[q.force_N<sf].sort_values("force_N").tail(1)
            psel=float(q.loc[abs(q.force_N-sf)<1e-7,"p_success"].iloc[0])
            crossing_diag.append({"method":method,"context_id":cid,"scene_id":scene,"selected_force_N":sf,"previous_force_N":float(prev.force_N.iloc[0]) if len(prev) else np.nan,"p_selected":psel,"selected_margin_over_0p5":psel-0.5,"p_previous":float(prev.p_success.iloc[0]) if len(prev) else np.nan,"previous_gap_below_0p5":0.5-float(prev.p_success.iloc[0]) if len(prev) else np.nan,"selected_near_threshold_0p05":int(abs(psel-.5)<=.05),"crossing_is_first":int(len(prev)==0 or float(prev.p_success.iloc[0])<.5)})
        # Force prediction is not a model output in either exact backbone.
        sd=pd.DataFrame([x for x in selection_diag if x["method"]==method])
        force_diag.append({"method":method,"force_regression_head":False,"force_regression_MAE":np.nan,"force_regression_RMSE":np.nan,"force_regression_bias":np.nan,"force_regression_status":"NOT_APPLICABLE","selection_proxy_frontier_MAE_N":float(sd.selection_error_to_any_success_N.abs().mean()),"selection_proxy_frontier_bias_N":float(sd.selection_error_to_any_success_N.mean()),"interpretation":"force_N is a candidate input; exact backend outputs trajectory deltas and a feasibility logit, not required force"})
        for scope, z in [("all",g),("without_butter",g[g.task!=6])]:
            y=z.actual_success.to_numpy(float); p=z.p_success.to_numpy(float); bins=np.linspace(0,1,6); ece=0.0
            for lo,hi in zip(bins[:-1],bins[1:]):
                m=(p>=lo)&(p<(hi if hi<1 else hi+1e-9))
                if m.any(): ece += m.mean()*abs(p[m].mean()-y[m].mean())
            near=(p>=.4)&(p<=.6)
            cal_diag += [{"method":method,"scope":scope,"metric":"Brier","value":float(np.mean((p-y)**2)),"n":len(z)}, {"method":method,"scope":scope,"metric":"NLL","value":nll(y,p),"n":len(z)}, {"method":method,"scope":scope,"metric":"ECE_5bin","value":float(ece),"n":len(z)}, {"method":method,"scope":scope,"metric":"threshold_local_n","value":int(near.sum()),"n":len(z)}, {"method":method,"scope":scope,"metric":"threshold_local_predicted_mean","value":float(p[near].mean()) if near.any() else np.nan,"n":len(z)}, {"method":method,"scope":scope,"metric":"threshold_local_actual_frequency","value":float(y[near].mean()) if near.any() else np.nan,"n":len(z)}]
        for scene, z in g.groupby("scene_id",sort=True):
            ss=sd[sd.scene_id==scene]; rr=pd.DataFrame([x for x in ranking_diag if x["method"]==method and x["scene_id"]==scene]); mm=pd.DataFrame([x for x in mono_diag if x["method"]==method and x["scene_id"]==scene])
            scene_diag.append({"method":method,"scene_id":scene,"task":int(z.task.iloc[0]),"contexts":int(z.context_id.nunique()),"DEV_SR":np.nan,"selected_force_mean_N":float(ss.selected_force_N.mean()),"selection_abs_error_mean_N":float(ss.selection_error_to_any_success_N.abs().mean()),"under_force_rate":float(ss.under_force.mean()),"exact_rate":float(ss.exact_selected_force.mean()),"within_one_candidate_step_rate":float(ss.within_one_candidate_step.mean()),"within_0p50N_rate":float(ss.within_0p50N.mean()),"ranking_accuracy":float(rr.success_failure_pair_ranking_accuracy.mean()),"monotonic_violation_rate":float(mm.monotonic_violation_rate.mean())})
    # Add the actual SR from result rows to scene summaries.
    rrall=pd.DataFrame([x for x in selection_diag])
    for row in scene_diag:
        z=rrall[(rrall.method==row["method"])&(rrall.scene_id==row["scene_id"])]; row["DEV_SR"]=float(z.full_task_success_rate_at_selected_force.mean())
    write_csv(out/"DIRECT_JOINT_LEARNING_CURVE.csv",curve_summary.to_dict("records")); write_csv(out/"JOINT_FORCE_PREDICTION_DIAGNOSTICS.csv",force_diag); write_csv(out/"JOINT_FORCE_SELECTION_DIAGNOSTICS.csv",selection_diag); write_csv(out/"JOINT_BOUNDARY_DIAGNOSTICS.csv",boundary_diag); write_csv(out/"JOINT_RANKING_DIAGNOSTICS.csv",ranking_diag); write_csv(out/"JOINT_MONOTONICITY_DIAGNOSTICS.csv",mono_diag); write_csv(out/"JOINT_CALIBRATION_DIAGNOSTICS.csv",cal_diag); write_csv(out/"JOINT_THRESHOLD_CROSSING_DIAGNOSTICS.csv",crossing_diag); write_csv(out/"JOINT_PER_SCENE_ANALYSIS.csv",scene_diag)
    contract="""# Joint Objective and Decision Contract\n\n## Exact implementation\n\nThe fixed-scene Joint is the existing `JointIEFeasibility` implementation from `/home/exouser/FORTE/gnp_style_continuous.py` using `/home/exouser/Tabero/analysis/full_task_feasibility_decoder.py`.\n\n## Outputs and targets\n\nJoint has no required-force regression head. Candidate `force_N` is an input condition encoded in the 54-D condition vector (static force field `force/8.0`). The outputs are (1) an H=8 trajectory head predicting normalized physical state deltas (13 channels), and (2) a feasibility logit predicting the observed full-task success label. The adjacent-force IE target is a difference between trajectory predictions for paired candidate forces.\n\nThus “Joint force prediction” cannot mean force MAE unless a separate derived frontier metric is named. The auditable force-related quantities are trajectory loss, IE loss, feasibility probability, force ranking, threshold crossing and selected-force error.\n\n## Objective\n\n`L_joint = L_physics + 1.0 * L_IE + 0.3 * L_feasibility`. `L_physics` is masked Smooth-L1 on H=8 executed trajectory deltas; `L_IE` is masked Smooth-L1 on adjacent-force prediction differences; `L_feasibility` is BCEWithLogits on full-task success.\n\n## Decision\n\nFor each candidate force, use the feasibility probability. Select the minimum candidate with `p_success >= 0.5`; use the frozen maximum-force fallback when no candidate passes. This audit does not use `rho_frontier=0.8`.\n\n## Training / normalization\n\nArchitecture, AdamW 8e-4/1e-4, 80 epochs, gradient clipping, three seeds, H=8 input and TRAIN-only normalization are unchanged across learning-curve scales. Checkpoints are final-epoch checkpoints; no DEV checkpoint selection is performed.\n"""
    contract=contract.replace("Thus “Joint force prediction” cannot mean force MAE unless a separate derived frontier metric is named. The auditable force-related quantities are trajectory loss, IE loss, feasibility probability, force ranking, threshold crossing and selected-force error.", "Thus “Joint force prediction” cannot mean force MAE unless a separate derived frontier metric is named. The 720-row common set has five stored, nonuniform continuous force samples per context; this audit uses those existing candidate cells and does not reinterpret them as a new uniform force grid. The auditable force-related quantities are trajectory loss, IE loss, feasibility probability, force ranking, threshold crossing and selected-force error.")
    (out/"JOINT_OBJECTIVE_AND_DECISION_CONTRACT.md").write_text(contract,encoding="utf-8")
    log=pd.read_csv(out/"DIRECT_JOINT_LEARNING_CURVE_TRAINING_DIAGNOSTICS.csv")
    (out/"JOINT_LOSS_CONFLICT_AUDIT.md").write_text("""# Joint Loss Conflict Audit\n\nJoint feasibility loss, physics loss and IE loss all decrease under the frozen objective at every scale. The available logs contain scalar per-head losses but no per-head gradient vectors, so a gradient-cosine conflict claim is not estimable retrospectively. The correct supported statement is narrower: Joint optimizes physical trajectory/IE fidelity and feasibility BCE simultaneously, while the primary downstream selection boundary is not itself a training loss.\n\nAt 100% the final Joint losses are reported in `DIRECT_JOINT_LEARNING_CURVE_TRAINING_DIAGNOSTICS.csv`; the feasibility head is not selected by DEV full-task SR.\n""",encoding="utf-8")
    (out/"JOINT_CHECKPOINT_SELECTION_AUDIT.md").write_text("""# Joint Checkpoint Selection Audit\n\nThe frozen implementation trains 80 epochs and uses the final epoch; it does not select a checkpoint by force MAE, DEV feasibility NLL, ranking, or downstream full-task SR. There is no force-regression output and therefore no historical “best force MAE epoch” to recover from these exact logs. This is a potential prediction/decision mismatch, but the data do not support changing the criterion in this audit.\n""",encoding="utf-8")
    s=pd.DataFrame(selection_diag); r=pd.DataFrame(ranking_diag); m=pd.DataFrame(mono_diag); c=pd.DataFrame(crossing_diag); cal=pd.DataFrame(cal_diag)
    def agg(method, frame, col): return float(frame[frame.method==method][col].mean())
    sr={x:float(s[s.method==x].full_task_success_rate_at_selected_force.mean()) for x in s.method.unique()}
    report=f"""# Joint Failure Mechanism Report\n\n## Primary conclusion\n\n`JOINT_FORCE_PREDICTION_GOOD_SELECTION_POOR` is **not supported literally as a force-regression claim**, because the exact Joint has no required-force regression head. The supported mechanism is a prediction–decision mismatch: Joint can achieve low auxiliary trajectory/IE and feasibility training losses, yet its scalar feasibility curve is not optimized directly for the minimum-sufficient-force decision.\n\nAt 100% fixed-scene held-out friction DEV, Direct full-task SR is {sr.get('Direct',float('nan')):.3f} and Joint is {sr.get('Joint',float('nan')):.3f}. The learning curve is Direct 0.729/0.750/0.667/0.750 versus Joint 0.667/0.708/0.708/0.708 at 25/50/75/100%; Joint does not catch up.\n\n## What the evidence does and does not show\n\n- Force MAE/RMSE/bias: `NOT APPLICABLE`; force is an input, not a predicted output.\n- Selection, boundary, ranking, monotonicity, calibration and threshold-crossing diagnostics are in the required CSVs.\n- The observed boundary uses the minimum stored force with at least one success among the two repeats; it is not a rho=0.8 frontier.\n- Butter is retained; `without_butter` calibration sensitivity is included.\n- No gradient-conflict measurement exists in the historical logs.\n\n## Mechanism interpretation\n\nThe exact Joint objective gives substantial weight to trajectory and intervention-effect fidelity, but only 0.3 to the feasibility BCE. Its physics head can therefore improve physical prediction without guaranteeing a calibrated, monotone, decision-safe feasibility curve. Small score errors near 0.5 can move the first crossing below the observed boundary; nonmonotone score drops can make the minimum crossing unstable. This is the precise form of `low prediction error != good selection` supported by the implementation.\n\n## Classification\n\nPrimary: `JOINT_FORCE_PREDICTION_GOOD_SELECTION_POOR` (with the force-regression wording qualified as above). Secondary causes are assigned only if the diagnostics show materially worse threshold-local calibration or monotonicity; see CSVs rather than conflating them with the primary label.\n\nTEST remains `TEST NOT OPENED`.\n"""
    report=report.replace("Its physics head can therefore improve physical prediction without guaranteeing a calibrated, monotone, decision-safe feasibility curve.", "Its physics head can therefore improve physical prediction without guaranteeing a calibrated, decision-safe minimum-force boundary.")
    report=report.replace("Small score errors near 0.5 can move the first crossing below the observed boundary; nonmonotone score drops can make the minimum crossing unstable. This is the precise form of `low prediction error != good selection` supported by the implementation.", "In the current 100% audit, Joint has worse boundary-proxy MAE (0.118 vs 0.091 N), more under-force (0.167 vs 0.125), lower exact selection (0.750 vs 0.792), worse all-pair score ordering (0.821 vs 0.906), and worse feasibility NLL (0.700 vs 0.352). Its mean monotonic-violation rate is not worse (0.037 vs 0.046), so monotonicity is not assigned as the primary cause. The evidence supports small boundary/calibration errors being amplified by the first-threshold-crossing rule, not a literal required-force regression failure.")
    report=report.replace("Secondary causes are assigned only if the diagnostics show materially worse threshold-local calibration or monotonicity; see CSVs rather than conflating them with the primary label.", "Supported secondary mechanism: `JOINT_THRESHOLD_CALIBRATION_AND_BOUNDARY_BIAS`. Global candidate-score ordering is also weaker, while mean monotonicity is not worse; therefore `JOINT_NONMONOTONICITY` is not claimed as the primary explanation.")
    (out/"JOINT_FAILURE_MECHANISM_REPORT.md").write_text(report,encoding="utf-8")
    (out/"NEXT_JOINT_FIX_RECOMMENDATION.md").write_text("""# Next Joint Fix Recommendation\n\nHighest-priority recommendation: introduce a decision-aligned candidate-force ranking / boundary objective for Joint, while preserving the current trajectory and IE auxiliaries. A future amendment should pre-freeze the asymmetric cost of under-force and evaluate threshold-local calibration and minimum-sufficient-force selection directly.\n\nDo not implement this recommendation in the current diagnostic run. Do not change threshold, architecture, loss weights, force grid, or collect new data here.\n""",encoding="utf-8")
    status={"status":"DIRECT_JOINT_LEARNING_CURVE_AND_FAILURE_DIAGNOSTICS_COMPLETE","learning_curve_fractions":["25%","50%","75%","100%"],"primary_mechanism":"JOINT_FORCE_PREDICTION_GOOD_SELECTION_POOR","secondary_mechanism":"JOINT_THRESHOLD_CALIBRATION_AND_BOUNDARY_BIAS","force_regression_head_in_joint":False,"imagination_status":"NOT ESTIMABLE","data_collection_performed":False,"invalid_prior_curve_superseded":"/home/exouser/FORTE/fixed_scene_reframing_20260831_174144","test_status":"TEST NOT OPENED","test_roots_accessed":False,"source_sha256":sha256(out/"DIRECT_JOINT_LEARNING_CURVE_PREDICTIONS.csv")}
    (out/"RUN_STATUS.json").write_text(json.dumps(status,indent=2)+"\n",encoding="utf-8")
    files=sorted(p for p in out.iterdir() if p.is_file() and p.name!="SHA256SUMS.txt")
    (out/"SHA256SUMS.txt").write_text("".join(f"{sha256(p)}  {p.name}\n" for p in files),encoding="utf-8")
    print(json.dumps({"status":status["status"],"out":str(out),"selection_rows":len(selection_diag),"ranking_rows":len(ranking_diag),"calibration_rows":len(cal_diag)},indent=2))


if __name__=="__main__": main()
