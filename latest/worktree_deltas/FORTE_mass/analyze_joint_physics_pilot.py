#!/usr/bin/env python3
"""Root-heldout CPU analysis for the joint friction/mass pilot."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np

MUS = np.array([0.25, 0.50, 0.75]); MASSES = np.array([0.05, 0.10, 0.20]); FORCES = np.array([0.5, 1.0, 1.5, 2.5, 4.0])

def val(r, k):
    try:
        x = float(r.get(k, 0)); return x if np.isfinite(x) else 0.0
    except (TypeError, ValueError): return 0.0

def features(r):
    return np.array([val(r, k) for k in ("f_meas_mean", "normal_force_mean", "normal_force_peak", "normal_force_hyst", "ftan_mean", "ftan_peak", "rho_mean", "rho_peak", "rho_impulse", "imb_ratio_mean", "marker_mean", "marker_peak", "marker_hyst", "gripper_opening_mean")], dtype=float)

def fit(x, y):
    mu=x.mean(0); sd=x.std(0); sd[sd<1e-9]=1; X=np.c_[np.ones(len(x)),(x-mu)/sd]; w=np.linalg.solve(X.T@X+.01*np.eye(X.shape[1]), X.T@y); return lambda q: np.c_[np.ones(len(q)),(q-mu)/sd]@w

def band(pred, levels): return levels[np.argmin(np.abs(pred[:,None]-levels[None,:]), axis=1)]
def proxy_force(mu, mass): return float(FORCES[np.argmin(np.abs(FORCES-(1.0+2.0*mu+5.0*mass)))])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",type=Path,required=True); ap.add_argument("--out",type=Path,required=True); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    with (a.input/"JOINT_PHYSICS_PILOT_EPISODES.csv").open(newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
    train=[r for r in rows if r["split"]=="TRAIN" and int(r["query_valid"])]; test=[r for r in rows if r["split"]=="TEST" and int(r["query_valid"])]
    Xtr=np.asarray([features(r) for r in train]); Xte=np.asarray([features(r) for r in test]); ymu_tr=np.asarray([val(r,"friction") for r in train]); ym_tr=np.asarray([val(r,"mass_kg") for r in train]); ymu=np.asarray([val(r,"friction") for r in test]); ym=np.asarray([val(r,"mass_kg") for r in test])
    # Frozen diagnostic baselines: single-parameter regressors are deliberately
    # restricted to one target; the joint model predicts both targets.
    models={"friction-only":(fit(Xtr,ymu_tr),None),"mass-only":(None,fit(Xtr,ym_tr)),"joint":(fit(Xtr,ymu_tr),fit(Xtr,ym_tr))}
    metric_rows=[]; detail=[]
    for name,(fm,mm) in models.items():
        pm=fm(Xte) if fm else np.full(len(test), .5); pma=mm(Xte) if mm else np.full(len(test), .1)
        mu_mae=float(np.mean(np.abs(pm-ymu))); mass_mae=float(np.mean(np.abs(pma-ym)))
        mu_band=float(np.mean(band(pm,MUS)==ymu)); mass_band=float(np.mean(band(pma,MASSES)==ym))
        # For the single-target baselines, the unestimated nuisance parameter
        # is held at the training prior before selecting from the frozen set.
        mu_for_force=pm if fm else np.full(len(test),.5); mass_for_force=pma if mm else np.full(len(test),.1)
        choices=np.asarray([proxy_force(x,y) for x,y in zip(mu_for_force,mass_for_force)])
        truth=np.asarray([proxy_force(x,y) for x,y in zip(ymu,ym)])
        agreement=float(np.mean(choices==truth))
        metric_rows.append({"estimator":name,"n_train":len(train),"n_test":len(test),"friction_mae":mu_mae,"mass_mae_kg":mass_mae,"friction_band_accuracy":mu_band,"mass_band_accuracy":mass_band,"force_choice_agreement_proxy":agreement})
        for r,x,y,a1,b1 in zip(test,pm,pma,choices,truth): detail.append({"trial_id":r["trial_id"],"estimator":name,"true_friction":r["friction"],"pred_friction":x,"true_mass_kg":r["mass_kg"],"pred_mass_kg":y,"selected_force_proxy_N":a1,"oracle_force_proxy_N":b1,"force_choice_agreement":int(a1==b1)})
    with (a.out/"TABLE_JOINT_PHYSICS_PILOT.csv").open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=list(metric_rows[0]));w.writeheader();w.writerows(metric_rows)
    with (a.out/"JOINT_PHYSICS_PILOT_PREDICTIONS.csv").open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=list(detail[0]));w.writeheader();w.writerows(detail)
    joint=next(x for x in metric_rows if x["estimator"]=="joint")
    status="JOINT_IDENTIFIABLE_WITH_CURRENT_QUERY" if joint["friction_band_accuracy"]>=.80 and joint["mass_band_accuracy"]>=.80 else "JOINT_PHYSICS_NOT_IDENTIFIABLE_WITH_CURRENT_QUERY"
    report={"status":status,"query":"frozen P4-B","grid":"3 friction x 3 mass","heldout_root_evaluation":True,"metrics":metric_rows,"force_choice_definition":"diagnostic analytic proxy only; not a full-task outcome"}
    (a.out/"JOINT_PHYSICS_PILOT_REPORT.md").write_text("# Joint friction × mass identifiability pilot\n\nThis is a diagnostic pilot. It does not alter the friction-only or mass-only protocols and does not use downstream outcomes. Force-choice agreement is against a pre-registered analytic proxy, not a task success label.\n\n"+json.dumps(report,indent=2)+"\n")
    (a.out/"JOINT_PHYSICS_PILOT_PROTOCOL.json").write_text(json.dumps({"status":status,"grid":"3x3","friction_values":MUS.tolist(),"mass_values_kg":MASSES.tolist(),"split":"train roots vs heldout root","source":"JOINT_PHYSICS_PILOT_EPISODES.csv"},indent=2)+"\n")

if __name__=="__main__": main()
