#!/usr/bin/env python3
"""Analyze formal structured Hidden-Mass branches and force adaptation."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np

FEATURES = ["f_meas_mean_N", "measured_force_peak_N", "steps"]
BANDS = {"LOW": 0.05, "MID": 0.10, "HIGH": 0.20}
FORCES = [0.5, 1.0, 1.5, 2.5, 4.0]

def read(path):
    with path.open(newline="", encoding="utf-8") as f: return list(csv.DictReader(f))

def fit(x, y):
    mu=x.mean(0); sd=x.std(0); sd[sd<1e-8]=1; z=(x-mu)/sd; X=np.c_[np.ones(len(z)),z]; w=np.linalg.solve(X.T@X+.01*np.eye(X.shape[1]), X.T@y)
    return lambda q: np.c_[np.ones(len(q)),(q-mu)/sd]@w

def metrics(rs):
    if not rs: return {}
    def mean(k): return float(np.mean([float(r[k]) for r in rs]))
    return {"n":len(rs),"full_task_sr":mean("corrected_full"),"lift_sr":mean("lift_success"),"transport_retention_sr":mean("transport_retention"),"placement_sr":mean("place_success"),"mean_force_N":mean("requested_force_N"),"under_force_rate":mean("under_force"),"excess_force_rate":mean("excess_force"),"delayed_failure_rate":float(np.mean([int(float(r["lift_success"])==1 and float(r["corrected_full"])==0) for r in rs]))}

def summary_row(policy, rs, scope, group):
    m = metrics(rs)
    roots = sorted({str(r.get("root_seed", "")) for r in rs})
    root_means = [float(np.mean([float(r["corrected_full"]) for r in rs if str(r.get("root_seed", "")) == root])) for root in roots]
    ci = 1.96 * float(np.std(root_means, ddof=1)) / max(len(root_means) ** 0.5, 1.0) if len(root_means) > 1 else 0.0
    m.update({"policy": policy, "scope": scope, "group": group, "n_root_clusters": len(roots), "full_task_sr_ci95": ci})
    m["force_choice_agreement_vs_gt"] = float(np.mean([r["selected_force_N"] == r["oracle_force_N"] for r in rs])) if rs else ""
    return m

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--inputs",type=Path,nargs="+",required=True); ap.add_argument("--out",type=Path,required=True); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    contexts=[]; branches=[]
    for d in a.inputs:
        cs=read(d / next(d.glob("M3_TASK*_STRUCTURED_FORMAL_CONTEXTS.csv")).name); bs=read(d / next(d.glob("M3_TASK*_STRUCTURED_FORMAL_BRANCHES.csv")).name)
        contexts += cs; branches += bs
    for r in branches:
        r["corrected_full"]=str(int(float(r["lift_success"]) and float(r["transport_retention"]) and float(r["place_success"]) and not float(r["dropped"])))
    train_c=[r for r in contexts if r["split"]=="TRAIN" and int(r["query_valid"])]
    test_c=[r for r in contexts if r["split"]=="TEST" and int(r["query_valid"])]
    def qfeat(c):
        try: q=json.loads(c["query_record"])
        except Exception: q={}
        return [float(q.get("f_meas_mean",q.get("f_meas_mean_N",0))),float(q.get("normal_force_peak",q.get("normal_force_peak_N",0))),float(c.get("query_history_rows",0))]
    pred=fit(np.asarray([qfeat(c) for c in train_c]),np.asarray([float(c["mass_kg"]) for c in train_c])) if train_c else (lambda q: np.full(len(q),.1))
    train_bs=[r for r in branches if r["split"]=="TRAIN"]
    utility={}
    for band in BANDS:
        for force in FORCES:
            rr=[r for r in train_bs if r["mass_band"]==band and abs(float(r["requested_force_N"])-force)<1e-6]
            utility[(band,force)] = float(np.mean([float(r["corrected_full"]) for r in rr])) if rr else 0.0
    def choose(band):
        vals=[(utility.get((band,f),0)-.002*f,f) for f in FORCES]; return max(vals)[1]
    bmap={}
    for r in branches: bmap.setdefault((r["context_id"],float(r["requested_force_N"])),[]).append(r)
    policy_rows=[]
    for c in test_c:
        pred_mass=float(pred(np.asarray([qfeat(c)]))[0]); est_band=min(BANDS,key=lambda b:abs(BANDS[b]-pred_mass)); truth=c["mass_band"]
        picks={"Frozen VLA Default":1.0,"Fixed-Max":4.0,"No-Physical-Information":choose("MID"),"ActiveForcing-Mass":choose(est_band),"GT-Mass + Direct":choose(truth)}
        avail=[f for f in FORCES if any(x["corrected_full"]=="1" for x in bmap.get((c["context_id"],f),[]))]; oracle=min(avail) if avail else max(FORCES)
        picks["Hindsight Grid Oracle"]=oracle
        for policy,force in picks.items():
            for br in bmap.get((c["context_id"],force),[]):
                x=dict(br); x.update({"policy":policy,"pred_mass_kg":pred_mass,"estimated_band":est_band,"truth_band":truth,"selected_force_N":force,"oracle_force_N":oracle,"under_force":int(force<oracle),"excess_force":int(force>oracle)}); policy_rows.append(x)
    fields=list(policy_rows[0]) if policy_rows else ["status"]
    with (a.out/"TABLE_MASS_FORCE_ADAPTATION.csv").open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(policy_rows)
    summary=[]
    for pol in ("Frozen VLA Default","Fixed-Max","No-Physical-Information","ActiveForcing-Mass","GT-Mass + Direct","Hindsight Grid Oracle"):
        rr=[r for r in policy_rows if r["policy"]==pol]; summary.append(summary_row(pol, rr, "overall", "ALL"))
        for task_id in sorted({r["task_id"] for r in rr}): summary.append(summary_row(pol,[r for r in rr if r["task_id"]==task_id],"per_task",str(task_id)))
        for band in ("LOW","MID","HIGH"): summary.append(summary_row(pol,[r for r in rr if r["mass_band"]==band],"per_mass_band",band))
    sf=list(summary[0]) if summary else ["policy"]
    with (a.out/"TABLE_MASS_FORCE_ADAPTATION_SUMMARY.csv").open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=sf);w.writeheader();w.writerows(summary)
    with (a.out/"TABLE_MASS_FORCE_ADAPTATION.md").open("w",encoding="utf-8") as f:
        f.write("# Mass force adaptation\n\nPaired TEST-root evaluation from the formal structured dataset. Under/excess force are defined against the lowest candidate force that succeeds for the same context (hindsight diagnostic).\n\n| Policy | n | Full-task SR | Lift SR | Transport SR | Placement SR | Mean force | Under-force | Excess-force | Delayed failure | Force agreement |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in [x for x in summary if x["scope"]=="overall"]: f.write(f"| {r['policy']} | {r.get('n',0)} | {r.get('full_task_sr',''):.3f} | {r.get('lift_sr',''):.3f} | {r.get('transport_retention_sr',''):.3f} | {r.get('placement_sr',''):.3f} | {r.get('mean_force_N',''):.3f} | {r.get('under_force_rate',''):.3f} | {r.get('excess_force_rate',''):.3f} | {r.get('delayed_failure_rate',''):.3f} | {r.get('force_choice_agreement_vs_gt',''):.3f} |\n")
    report={"status":"COMPLETED","train_contexts":len(train_c),"test_contexts":len(test_c),"formal_branches":len(branches),"methods":summary,"mass_identifier":"physical query history ridge","ci95":"1.96*SD(root-cluster means)/sqrt(number of roots)"}
    (a.out/"MASS_FORCE_ADAPTATION_REPORT.md").write_text("# Mass force adaptation report\n\nThe formal comparison uses the same task, root, mass, initial post-query state, candidate forces, and repeats for every policy. Query selection is made without downstream branch outcomes.\n\n"+json.dumps(report,indent=2)+"\n")
    # Full-task vs local-lift is a direct label-matched contrast over all formal branches.
    fl=[]
    for r in branches:
        fl.append({"task_id":r["task_id"],"split":r["split"],"mass_band":r["mass_band"],"requested_force_N":r["requested_force_N"],"repeat":r["repeat"],"lift_success":r["lift_success"],"full_task_success":r["corrected_full"],"post_lift_failure":int(float(r["lift_success"])==1 and float(r["corrected_full"])==0),"transport_retention":r["transport_retention"],"placement_success":r["place_success"]})
    lf=list(fl[0]) if fl else ["status"]
    with (a.out/"TABLE_MASS_FULLTASK_VS_LOCALLIFT.csv").open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=lf);w.writeheader();w.writerows(fl)
    (a.out/"MASS_FULLTASK_LOCALLIFT_REPORT.md").write_text("# Mass FullTask vs LocalLift\n\nThe matched formal branches use identical query, architecture, roots, force candidates, and repeats; only the label changes from lift success to corrected full-task success. Delayed post-lift failures are retained explicitly.\n\n"+json.dumps({"branches":len(fl),"lift_sr":float(np.mean([float(x["lift_success"]) for x in fl])) if fl else None,"full_task_sr":float(np.mean([float(x["full_task_success"]) for x in fl])) if fl else None,"post_lift_failure_rate":float(np.mean([float(x["post_lift_failure"]) for x in fl])) if fl else None},indent=2)+"\n")

if __name__=="__main__": main()
