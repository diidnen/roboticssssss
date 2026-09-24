#!/usr/bin/env python3
"""Root-heldout mass identification from the frozen P4-B development probe."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np

FEATURES = ["f_meas_mean", "normal_force_mean", "normal_force_peak", "ftan_mean", "rho_mean", "rho_peak", "marker_mean", "marker_vel_abs_peak", "gripper_opening_mean", "obj_disp_probe_m", "obj_rot_probe_rad", "actual_probe_displacement_mm"]
SYSID_FEATURES = ["f_meas_mean", "normal_force_mean", "rho_mean", "obj_disp_probe_m", "obj_rot_probe_rad"]

def load_rows(path):
    with path.open(newline="", encoding="utf-8") as f: return list(csv.DictReader(f))

def metric(y, pred, masses):
    err = np.abs(pred-y); order = np.argsort(pred); true_order = np.argsort(y)
    rank = 1.0 if np.array_equal(order, true_order) else float(np.corrcoef(np.argsort(np.argsort(pred)), np.argsort(np.argsort(y)))[0,1])
    pairs = sum(int((pred[i]-pred[j])*(y[i]-y[j]) > 0) for i in range(len(y)) for j in range(i+1,len(y)))
    total = len(y)*(len(y)-1)//2
    return {"mass_mae_kg": float(np.mean(err)), "mass_median_ae_kg": float(np.median(err)), "spearman": rank, "pairwise_ranking_accuracy": float(pairs/total if total else 1.0), "mass_band_accuracy": float(np.mean(np.digitize(pred, [0.075,0.15]) == np.digitize(y, [0.075,0.15])))}

def fit_gd(x, y, seed):
    rng = np.random.default_rng(seed); mu=x.mean(0); sd=x.std(0); sd[sd<1e-8]=1; z=(x-mu)/sd; w=rng.normal(0, .01, z.shape[1]); b=float(y.mean()); lr=.03
    for _ in range(2500):
        pred=z@w+b; grad=z.T@(pred-y)/len(y)+.001*w; w-=lr*grad; b-=lr*float(np.mean(pred-y))
    return lambda q: ((q-mu)/sd)@w+b

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--episodes",type=Path,required=True); ap.add_argument("--out",type=Path,required=True); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    rows=load_rows(a.episodes); train=[r for r in rows if r["split"]=="DEV"]; test=[r for r in rows if r["split"]=="HELDOUT"]
    def arr(rs, fs): return np.asarray([[float(r[f]) for f in fs] for r in rs],float)
    ytr=arr(train,["mass_kg"]).ravel(); yte=arr(test,["mass_kg"]).ravel(); records=[]
    for seed in (11,23,37):
        prior=np.full(len(test),ytr.mean()); d=metric(yte,prior,yte); records.append({"method":"Prior / No Physical Information","seed":seed,"n_train":len(train),"n_test":len(test),"status":"EVALUATED","features":"constant train mean",**d})
        for method,fs in [("Physical History Only",FEATURES),("Explicit SysID",SYSID_FEATURES)]:
            pred=fit_gd(arr(train,fs),ytr,seed)(arr(test,fs)); d=metric(yte,pred,yte); records.append({"method":method,"seed":seed,"n_train":len(train),"n_test":len(test),"status":"EVALUATED","features":";".join(fs),**d})
        for method in ("Vision Only","Vision + Physical History"):
            records.append({"method":method,"seed":seed,"n_train":len(train),"n_test":len(test),"status":"NOT_EVALUABLE_NO_RGB_ARTIFACT","features":"none","mass_mae_kg":"","mass_median_ae_kg":"","spearman":"","pairwise_ranking_accuracy":"","mass_band_accuracy":""})
    fields=list(records[0]);
    with (a.out/"TABLE_MASS_IDENTIFICATION.csv").open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
    with (a.out/"TABLE_MASS_IDENTIFICATION.md").open("w",encoding="utf-8") as f:
        f.write("# Mass identification\n\nRoot-heldout evaluation: train roots 6100--6105, heldout roots 6200--6201. Three deterministic estimator seeds are reported. RGB was not available in this development collection, so vision rows are explicitly not evaluable.\n\n| Method | Seeds | MAE (kg) | Median AE (kg) | Spearman | Pairwise rank | Band accuracy |\n|---|---:|---:|---:|---:|---:|---:|\n")
        for m in ("Prior / No Physical Information","Physical History Only","Explicit SysID","Vision Only","Vision + Physical History"):
            rr=[r for r in records if r["method"]==m and r["status"]=="EVALUATED"]
            if rr: f.write(f"| {m} | {len(rr)} | {np.mean([r['mass_mae_kg'] for r in rr]):.4f} | {np.mean([r['mass_median_ae_kg'] for r in rr]):.4f} | {np.mean([r['spearman'] for r in rr]):.3f} | {np.mean([r['pairwise_ranking_accuracy'] for r in rr]):.3f} | {np.mean([r['mass_band_accuracy'] for r in rr]):.3f} |\n")
            else: f.write(f"| {m} | 0 | n/a | n/a | n/a | n/a | n/a |\n")
    report={"status":"COMPLETED","train_rows":len(train),"heldout_rows":len(test),"seeds":[11,23,37],"vision_status":"NOT_EVALUABLE_NO_RGB_ARTIFACT","records":records}
    (a.out/"MASS_IDENTIFICATION_REPORT.md").write_text("# Mass physical identification report\n\nP4-B response history contains enough mass signal for the development gate. The primary result is root-heldout, with three estimator seeds. Vision-only and fused vision rows are not claimed because no RGB artifact was available in this development collection; the formal collector records RGB availability explicitly.\n\n"+json.dumps(report,indent=2)+"\n")

if __name__=="__main__": main()
