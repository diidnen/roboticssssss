#!/usr/bin/env python3
"""Consolidate per-seed fixed-scene predictions into frozen ensembles."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--audit", type=Path, required=True); args = ap.parse_args(); out = args.audit
    d = pd.read_csv(out / "FIXED_SCENE_DEV_PREDICTIONS.csv")
    rows = []
    for (method, branch_id), g in d.groupby(["method", "branch_id"], sort=True):
        r = g.iloc[0].to_dict(); r["p_success"] = float(g.p_success.mean()); r["seed_ensemble_n"] = int(len(g)); rows.append(r)
    e = pd.DataFrame(rows)
    results = []
    for (method, cid), g in e.groupby(["method", "context_id"], sort=True):
        g = g.sort_values("force_N"); candidates = g[g.p_success >= 0.5]
        chosen = candidates.iloc[0] if len(candidates) else g.iloc[-1]
        q = g[abs(g.force_N - float(chosen.force_N)) < 1e-7]
        results.append({
            "method": "ActiveForcing-Direct" if method == "FEAS" else "ActiveForcing-Joint",
            "backend": method, "context_id": cid, "scene_id": chosen.scene_id,
            "task": int(chosen.task), "mu": float(chosen.mu), "selected_force_N": float(chosen.force_N),
            "threshold": 0.5, "fallback_used": int(len(candidates) == 0),
            "full_task_success_rate_at_selected_force": float(q.actual_success.mean()),
            "n_repeats_at_selected_force": int(len(q)), "seed_ensemble_n": int(chosen.seed_ensemble_n),
        })
    e["method"] = e.method.map({"FEAS":"ActiveForcing-Direct", "JOINT":"ActiveForcing-Joint"})
    write_csv(out / "FIXED_SCENE_DEV_ENSEMBLE_PREDICTIONS.csv", e.to_dict("records"))
    write_csv(out / "FIXED_SCENE_DEV_RESULTS.csv", results)
    by = pd.DataFrame(results)
    summary = []
    for method, g in by.groupby("method", sort=True):
        summary.append({"method":method,"contexts":len(g),"full_task_DEV_SR":float(g.full_task_success_rate_at_selected_force.mean()),"fallback_rate":float(g.fallback_used.mean()),"mean_selected_force_N":float(g.selected_force_N.mean()),"threshold":0.5})
    write_csv(out / "FIXED_SCENE_METHOD_DEV_SUMMARY.csv", summary)
    train_rows = pd.read_csv(out / "FIXED_SCENE_TRAIN_MANIFEST.csv")
    dev_rows = pd.read_csv(out / "FIXED_SCENE_DEV_MANIFEST.csv")
    for name, q, split in [("FIXED_SCENE_TRAIN_MANIFEST.json", train_rows, "TRAIN"),("FIXED_SCENE_DEV_MANIFEST.json", dev_rows, "DEV")]:
        (out / name).write_text(json.dumps({"split":split,"rows":len(q),"contexts":int(q.context_id.nunique()),"scene_families":sorted(q.scene_id.unique()),"source_csv":str(out/name.replace('.json','.csv')) if False else str(out/("FIXED_SCENE_TRAIN_MANIFEST.csv" if split=="TRAIN" else "FIXED_SCENE_DEV_MANIFEST.csv")),"context_level_split":True,"test_used":False,"rows_data":q.to_dict("records")}, indent=2, default=str)+"\n", encoding="utf-8")
    (out / "DIRECT_TRAINING_AUDIT.md").write_text("""# Direct Training Audit\n\nStatus: `COMPLETE`.\n\nThe existing `FeasibilityOnly` implementation from `gnp_style_continuous.py` was trained on 480 TRAIN branches from the fixed-scene common dataset, using H=8 x 71 input, TRAIN-only normalization, BCE feasibility loss, AdamW 8e-4/1e-4, 80 epochs, and seeds 0/1/2. DEV contains 24 held-out friction contexts and was not used for training or checkpoint selection. Online threshold remains 0.5; this audit does not apply rho_frontier.\n""", encoding="utf-8")
    (out / "JOINT_TRAINING_AUDIT.md").write_text("""# Joint Training Audit\n\nStatus: `COMPLETE`.\n\nThe existing `JointIEFeasibility` implementation was trained on the same 480 TRAIN branches and 384 adjacent-force IE pairs (48 contexts x 2 repeats x 4 adjacent pairs). It uses the frozen H=8 Physics-GRU initialization, physical trajectory + IE + 0.3 feasibility objective, AdamW 8e-4/1e-4, 80 epochs, and seeds 0/1/2. DEV is held out at complete friction-context level.\n\nJoint remains the historical Direct+Imagination/physics-auxiliary comparison method; it is not visual/probe fusion and is not substituted for Direct online semantics.\n""", encoding="utf-8")
    (out / "IMAGINATION_TRAINING_AUDIT.md").write_text("""# Imagination Training Audit\n\nStatus: `BLOCKED_BY_EXACT_IMPLEMENTATION_ADAPTER`.\n\nThe archived exact ActiveForcing-Imagination entry is `/home/exouser/Tabero/analysis/active_friction_imagination.py`. Its frozen protocol consumes the separate 144-context P5-S0-C estimator population and performs probe -> friction belief -> deterministic snapshot/physics imagination. The archived learned trajectory-imagination entry consumes a different 576-branch/144-context historical population and has no direct common-720 force-selector adapter.\n\nNo new imagination architecture, target, calibration, or selector was invented to force an apparent three-way comparison. The fixed-scene Direct and Joint models are therefore complete; ActiveForcing-Imagination is `NOT ESTIMABLE` for this fixed-scene run until an exact, pre-frozen adapter is specified. This is an implementation blocker only and does not open TEST.\n""", encoding="utf-8")
    s = pd.DataFrame(summary)
    comp = """# Fixed-Scene Method Comparison\n\nScope: retrospective fixed-scene held-out-friction DEV labels only; no formal TEST and no simulator closed loop. The final primary metric is full-task success rate at the selected force. Training losses are diagnostics.\n\n"""
    if len(s): comp += s.to_csv(index=False) + "\n"
    comp += "ActiveForcing-Imagination: NOT ESTIMABLE — exact archived implementation has no common-720 fixed-scene adapter; no substitute method was introduced.\n\nDirect and Joint use the same 0.5 decision threshold in this diagnostic. `rho_frontier=0.8` is not used for online selection.\n"
    (out / "FIXED_SCENE_METHOD_COMPARISON.md").write_text(comp, encoding="utf-8")
    print(json.dumps({"status":"CONSOLIDATED","out":str(out),"ensemble_rows":len(e),"contexts_per_method":int(by.groupby('method').size().min()),"summary":summary}, indent=2))


if __name__ == "__main__": main()
