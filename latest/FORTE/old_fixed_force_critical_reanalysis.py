#!/usr/bin/env python3
"""Outcome-only strict force-critical reanalysis of the old fixed-scene DEV."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/home/exouser/FORTE")
OUT = ROOT / "existing_force_critical_benchmark_20260901_052000"
PRED = ROOT / "joint_decision_alignment_20260831_200441/JOINT_DECISIONALIGNED_ALL_PREDICTIONS.csv"
SEL = ROOT / "direct_worldmodel_verifier_20260901_020334/DIRECT_WORLD_MODEL_VERIFIER_SELECTIONS.csv"
HARD = ROOT / "direct_worldmodel_verifier_20260901_020334/DIRECT_WORLD_MODEL_HARD_SEARCH_SELECTIONS.csv"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    OUT.mkdir(exist_ok=True)
    d = pd.read_csv(PRED)
    # Membership uses only identity, force, repeat and actual outcome columns.
    d = d[(d.method == "Direct") & (d.fraction == "100%")][["context_id", "task", "force_N", "repeat", "actual_success", "scene_id"]].copy()
    cases = []
    for cid, q0 in d.groupby("context_id", sort=True):
        cells = q0.groupby("force_N", as_index=False).agg(n=("actual_success", "size"), ymin=("actual_success", "min"), ymax=("actual_success", "max")).sort_values("force_N")
        for i in range(len(cells)-1):
            lo, hi = cells.iloc[i], cells.iloc[i+1]
            if lo.n == 2 and hi.n == 2 and lo.ymax == 0 and hi.ymin == 1:
                cases.append({"context_id": cid, "task": int(q0.task.iloc[0]), "scene_id": str(q0.scene_id.iloc[0]), "F_low_N": float(lo.force_N), "F_high_N": float(hi.force_N), "selection_used_model_output": False}); break
    if len(cases) != 6:
        raise RuntimeError(f"expected 6 strict cases, got {len(cases)}")
    manifest = {
        "status": "RETROSPECTIVE_OUTCOME_ONLY_MEMBERSHIP",
        "rule": "adjacent force cells; both F_low repeats fail and both F_high repeats succeed",
        "cases": cases, "counts": {"contexts": 6, "per_task": pd.Series([x["task"] for x in cases]).value_counts().sort_index().astype(int).to_dict()},
        "model_output_used_for_membership": False, "source_sha256": sha(PRED),
        "caveat": "old fixed-scene DEV already viewed; retrospective diagnostic only",
    }
    mp = OUT / "OLD_FIXED_FORCE_CRITICAL_MANIFEST.json"; mp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    ids = {x["context_id"] for x in cases}
    direct = pd.read_csv(SEL); direct = direct[(direct.policy == "DIRECT_UTILITY") & direct.context_id.isin(ids)].set_index("context_id")
    hard = pd.read_csv(HARD); hard = hard[hard.context_id.isin(ids)]
    rows = []
    percase = []
    for case in cases:
        cid = case["context_id"]; q = d[d.context_id == cid].groupby("force_N", as_index=False).actual_success.mean().sort_values("force_N")
        dr = direct.loc[cid]; di = int(np.argmin(np.abs(q.force_N.to_numpy()-float(dr.selected_force_N))))
        one = q.iloc[min(di+1, len(q)-1)]; mx = q.iloc[-1]
        percase += [
            {**case, "method": "Direct", "selected_force_N": float(dr.selected_force_N), "actual_success_rate": float(dr.actual_success_rate), "coverage": 1},
            {**case, "method": "One-Step", "selected_force_N": float(one.force_N), "actual_success_rate": float(one.actual_success), "coverage": 1},
            {**case, "method": "Fixed-Max", "selected_force_N": float(mx.force_N), "actual_success_rate": float(mx.actual_success), "coverage": 1},
        ]
    hmap = {(r.context_id, r.policy): r for _, r in hard.iterrows()}
    for case in cases:
        cid = case["context_id"]
        for policy, name in [("HARD_SEARCH_STRICT_FINITE_ONLY", "Strict-Verifier"), ("HARD_SEARCH_WITH_EXPLICIT_MAX_FALLBACK", "Verifier-MaxFallback")]:
            r = hmap.get((cid, policy))
            percase.append({**case, "method": name, "selected_force_N": float(r.selected_force_N) if r is not None else np.nan, "actual_success_rate": float(r.actual_success_rate) if r is not None else 0.0, "coverage": int(r is not None), "changed_from_direct": int(r.changed_from_direct) if r is not None else np.nan})
    pc = pd.DataFrame(percase); pc.to_csv(OUT / "OLD_FIXED_FORCE_CRITICAL_PER_CASE.csv", index=False)
    for method, q in pc.groupby("method", sort=False):
        rows.append({"method": method, "contexts": len(q), "coverage": float(q.coverage.mean()), "unconditional_SR": float((q.actual_success_rate*q.coverage).mean()), "conditional_SR": float(q.loc[q.coverage==1, "actual_success_rate"].mean()), "mean_force_N": float(q.loc[q.coverage==1, "selected_force_N"].mean())})
    table = pd.DataFrame(rows); table.to_csv(OUT / "OLD_FIXED_FORCE_CRITICAL_TABLE.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
